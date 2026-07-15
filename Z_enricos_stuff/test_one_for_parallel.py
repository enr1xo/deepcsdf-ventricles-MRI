#!/usr/bin/env python3

"""
This code is the analogous of train_one_for_parallel, it tests the DeepSDF model for one combination forlder

Expected combination structure:
    combo_dir/
    ├── config.py
    ├── specs_files/
    │   └── specs.json
    ├── train/
    │   └── data_fnames_train.json
    ├── test/
    │   └── data_fnames_test.json
    ├── experiments/
    └── results/
"""

import sys
from pathlib import Path
import importlib.util
import json
import math
import argparse
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import pyvista as pv
import pandas as pd
from pprint import pprint
from tqdm import tqdm
from scipy.interpolate import griddata
from vtk import vtkImplicitPolyDataDistance

from model.deepsdf_decoder import Decoder, DeepSDF
from model.deepsdf_dataloader import SDFDataModule
from utils.metrics import chamfer_distance_L2, LDDMM_loss, haussdorff, f1_score_function, compute_dice_score, sdf_gt_on_regular_grid
from utils.surface_utils import remesh, make_trimesh_from_pv
from utils.reconstruction_utils import isosurface_from_sdf
from utils.visual_utils import plot_gt_vs_reconstructed_with_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using GPU:", torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else "CPU")


# ======================== #
# Helpers
# ======================== #
def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_dataset_patients_names(data: list[str]):
    patient_names = []

    for fullfname in data:
        fname = Path(fullfname).name

        if "-epi_lv_rv_" in fname:
            patient_name = fname.split("-epi_lv_rv_")[0]

        elif "_MRI_like_" in fname:
            patient_name = fname.split("_MRI_like_")[0]

        elif fname.endswith("_three_axis_mri_samples.npy"):
            # patient_name = fname.replace("_three_axis_mri_samples.npy", "")
            patient_name = fname.removesuffix("_three_axis_mri_samples.npy")

        elif fname.endswith("_mri_samples.npy"):
            patient_name = fname.replace("_mri_samples.npy", "")
        
        

        else:
            patient_name = Path(fname).stem
        
        patient_names.append(patient_name)

    return patient_names


def save_metrics_csv(metric_name, experiment_name, version, which_shapes, metric_data: dict, metrics_dir: Path):
    rows = []
    for name, organs in metric_data.items():
        for organ, metric in organs.items():
            rows.append({
                "version": int(version.split("_")[-1]),
                "patient": name,
                "organ": organ,
                "metric": metric_name,
                "value": metric,
            })
    df = pd.DataFrame(rows)

    filename = f"{experiment_name}-{version}-{metric_name}-{which_shapes}.csv"

    if which_shapes == "train":
        output_path = metrics_dir / experiment_name / "train_data" / filename
    else:
        output_path = metrics_dir / experiment_name / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_latents_npz(experiment_name, version, which_shapes, code_reg_lambda, num_epochs, init_from, loss_type,
                     latent_codes: dict, latents_dir: Path):
    fname = (
        f"{experiment_name}-{version}-latents_{len(set(latent_codes.keys()))}_{which_shapes}_patients"
        f"-codereg={code_reg_lambda:.0e}-epochs={num_epochs}-init={init_from}-loss={loss_type}.npz"
    )
    output_path = latents_dir / experiment_name / fname
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **latent_codes)


def find_pointcloud_noise_old(
    decoder: Decoder,
    model: DeepSDF,
    xyz_gt,
    sdf_gt,
    num_epochs_fit_latent,
    lr_fit_latent,
    code_reg_lambda,
    max_iter=10,
):
    latent_size = decoder.latent_size
    mean_code = torch.zeros(latent_size, device=DEVICE)
    latent = mean_code
    latent.requires_grad = True

    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(params=[latent], lr=lr_fit_latent)
    num_samp_per_scene = sdf_gt.shape[0]

    epsilon = 0.0
    epsilons = [epsilon]

    for _ in range(max_iter):
        mean_code = torch.zeros(latent_size, device=DEVICE, requires_grad=True)
        latent = mean_code
        optimizer = torch.optim.Adam(params=[latent], lr=lr_fit_latent)

        for i in range(num_epochs_fit_latent):
            decoder.eval()
            optimizer.zero_grad()

            batch_vecs = latent.expand(num_samp_per_scene, -1)
            input_ = torch.cat([batch_vecs, xyz_gt], dim=1)

            sdf_pred = decoder(input_)
            if model.enforce_minmax:
                sdf_pred = torch.clamp(sdf_pred, min=-model.clamp_distance, max=model.clamp_distance)

            reg_loss = torch.linalg.norm(latent) ** 2
            recon_loss = loss_fn(sdf_pred, sdf_gt)
            chunk_loss = recon_loss / num_samp_per_scene
            loss = chunk_loss + 100 * epsilon * code_reg_lambda * reg_loss

            loss.backward()
            optimizer.step()

            if i == num_epochs_fit_latent - 1:
                epsilon = np.sqrt(recon_loss.detach().item() / (num_samp_per_scene - 1))
                epsilons.append(epsilon)

        tol = 1e-7
        if abs(epsilons[-1] ** 2 - epsilons[-2] ** 2) < tol:
            break

    return epsilons[-1]


def find_pointcloud_noise(
    decoder: Decoder,
    model: DeepSDF,
    xyz_gt,
    sdf_gt,
    mask_gt,
    num_epochs_fit_latent,
    lr_fit_latent,
    code_reg_lambda,
    loss_type="MSE",
    max_iter=10,
):
    """
    Stima epsilon usando esclusivamente le SDF valide secondo mask_gt.

    La loss usata qui è coerente con quella impiegata successivamente
    per il fitting del latent code.
    """

    latent_size = decoder.latent_size
    num_samp_per_scene = sdf_gt.shape[0]

    epsilon = 0.0
    epsilons = [epsilon]

    for _ in range(max_iter):

        latent = torch.zeros(
            latent_size,
            device=DEVICE,
            requires_grad=True,
        )

        optimizer = torch.optim.Adam(
            params=[latent],
            lr=lr_fit_latent,
        )

        last_recon_loss = None

        for epoch in range(num_epochs_fit_latent):
            decoder.eval()
            optimizer.zero_grad()

            batch_vecs = latent.expand(num_samp_per_scene, -1)
            input_ = torch.cat([batch_vecs, xyz_gt], dim=1)

            sdf_pred = decoder(input_)

            if model.enforce_minmax:
                sdf_pred = torch.clamp(
                    sdf_pred,
                    min=-model.clamp_distance,
                    max=model.clamp_distance,
                )

            recon_loss = masked_regression_loss(
                pred=sdf_pred,
                gt=sdf_gt,
                mask=mask_gt,
                loss_type=loss_type,
            )

            reg_loss = latent.pow(2).sum()

            loss = (
                recon_loss
                + 100.0
                * epsilon
                * code_reg_lambda
                * reg_loss
            )

            loss.backward()
            optimizer.step()

            last_recon_loss = recon_loss

        if last_recon_loss is None:
            raise RuntimeError(
                "No reconstruction loss was computed "
                "inside find_pointcloud_noise."
            )

        # masked_regression_loss restituisce già un errore medio.
        # Per una MSE, epsilon è la RMSE.
        epsilon_new = torch.sqrt(
            torch.clamp(last_recon_loss.detach(), min=0.0)
        ).item()

        epsilons.append(epsilon_new)

        tol = 1e-7

        if abs(epsilons[-1] ** 2 - epsilons[-2] ** 2) < tol:
            epsilon = epsilon_new
            break

        epsilon = epsilon_new

    return epsilon


def get_latest_version_dir(exp_dir: Path) -> Path:
    version_dirs = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("version_")]
    if not version_dirs:
        raise FileNotFoundError(f"No version_* directories found in {exp_dir}")
    return max(version_dirs, key=lambda p: int(p.name.split("_")[-1]))



def masked_regression_loss(pred, gt, mask, loss_type="MSE"):
    if loss_type == "L1":
        err = torch.abs(pred - gt)
    elif loss_type == "MSE":
        err = (pred - gt) ** 2
    elif loss_type == "SmoothL1":
        err = torch.nn.functional.smooth_l1_loss(
            pred,
            gt,
            reduction="none",
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss = 0.0

    for j in range(pred.shape[1]):
        denom = mask[:, j].sum() + 1e-8
        loss_j = (err[:, j] * mask[:, j]).sum() / denom
        loss = loss + loss_j

    loss = loss / pred.shape[1]

    return loss

# ======================== #
# RUN TESTS
# ======================== #
def run(
    combo_dir: Path,
    combo_config,
    experiment_name: str,
    version: str | None = None,
    override_with_dataset: str | None = None,
    num_samp_per_scene_for_fit: int | None = None,
    hparams_file: Path | None = None,
    num_epochs_fit_latent: int = 250,
    latent_reg_factor: float | None = None,
    lr_fit_latent: float = 5e-3,
    initialize_latent_from: str = "zero",
    use_mahalanobis_loss: bool = False,
    reconstruct_surface: bool = True,
    reconstruct_from: str = "all",
    show_reconstruction_images: bool = True,
    save_reconstruction_images: bool = False,
    save_reconstructed_mesh: bool = False,
    compute_chamfer: bool = False,
    compute_lddmm: bool = False,
    compute_haussdorff: bool = False,
    compute_f1_score: bool = False,
    compute_dice: bool = False,
    save_latent_codes: bool = True,
    use_old_chamfer_surface_metric: bool = False,
    patient : str | None=None,
    surface : str = "all",
    reconstructed_meshes_dir_override: Path | None = None,
):
    experiments_dir = combo_dir / "experiments"
    images_dir = combo_config.IMAGES_DIR
    
    if reconstructed_meshes_dir_override is not None:
        reconstructed_meshes_dir = Path(
            reconstructed_meshes_dir_override
        )
    else:
        reconstructed_meshes_dir = Path(
            combo_config.RECONSTRUCTED_MESHES_DIR
        )

    reconstructed_meshes_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latents_dir = combo_config.LATENTS_DIR
    metrics_dir = combo_config.METRICS_DIR
    patient_meshes_dir = combo_config.PATIENT_MESHES_DIR
    patients_npy_data_dir = Path(combo_config.PATIENTS_NPY_DATA_DIR)

    exp_dir = experiments_dir / experiment_name
    if version is None or version == "latest":
        version_dir = get_latest_version_dir(exp_dir)
        version = version_dir.name
    else:
        version_dir = exp_dir / version

    if not version_dir.exists():
        raise FileNotFoundError(f"Version directory not found: {version_dir}")

    if hparams_file is None:
        hparams_file = version_dir / "hparams.json"

    specs = json.load(open(hparams_file, "r", encoding="utf-8"))

    specs["TestSplit"] = str((combo_dir / "test" / "data_fnames_test.json").resolve())
    # specs["TestSplit"] = str((combo_dir / "train" / "data_fnames_train.json").resolve())
    
    specs["TrainSplit"] = str((combo_dir / "train" / "data_fnames_train.json").resolve())

    # important: use combo-specific datasource
    specs["DataSource"] = str(patients_npy_data_dir)

    #old
    if override_with_dataset is not None:
        specs["TestSplit"] = override_with_dataset
    # fine old

    # debug
    # if override_with_dataset is not None:
    #     override_path = Path(override_with_dataset)
    # if not override_path.is_absolute():
    #     override_path = (combo_dir / override_path).resolve()
    # specs["TestSplit"] = str(override_path)
    # fine debug 


    print("\n")
    print(f"Combination: {combo_dir.name}")
    print(f"Experiment: {experiment_name}")
    print(f"Version: {version}")
    print("Specs:")
    pprint(specs)
    print("\n")

    decoder = Decoder(**specs["Network_specs"])

    decoder_weights_path = version_dir / "decoder_weights.pth"
    if decoder_weights_path.is_file():
        state_dict = torch.load(decoder_weights_path)
        decoder.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Decoder weights file not found in {version_dir}")

    decoder.to(DEVICE)
    model = DeepSDF(decoder=decoder, specs=specs)

    if num_samp_per_scene_for_fit is not None:
        specs["num_samp_per_scene"] = num_samp_per_scene_for_fit

    dataloader = SDFDataModule(specs=specs)
    dataloader.setup("test")
    dataset = dataloader.test_dataloader().dataset

    # degbug

    print("[DEBUG] len(dataset):", len(dataset))
    

    # fin debug

    data_file = dataset.data_file
    which_shapes = "test" if "test" in Path(data_file).name else "train"
    patient_names = get_dataset_patients_names(json.load(open(data_file, "r", encoding="utf-8")))

    if patient is not None:
        if patient not in patient_names:
            raise ValueError(
                f"Patient '{patient}' not found in dataset.\n"
                f"Available patients: {patient_names}"
            )

        shape_indices = [patient_names.index(patient)]
    else:
        shape_indices = range(len(dataset))

    print("[DEBUG]  len(patient_names):", len(patient_names))
    print("[DEBUG]  data_file:", data_file)

    decoder_input_scale = specs.get("scale_spatial_inputs_by", 100)

    # debug
    # decoder_input_scale /= decoder_input_scale
    # debug
    
    enforce_minmax = specs.get("enforce_minmax", False)
    clamp_distance = specs.get("clamp_distance", 0.1)

    if latent_reg_factor is None:
        latent_reg_factor = specs["code_reg_lambda"]

    loss_fn = torch.nn.MSELoss(reduction="sum")

    chamfer_dists = {}
    lddmm_losses = {}
    haussdorff_dists = {}
    f1_scores = {}
    precision_values = {}
    recall_values ={}
    latent_codes = {}
    dice_scores = {}



    print("\nRECONSTRUCTION PARAMETERS")
    print(f"    - {specs['num_samp_per_scene']} samples per scene to fit latents")
    print(f"    - reconstruct_from = {reconstruct_from}")
    print(f"    - init latent = {initialize_latent_from}")
    print(f"    - mahalanobis loss = {use_mahalanobis_loss}")
    print("\n")

    for shape_idx in shape_indices:

        patient_name = patient_names[shape_idx]
        print("\n\033[48;2;30;30;30;0;38;2;255;200;0m" + f"# {'='*10} PATIENT {patient_name} : {shape_idx+1}/{len(dataset)} {'='*10} #" + "\033[0m")

        coords_and_sdf_file = next(patients_npy_data_dir.glob(f"{patient_name}*.npy"), None)
        if coords_and_sdf_file is None:
            raise FileNotFoundError(
                f"No original npy data found in {patients_npy_data_dir} for patient {patient_name}"
            )

        chamfer_dists[patient_name] = {}
        haussdorff_dists[patient_name] = {}
        lddmm_losses[patient_name] = {}
        f1_scores[patient_name] = {}

        precision_values[patient_name] = {}
        recall_values[patient_name] = {}

        dice_scores[patient_name] = {}

        batch = dataset[shape_idx]
        data = batch[0]

        xyz_gt = data["coords"]
        sdf_gt = data["sdf"]
        mask_gt = data["mask"]

        if reconstruct_from == "la":
            # near_la = np.where(np.abs(sdf_gt[:, 1]) <= 0.005)
            near_la = (
                (mask_gt[:, 1] > 0.5)
                & (torch.abs(sdf_gt[:, 1]) <= 0.005)
            )

            xyz_gt = xyz_gt[near_la]
            sdf_gt = sdf_gt[near_la]
            mask_gt = mask_gt[near_la]

        xyz_gt = xyz_gt.reshape(-1, 3) * decoder_input_scale
        sdf_gt = sdf_gt.reshape(-1, decoder.out_dim)
        mask_gt = mask_gt.reshape(-1, decoder.out_dim)

        if enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, min=-clamp_distance, max=clamp_distance)

        sdf_gt = sdf_gt.to(DEVICE)
        xyz = xyz_gt.to(DEVICE)
        mask_gt = mask_gt.to(DEVICE)

        latent_size = decoder.latent_size

        if use_mahalanobis_loss or initialize_latent_from == "empirical":
            trained_latents_file = next(version_dir.glob("latents.npy"), None)
            if trained_latents_file is None:
                raise ValueError(f"latents.npy not found in version dir {version_dir}")

            trained_latents = torch.from_numpy(np.load(trained_latents_file)).to(device=DEVICE)
            mean_code = torch.mean(trained_latents, axis=0)
            cov = torch.cov(trained_latents.T)
            cov_inv = cov.inverse()

        if initialize_latent_from == "zero":
            latent = torch.zeros(latent_size, device=DEVICE)
        elif initialize_latent_from == "normal":
            latent = torch.randn(latent_size, device=DEVICE) * (1.0 / math.sqrt(latent_size))
        elif initialize_latent_from == "empirical":
            distrib = MultivariateNormal(loc=mean_code, covariance_matrix=cov)
            latent = distrib.rsample()
        else:
            raise ValueError(f"Unknown initialize_latent_from: {initialize_latent_from}")

        latent.requires_grad = True
        code_reg_lambda = latent_reg_factor

        epsilon = find_pointcloud_noise(
            decoder=decoder,
            model=model,
            xyz_gt=xyz,
            sdf_gt=sdf_gt,
            mask_gt=mask_gt,
            code_reg_lambda=code_reg_lambda,
            num_epochs_fit_latent=250,
            lr_fit_latent=0.005,
            loss_type=specs.get("use_loss", "MSE"),
        )

        beta = 100.0 * epsilon

        optimizer = torch.optim.Adam(params=[latent], lr=lr_fit_latent)
        num_samp_per_scene = sdf_gt.shape[0]
        num_epochs = num_epochs_fit_latent

        print("\n Fitting code ... \n")
        for _ in tqdm(range(num_epochs)):
            decoder.eval()
            optimizer.zero_grad()

            batch_vecs = latent.expand(num_samp_per_scene, -1)
            input_ = torch.cat([batch_vecs, xyz], dim=1)

            sdf_pred = decoder(input_)
            if enforce_minmax:
                sdf_pred = torch.clamp(sdf_pred, min=-clamp_distance, max=clamp_distance)

            if use_mahalanobis_loss:
                diff = latent - mean_code
                reg_loss = diff @ cov_inv @ diff
            else:
                reg_loss = latent.pow(2).sum()

            # chunk_loss = loss_fn(sdf_pred, sdf_gt) / (num_samp_per_scene * decoder.out_dim)
            chunk_loss = masked_regression_loss(
                pred=sdf_pred,
                gt=sdf_gt,
                mask=mask_gt,
                loss_type=specs.get("use_loss", "MSE"),
            )
                        
            loss = chunk_loss + beta * code_reg_lambda * reg_loss

            loss.backward()
            optimizer.step()

        latent.requires_grad = False

        if save_latent_codes:
            latent_codes[patient_name] = latent.cpu().numpy().ravel()

        if not reconstruct_surface:
            continue

        resolution = 256
        box_lim = 1.05

        with torch.no_grad():
            print("\n Computing prediction on grid ...")
            decoder.eval()

            x = np.linspace(-box_lim, box_lim, resolution)
            y = np.linspace(-box_lim, box_lim, resolution)
            z = np.linspace(-box_lim, box_lim, resolution)
            xx, yy, zz = np.meshgrid(x, y, z)
            grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

            ppb = 500000
            n_batches = len(grid) // ppb
            sdf_preds = []

            for i in range(n_batches + 1):
                xyz_chunk = torch.from_numpy(grid[ppb * i: ppb * (i + 1)] if i < n_batches else grid[ppb * i:]).to(DEVICE)
                xyz_chunk *= decoder_input_scale
                batch_vecs = latent.expand(xyz_chunk.shape[0], -1)
                input_ = torch.cat([batch_vecs, xyz_chunk], dim=1)
                sdf_pred_batch = decoder(input_.to(torch.float32)).cpu().data.numpy()
                sdf_preds.append(sdf_pred_batch)

        sdf_pred = np.concatenate(sdf_preds, axis=0)

        # debug
        print("\nDEBUG SDF PRED GRID")
        for i, name in enumerate(["epicardium", "lv_endo", "rv_endo"]):
            v = sdf_pred[:, i]
            print(
                name,
                "min:", np.min(v),
                "max:", np.max(v),
                "mean:", np.mean(v),
                "std:", np.std(v),
                "crosses zero:", np.min(v) <= 0.0 <= np.max(v)
            )
        # fine debug

        if decoder.out_dim != 3:
            raise NotImplementedError("Current test script assumes out_dim == 3")

        sdf_grid_pred = {
            "epicardium": sdf_pred[:, 0],
            "lv_endo": sdf_pred[:, 1],
            "rv_endo": sdf_pred[:, 2],
        }

        data_ = np.load(coords_and_sdf_file)
        points_all_in_scene = data_[:, :3]
        sdfs_all_gt = {
            "epicardium": data_[:, 0 + 3],
            "lv_endo": data_[:, 1 + 3],
            "rv_endo": data_[:, 2 + 3],
        }

        if surface == "all":
            organs_to_process = [
                "epicardium",
                "lv_endo",
                "rv_endo",
            ]
        else:
            organs_to_process = [surface]

        mesh_gt_dict = {}
        patient_dir = patient_meshes_dir / patient_name

        for organ_name in organs_to_process:
            mesh_file = next(patient_dir.rglob(f"{organ_name}-processed.vtp"), None)
            if mesh_file is None:
                raise FileNotFoundError(f"Ground-truth mesh not found for {patient_name}, organ {organ_name}, dir {patient_meshes_dir}")
            mesh_gt_dict[organ_name] = pv.read(mesh_file)

        patient_start = time.time()

        for organ in organs_to_process:
            print(f"\n > > > Processing {organ} surface ")

            organ_start = time.time()
            
            mesh_gt = mesh_gt_dict[organ]

            try:
                mesh_reconstructed = isosurface_from_sdf(
                    x, y, z,
                    sdf_pred=sdf_grid_pred[organ],
                    level=0.0,
                    box_lim=box_lim
                )
            except Exception:
                print(f"Skipping {organ} isosurface extraction: not found for current isovalue")
                continue

            if save_reconstructed_mesh:
                fname = f"{version}-{patient_name}-{organ}.vtp" if reconstruct_from == "all" \
                    else f"{version}-{patient_name}-{organ}-from_la_only.vtp"
                reconstructed_meshes_dir.mkdir(parents=True, exist_ok=True)
                mesh_reconstructed.save(reconstructed_meshes_dir / fname)

            if show_reconstruction_images or save_reconstruction_images:
                mesh_gt_show = mesh_gt.copy()
                mesh_reconstructed_show = mesh_reconstructed.copy()

                scale = mesh_gt.field_data["scale-tooriginalrange"]
                mesh_gt_show.points *= scale
                mesh_reconstructed_show.points *= scale

                implicit_distance = vtkImplicitPolyDataDistance()
                implicit_distance.SetInput(mesh_gt_show)
                points_pred = mesh_reconstructed_show.points
                signed_distances = np.array([implicit_distance.EvaluateFunction(p) for p in points_pred])
                mesh_reconstructed_show.point_data["error"] = signed_distances

                last_cam_pos = None
                if show_reconstruction_images:
                    plotter = plot_gt_vs_reconstructed_with_error(
                        mesh_gt_show, mesh_reconstructed_show, patient_name, signed_distances, off_screen=False
                    )
                    plotter.show(interactive=True)
                    last_cam_pos = plotter.camera_position
                    plotter.close()

                if save_reconstruction_images:
                    images_dir.mkdir(parents=True, exist_ok=True)
                    save_fname = images_dir / f"{patient_name}_{organ}_gt_vs_reconstructed_with_error_{version}.png"
                    plotter = plot_gt_vs_reconstructed_with_error(
                        mesh_gt_show, mesh_reconstructed_show, patient_name, signed_distances, off_screen=True
                    )
                    if last_cam_pos is not None:
                        plotter.camera_position = last_cam_pos
                    plotter.screenshot(save_fname, transparent_background=True)
                    pv.close_all()

            scale = mesh_gt.field_data["scale-tooriginalrange"][0]
            scale_mm = scale * 0.001

            if compute_chamfer or compute_haussdorff or compute_f1_score:
                samples_count = 50000
                samples_gt = make_trimesh_from_pv(mesh_gt).sample(count=samples_count)
                samples_pred = make_trimesh_from_pv(mesh_reconstructed).sample(count=samples_count)

                if compute_chamfer:
                    chamfer_dists[patient_name][organ] = chamfer_distance_L2(samples_gt, samples_pred) * scale_mm
                if compute_haussdorff:
                    haussdorff_dists[patient_name][organ] = haussdorff(samples_gt, samples_pred) * scale_mm
                
                if compute_f1_score:
                    diag = np.linalg.norm(samples_gt.max(axis=0) - samples_gt.min(axis=0))
                    tau_factor = 0.01
                    tau = tau_factor * diag

                    #----------------------
                    print("||max - min|| = ", diag)
                    print("scale:", tau_factor)
                    print("tau:", tau)


                    f1 = f1_score_function(samples_pred, samples_gt, tau)

                    f1_scores[patient_name][organ] = f1["f1score"]

                    precision_values[patient_name][organ] = f1["precision"]
                    recall_values[patient_name][organ] = f1["recall"]

            if compute_lddmm:
                mesh_gt_remeshed_file = next(patient_dir.rglob(f"{organ}-processed-remeshed.vtp"), None)
                if mesh_gt_remeshed_file is not None:
                    mesh_gt_lddmm = pv.read(mesh_gt_remeshed_file)
                else:
                    mesh_gt_lddmm = remesh(mesh_gt, n_points=50000)

                mesh_reconstructed_lddmm = remesh(mesh_reconstructed, n_points=50000)
                lddmm_losses[patient_name][organ] = LDDMM_loss(
                    mesh_gt_lddmm, mesh_reconstructed_lddmm, remeshing=False, gamma=1.0, device=DEVICE
                )
            
            if compute_dice:

                SDF_SIGN_FOR_DICE = {
                        "epicardium": 1.0,
                        "lv_endo": -1.0,
                        "rv_endo": -1.0,

                    }
                
                sign = SDF_SIGN_FOR_DICE[organ]

                sdf_gt_on_grid = sdf_gt_on_regular_grid(
                    mesh_gt=mesh_gt,
                    resolution=resolution,
                    box_lim=box_lim
                )

                dice = compute_dice_score(
                    sdf_pred=sdf_grid_pred[organ],
                    sdf_gt= sign * sdf_gt_on_grid,
                    level=0.0
                )

                dice_scores[patient_name][organ] = dice
            

            organ_elapsed = time.time() - organ_start
            print(f"computed metrics for patient {patient_name}, {organ} in {organ_elapsed} seconds.")


        patient_elapsed = time.time() - patient_start
        print(f"computed all metrics for patient {patient_name} in {patient_elapsed} seconds.")


    if compute_chamfer:
        save_metrics_csv("chamfer", experiment_name, version, which_shapes, chamfer_dists, metrics_dir)
        print("Saved chamfer distances.")

    if compute_haussdorff:
        save_metrics_csv("haussdorff", experiment_name, version, which_shapes, haussdorff_dists, metrics_dir)
        print("Saved haussdorff distances.")

    if compute_lddmm:
        save_metrics_csv("LDDMM", experiment_name, version, which_shapes, lddmm_losses, metrics_dir)
        print("Saved LDDMM distances.")

    if compute_f1_score:
        save_metrics_csv("f1_score", experiment_name, version, which_shapes, f1_scores, metrics_dir)
        print("Saved F1 scores.")

        #--------------------
        save_metrics_csv("precision", experiment_name, version, which_shapes, precision_values, metrics_dir)
        save_metrics_csv("recall", experiment_name, version, which_shapes, recall_values, metrics_dir)

        print("Saved recall and precision")
        #-------------------------
    
    if compute_dice:
        save_metrics_csv("dice", experiment_name, version, which_shapes, dice_scores, metrics_dir)
        print("Saved Dice scores.")

    if save_latent_codes:
        loss_type = "L2" if not use_mahalanobis_loss else "Maha"
        save_latents_npz(
            experiment_name, version, which_shapes,
            code_reg_lambda, num_epochs, initialize_latent_from, 
            loss_type, latent_codes, latents_dir
        )
        print("Saved fitted latents.")

    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo_dir", required=True, type=Path)
    parser.add_argument("--experiment_name", "-e", type=str, default="noise_study")
    parser.add_argument("--version", "-v", type=str, default="latest")
    parser.add_argument("--override_with_dataset", "-od", type=str, default=None)
    parser.add_argument("--mode", "-m", type=int, default=1, choices=[1, 2])
    parser.add_argument("--reconstruct_from", "-r", type=str, default="all", choices=["la", "all"])
    parser.add_argument("--num_samp_per_scene_for_fit", "-nsamp", type=int, default=None)
    parser.add_argument("--num_epochs", "-N", type=int, default=250)
    parser.add_argument("--latent_reg_factor", "-lreg", type=float, default=None)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--init_latent_from", type=str, default="zero", choices=["zero", "normal", "empirical"])
    parser.add_argument("--use_mahalanobis_loss", "-maha", action="store_true")
    parser.add_argument("--save_latent_codes", "-sc", action="store_true")
    parser.add_argument("--interactive_images", "-i", action="store_true")
    parser.add_argument("--save_images", "-si", action="store_true")
    parser.add_argument("--save_reconstructed_meshes", "-sm", action="store_true")
    parser.add_argument("--compute_chamfer", "-chd", action="store_true")
    parser.add_argument("--compute_lddmm", "-lddmm", action="store_true")
    parser.add_argument("--compute_haussdorff", "-hauss", action="store_true")
    parser.add_argument("--compute_f1_score", "-f1", action="store_true")
    parser.add_argument("--compute_dice", "-dice", action="store_true")

    parser.add_argument("--patient", type=str, default=None, help="Test only the specified patient")

    parser.add_argument("--surface", type=str, default="all", choices=["all", "epicardium", "lv_endo", "rv_endo"], help="reconstruct and display only the selected surface",)
    
    parser.add_argument(
        "--reconstructed_meshes_dir",
        type=Path,
        default=None,
        help="Optional override for reconstructed meshes output directory.",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    combo_dir = args.combo_dir.resolve()
    config_path = combo_dir / "config.py"

    if not combo_dir.exists():
        raise FileNotFoundError(f"Combination directory does not exist: {combo_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.py in {combo_dir}")

    combo_config = load_module_from_path("combo_config_test", config_path)

    kwargs = {
        "combo_dir": combo_dir,
        "combo_config": combo_config,
        "experiment_name": args.experiment_name,
        "version": args.version,
        "override_with_dataset": args.override_with_dataset,
        "num_samp_per_scene_for_fit": args.num_samp_per_scene_for_fit,
        "num_epochs_fit_latent": args.num_epochs,
        "latent_reg_factor": args.latent_reg_factor,
        "lr_fit_latent": args.lr,
        "initialize_latent_from": args.init_latent_from,
        "use_mahalanobis_loss": args.use_mahalanobis_loss,
        "patient": args.patient,
        "surface": args.surface,
        "reconstructed_meshes_dir_override": args.reconstructed_meshes_dir,
    }

    if args.mode == 1:
        run_kwargs = {
            **kwargs,
            "reconstruct_surface": True,
            "reconstruct_from": args.reconstruct_from,
            "show_reconstruction_images": args.interactive_images,
            "save_reconstruction_images": args.save_images,
            "save_reconstructed_mesh": args.save_reconstructed_meshes,
            "compute_chamfer": args.compute_chamfer,
            "compute_lddmm": args.compute_lddmm,
            "compute_haussdorff": args.compute_haussdorff,
            "compute_f1_score": args.compute_f1_score,
            "compute_dice": args.compute_dice,
            "save_latent_codes": args.save_latent_codes,
        }
    else:
        run_kwargs = {
            **kwargs,
            "reconstruct_surface": False,
            "reconstruct_from": args.reconstruct_from,
            "show_reconstruction_images": False,
            "save_reconstruction_images": False,
            "save_reconstructed_mesh": False,
            "compute_chamfer": False,
            "compute_lddmm": False,
            "compute_haussdorff": False,
            "compute_f1_score": False,
            "compute_dice": False,
            "save_latent_codes": True,
        }

    run(**run_kwargs)


if __name__ == "__main__":
    main()
