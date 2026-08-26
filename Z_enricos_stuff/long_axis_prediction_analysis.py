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
import matplotlib.pyplot as plt
from vtk import vtkImplicitPolyDataDistance

from model.deepsdf_decoder import Decoder, DeepSDF
from model.deepsdf_dataloader import SDFDataModule
from utils.metrics import chamfer_distance_L2, LDDMM_loss, haussdorff, f1_score_function, compute_dice_score, sdf_gt_on_regular_grid
from utils.surface_utils import remesh, make_trimesh_from_pv
from utils.reconstruction_utils import isosurface_from_sdf
from utils.visual_utils import plot_gt_vs_reconstructed_with_error

import random
import hashlib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using GPU:", torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else "CPU")


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


# def get_dataset_patients_names(data: list[str]):
#     patient_names = []

#     for fullfname in data:
#         fname = Path(fullfname).name

#         if "-epi_lv_rv_" in fname:
#             patient_name = fname.split("-epi_lv_rv_")[0]
#         elif "_MRI_like_" in fname:
#             patient_name = fname.split("_MRI_like_")[0]
#         else:
#             patient_name = fname.replace(".npy", "")

#         patient_names.append(patient_name)

#     return patient_names

def get_dataset_patients_names(data: list[str]):
    patient_names = []

    known_suffixes = [
        "_three_axis_mri_samples.npy",
        "_mri_samples.npy",
        "_MRI_like_samples.npy",
        "_MRI_like.npy",
        "_echo_samples.npy",
        "_three_axis_mri_grid_samples.npy",
        "_short_axis_mri_grid_samples.npy",
    ]

    for fullfname in data:
        fname = Path(fullfname).name

        if "-epi_lv_rv_" in fname:
            patient_name = fname.split("-epi_lv_rv_")[0]
        else:
            patient_name = None

            for suffix in known_suffixes:
                if fname.endswith(suffix):
                    patient_name = fname[:-len(suffix)]
                    break

            if patient_name is None:
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


def find_pointcloud_noise(
    decoder: Decoder,
    model: DeepSDF,
    xyz_gt,
    sdf_gt,
    mask_gt,
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
    num_samp_per_scene = xyz_gt.shape[0]
    num_valid_values = mask_gt.sum().clamp_min(1.0)

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

            squared_error = (sdf_pred - sdf_gt) ** 2
            recon_loss = (squared_error * mask_gt).sum()
            chunk_loss = recon_loss / num_valid_values
            loss = chunk_loss + 100 * epsilon * code_reg_lambda * reg_loss

            loss.backward()
            optimizer.step()

            if i == num_epochs_fit_latent - 1:
                epsilon = np.sqrt(
                    recon_loss.detach().item()
                    / max(num_valid_values.detach().item() - 1.0, 1.0)
                )
                epsilons.append(epsilon)

        tol = 1e-7
        if abs(epsilons[-1] ** 2 - epsilons[-2] ** 2) < tol:
            break

    return epsilons[-1]


def get_latest_version_dir(exp_dir: Path) -> Path:
    version_dirs = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("version_")]
    if not version_dirs:
        raise FileNotFoundError(f"No version_* directories found in {exp_dir}")
    return max(version_dirs, key=lambda p: int(p.name.split("_")[-1]))

def tensor_sha256(tensor: torch.Tensor) -> str:
    array = (
        tensor
        .detach()
        .cpu()
        .contiguous()
        .numpy()
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def predict_sdf_on_la_points(
    decoder: Decoder,
    latent: torch.Tensor,
    raw_data: np.ndarray,
    decoder_input_scale: float,
    enforce_minmax: bool = False,
    clamp_distance: float = 0.1,
    n_la_points: int = 2000,
):
    """
    Predict the three SDFs only on the last `n_la_points` rows of the
    ORIGINAL .npy file.

    Expected columns:
        0:3 -> xyz
        3:6 -> sdf_epi, sdf_lv, sdf_rv
        6:9 -> mask_epi, mask_lv, mask_rv
    """
    if raw_data.ndim != 2 or raw_data.shape[1] < 9:
        raise ValueError(
            f"Expected an N x >=9 npy array, got shape {raw_data.shape}"
        )

    if raw_data.shape[0] < n_la_points:
        raise ValueError(
            f"File contains only {raw_data.shape[0]} points, "
            f"but n_la_points={n_la_points}."
        )

    la_data = raw_data[-n_la_points:]

    xyz_np = la_data[:, 0:3].astype(np.float32, copy=False)
    sdf_gt_np = la_data[:, 3:6].astype(np.float32, copy=False)
    mask_np = la_data[:, 6:9].astype(np.float32, copy=False)

    xyz_decoder = torch.from_numpy(xyz_np).to(DEVICE)
    xyz_decoder = xyz_decoder * decoder_input_scale

    with torch.no_grad():
        decoder.eval()
        batch_vecs = latent.expand(xyz_decoder.shape[0], -1)
        input_ = torch.cat([batch_vecs, xyz_decoder], dim=1)

        sdf_pred = decoder(input_.float())

        if enforce_minmax:
            sdf_pred = torch.clamp(
                sdf_pred,
                min=-clamp_distance,
                max=clamp_distance,
            )

    return (
        xyz_np,
        sdf_gt_np,
        sdf_pred.detach().cpu().numpy(),
        mask_np,
    )


def project_points_to_plane(points: np.ndarray):
    """
    Project approximately planar 3D points to a best-fit 2D coordinate system
    using PCA/SVD.

    Returns:
        u, v : planar coordinates
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected N x 3 points, got {points.shape}")

    center = points.mean(axis=0)
    centered = points - center

    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    axis_u = vh[0]
    axis_v = vh[1]

    u = centered @ axis_u
    v = centered @ axis_v

    return u, v


def _interpolate_sdf_on_plane(
    u: np.ndarray,
    v: np.ndarray,
    sdf: np.ndarray,
    valid: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
):
    """
    Interpolate a sparse SDF sampled on a plane onto a regular 2D grid.
    Linear interpolation is used first; nearest-neighbour fills holes.
    """
    valid = np.asarray(valid, dtype=bool)
    finite = (
        valid
        & np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(sdf)
    )

    if finite.sum() < 3:
        return None

    uv = np.column_stack([u[finite], v[finite]])
    values = sdf[finite]

    try:
        grid_linear = griddata(
            uv,
            values,
            (uu, vv),
            method="linear",
        )
    except Exception:
        grid_linear = None

    if grid_linear is None:
        try:
            return griddata(
                uv,
                values,
                (uu, vv),
                method="nearest",
            )
        except Exception:
            return None

    # Fill NaNs outside the linear interpolation convex hull with nearest.
    if np.isnan(grid_linear).any():
        try:
            grid_nearest = griddata(
                uv,
                values,
                (uu, vv),
                method="nearest",
            )
            fill = np.isnan(grid_linear)
            grid_linear[fill] = grid_nearest[fill]
        except Exception:
            pass

    return grid_linear


def plot_la_zero_level(
    xyz: np.ndarray,
    sdf_gt: np.ndarray,
    sdf_pred: np.ndarray,
    mask: np.ndarray,
    patient_name: str,
    plane_name: str,
    grid_resolution: int = 1000,
    save_path: Path | None = None,
    show: bool = True,
):
    """
    Plot GT SDF=0 and predicted SDF=0 on one LAX plane.

    One subplot per output:
        epicardium, LV endocardium, RV endocardium.

    Black = GT zero level
    Red   = predicted zero level
    """
    organ_names = ["epicardium", "lv_endo", "rv_endo"]

    u, v = project_points_to_plane(xyz)

    ui = np.linspace(u.min(), u.max(), grid_resolution)
    vi = np.linspace(v.min(), v.max(), grid_resolution)
    uu, vv = np.meshgrid(ui, vi)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for organ_idx, organ_name in enumerate(organ_names):
        ax = axes[organ_idx]

        valid = mask[:, organ_idx] > 0.5

        gt_grid = _interpolate_sdf_on_plane(
            u,
            v,
            sdf_gt[:, organ_idx],
            valid,
            uu,
            vv,
        )

        pred_grid = _interpolate_sdf_on_plane(
            u,
            v,
            sdf_pred[:, organ_idx],
            valid,
            uu,
            vv,
        )

        ax.scatter(
            u[valid],
            v[valid],
            s=4,
            alpha=0.15,
        )

        gt_has_zero = False
        pred_has_zero = False

        if gt_grid is not None and np.isfinite(gt_grid).any():
            gt_min = np.nanmin(gt_grid)
            gt_max = np.nanmax(gt_grid)
            gt_has_zero = gt_min <= 0.0 <= gt_max

            if gt_has_zero:
                ax.contour(
                    uu,
                    vv,
                    gt_grid,
                    levels=[0.0],
                    colors="black",
                    linewidths=2.5,
                )

        if pred_grid is not None and np.isfinite(pred_grid).any():
            pred_min = np.nanmin(pred_grid)
            pred_max = np.nanmax(pred_grid)
            pred_has_zero = pred_min <= 0.0 <= pred_max

            if pred_has_zero:
                ax.contour(
                    uu,
                    vv,
                    pred_grid,
                    levels=[0.0],
                    colors="red",
                    linewidths=2.0,
                )

        ax.set_title(
            f"{organ_name}\n"
            f"GT zero: {gt_has_zero} | pred zero: {pred_has_zero}"
        )
        ax.set_xlabel("plane u")
        ax.set_ylabel("plane v")
        ax.set_aspect("equal", adjustable="box")

        # Dummy artists for a stable legend.
        ax.plot([], [], color="black", linewidth=2.5, label="GT SDF = 0")
        ax.plot([], [], color="red", linewidth=2.0, label="Pred SDF = 0")
        ax.legend()

    fig.suptitle(
        f"{patient_name} - {plane_name}\n"
        "Zero-level comparison on original LAX sample points",
        fontsize=14,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved LAX zero-level image: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_la_sdf_error_summary(
    patient_name: str,
    plane_name: str,
    sdf_gt: np.ndarray,
    sdf_pred: np.ndarray,
    mask: np.ndarray,
):
    """Print simple SDF regression errors on one LAX plane."""
    organ_names = ["epicardium", "lv_endo", "rv_endo"]

    print(f"\nLAX SDF ERROR SUMMARY - {patient_name} - {plane_name}")

    for i, organ in enumerate(organ_names):
        valid = mask[:, i] > 0.5

        if valid.sum() == 0:
            print(f"  {organ}: no valid mask values")
            continue

        err = sdf_pred[valid, i] - sdf_gt[valid, i]

        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        bias = np.mean(err)

        print(
            f"  {organ}: "
            f"N={valid.sum()} | "
            f"MAE={mae:.6e} | "
            f"RMSE={rmse:.6e} | "
            f"bias(pred-GT)={bias:.6e}"
        )



def plot_both_la_zero_levels(
    planes_data,
    patient_name: str,
    grid_resolution: int = 1000,
    save_path: Path | None = None,
    show: bool = True,
):
    """
    Plot both LAX planes in a single 3x2 panel.

    Rows:
        epicardium
        lv_endo
        rv_endo

    Columns:
        LAX 1
        LAX 2

    Black = GT zero level
    Red   = predicted zero level
    """

    organ_names = ["epicardium", "lv_endo", "rv_endo"]

    n_planes = len(planes_data)

    if n_planes != 2:
        raise ValueError(
            f"Expected exactly 2 LAX planes, got {n_planes}"
        )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 16),
        squeeze=False,
    )

    for plane_idx, plane_data in enumerate(planes_data):

        xyz = plane_data["xyz"]
        sdf_gt = plane_data["sdf_gt"]
        sdf_pred = plane_data["sdf_pred"]
        mask = plane_data["mask"]
        plane_name = plane_data["name"]

        # Important: compute the planar coordinates independently
        # for each LAX plane
        u, v = project_points_to_plane(xyz)

        ui = np.linspace(
            u.min(),
            u.max(),
            grid_resolution,
        )

        vi = np.linspace(
            v.min(),
            v.max(),
            grid_resolution,
        )

        uu, vv = np.meshgrid(ui, vi)

        for organ_idx, organ_name in enumerate(organ_names):

            ax = axes[organ_idx, plane_idx]

            valid = mask[:, organ_idx] > 0.5

            gt_grid = _interpolate_sdf_on_plane(
                u,
                v,
                sdf_gt[:, organ_idx],
                valid,
                uu,
                vv,
            )

            pred_grid = _interpolate_sdf_on_plane(
                u,
                v,
                sdf_pred[:, organ_idx],
                valid,
                uu,
                vv,
            )

            # sampled points
            ax.scatter(
                u[valid],
                v[valid],
                s=4,
                alpha=0.15,
            )

            gt_has_zero = False
            pred_has_zero = False

            # -------------------------
            # Ground truth zero level
            # -------------------------
            if (
                gt_grid is not None
                and np.isfinite(gt_grid).any()
            ):
                gt_min = np.nanmin(gt_grid)
                gt_max = np.nanmax(gt_grid)

                gt_has_zero = (
                    gt_min <= 0.0 <= gt_max
                )

                if gt_has_zero:
                    ax.contour(
                        uu,
                        vv,
                        gt_grid,
                        levels=[0.0],
                        colors="black",
                        linewidths=2.5,
                    )

            # -------------------------
            # Predicted zero level
            # -------------------------
            if (
                pred_grid is not None
                and np.isfinite(pred_grid).any()
            ):
                pred_min = np.nanmin(pred_grid)
                pred_max = np.nanmax(pred_grid)

                pred_has_zero = (
                    pred_min <= 0.0 <= pred_max
                )

                if pred_has_zero:
                    ax.contour(
                        uu,
                        vv,
                        pred_grid,
                        levels=[0.0],
                        colors="red",
                        linewidths=2.0,
                    )

            ax.set_title(
                f"{organ_name} - {plane_name}\n"
                f"GT zero: {gt_has_zero} | "
                f"pred zero: {pred_has_zero}"
            )

            ax.set_xlabel("plane u")
            ax.set_ylabel("plane v")

            ax.set_aspect(
                "equal",
                adjustable="box",
            )

            # dummy artists for legend
            ax.plot(
                [],
                [],
                color="black",
                linewidth=2.5,
                label="GT SDF = 0",
            )

            ax.plot(
                [],
                [],
                color="red",
                linewidth=2.0,
                label="Pred SDF = 0",
            )

            ax.legend()

    fig.suptitle(
        f"{patient_name}\n"
        "Zero-level comparison on both LAX planes",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"Saved combined LAX zero-level image: "
            f"{save_path}"
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


# ======================== #
# RUN TESTS
# ======================== #
def run(
    combo_dir: Path,
    combo_config,
    experiment_name: str,
    version: str | None = None,
    override_with_dataset: str | None = None,
    test_data_dir: Path | None = None,
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
    surface: str = "all",
    la_diagnostics: bool = False,
    n_la_points: int = 2000,
    n_points_per_la_plane: int = 1000,
    show_la_zero_levels: bool = True,
    save_la_zero_levels: bool = False,
):
    experiments_dir = combo_dir / "experiments"
    images_dir = combo_config.IMAGES_DIR
    reconstructed_meshes_dir = combo_config.RECONSTRUCTED_MESHES_DIR
    latents_dir = combo_config.LATENTS_DIR
    metrics_dir = combo_config.METRICS_DIR
    patient_meshes_dir = combo_config.PATIENT_MESHES_DIR

    # patients_npy_data_dir = Path(combo_config.PATIENTS_NPY_DATA_DIR)
    if test_data_dir is not None:
        patients_npy_data_dir = Path(test_data_dir).resolve()
    else:
        patients_npy_data_dir = Path(
            combo_config.PATIENTS_NPY_DATA_DIR
        ).resolve()

    if not patients_npy_data_dir.is_dir():
        raise FileNotFoundError(
            f"Test data directory does not exist: {patients_npy_data_dir}"
        )


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
    specs["TrainSplit"] = str((combo_dir / "train" / "data_fnames_train.json").resolve())

    # important: use combo-specific datasource
    specs["DataSource"] = str(patients_npy_data_dir)

    #old
    # if override_with_dataset is not None:
    #     specs["TestSplit"] = override_with_dataset
    # fine old

    if override_with_dataset is not None:
        override_path = Path(override_with_dataset)

        if not override_path.is_absolute():
            override_path = (combo_dir / override_path).resolve()
        else:
            override_path = override_path.resolve()

        if not override_path.is_file():
            raise FileNotFoundError(
                f"Test split JSON does not exist: {override_path}"
            )

        specs["TestSplit"] = str(override_path)


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

    data_file = dataset.data_file
    which_shapes = "test" if "test" in Path(data_file).name else "train"
    patient_names = get_dataset_patients_names(json.load(open(data_file, "r", encoding="utf-8")))

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

    for shape_idx in range(len(dataset)):

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

        patient_seed = SEED + shape_idx

        random.seed(patient_seed)
        np.random.seed(patient_seed)
        torch.manual_seed(patient_seed)
        torch.cuda.manual_seed_all(patient_seed)

        batch = dataset[shape_idx]

        # batch = dataset[shape_idx]
        data = batch[0]

        # debug
        print("\n" + "=" * 70)
        print(f"DEBUG PATIENT: {patient_name}")
        print(f"coords_and_sdf_file: {coords_and_sdf_file}")

        raw_data = np.load(coords_and_sdf_file)

        print("\nRAW NPY")
        print("shape:", raw_data.shape)
        print("dtype:", raw_data.dtype)

        if raw_data.ndim == 2:
            print("number of columns:", raw_data.shape[1])

            if raw_data.shape[1] >= 9:
                print("coords shape:", raw_data[:, :3].shape)
                print("sdf shape:", raw_data[:, 3:6].shape)
                print("mask shape:", raw_data[:, 6:9].shape)

                print("first row coords:", raw_data[0, :3])
                print("first row sdf:", raw_data[0, 3:6])
                print("first row mask:", raw_data[0, 6:9])

                print(
                    "unique mask values:",
                    [np.unique(raw_data[:, i]) for i in range(6, 9)]
                )

        print("\nDATASET OUTPUT")
        print("batch type:", type(batch))
        print("data type:", type(data))

        if isinstance(data, dict):
            print("data keys:", list(data.keys()))

            for key, value in data.items():
                if hasattr(value, "shape"):
                    print(
                        f"{key}: shape={tuple(value.shape)}, "
                        f"dtype={value.dtype}, type={type(value)}"
                    )
                else:
                    print(f"{key}: type={type(value)}, value={value}")

        print("=" * 70)

        # fine debug

        xyz_gt = data["coords"]
        sdf_and_mask_gt = data["sdf"]

        print("\nSELECTED INPUT FINGERPRINT")
        print("patient seed:", patient_seed)
        print("coords SHA256:", tensor_sha256(xyz_gt))
        print("sdf+mask SHA256:", tensor_sha256(sdf_and_mask_gt))

        # print("\nBEFORE PROCESSING")
        # print("xyz_gt shape:", tuple(xyz_gt.shape))
        # print("sdf_and_mask_gt shape:", tuple(sdf_and_mask_gt.shape))
        # print("decoder.out_dim:", decoder.out_dim)

        if sdf_and_mask_gt.ndim != 2:
            raise ValueError(
                f"Unexpected data['sdf'] shape: {tuple(sdf_and_mask_gt.shape)}"
            )

        if sdf_and_mask_gt.shape[1] == decoder.out_dim:
            sdf_gt = sdf_and_mask_gt
            mask_gt = torch.ones_like(sdf_gt)
        elif sdf_and_mask_gt.shape[1] == 2 * decoder.out_dim:
            sdf_gt = sdf_and_mask_gt[:, :decoder.out_dim]
            mask_gt = sdf_and_mask_gt[:, decoder.out_dim:]
        else:
            raise ValueError(
                f"Unexpected number of columns in data['sdf']: "
                f"{sdf_and_mask_gt.shape[1]}. Expected "
                f"{decoder.out_dim} or {2 * decoder.out_dim}."
            )

        xyz_gt = xyz_gt.reshape(-1, 3) * decoder_input_scale
        sdf_gt = sdf_gt.reshape(-1, decoder.out_dim)
        mask_gt = mask_gt.reshape(-1, decoder.out_dim)

        if xyz_gt.shape[0] != sdf_gt.shape[0]:
            raise RuntimeError(
                f"Coordinates/SDF mismatch: {tuple(xyz_gt.shape)} vs {tuple(sdf_gt.shape)}"
            )

        if enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, min=-clamp_distance, max=clamp_distance)

        sdf_gt = sdf_gt.to(DEVICE)
        mask_gt = mask_gt.to(DEVICE)
        xyz = xyz_gt.to(DEVICE)

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
        beta = 100 * find_pointcloud_noise(
            decoder, model, xyz, sdf_gt, mask_gt,
            code_reg_lambda=code_reg_lambda,
            num_epochs_fit_latent=250,
            lr_fit_latent=0.005
        )

        print("beta", beta)

        optimizer = torch.optim.Adam(params=[latent], lr=lr_fit_latent)
        num_samp_per_scene = xyz.shape[0]
        num_valid_values = mask_gt.sum().clamp_min(1.0)
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

            squared_error = (sdf_pred - sdf_gt) ** 2
            chunk_loss = (squared_error * mask_gt).sum() / num_valid_values
            loss = chunk_loss + beta * code_reg_lambda * reg_loss

            loss.backward()
            optimizer.step()

        latent.requires_grad = False

        print("latent SHA256:", tensor_sha256(latent))
        print("latent norm:", latent.norm().item())

        if save_latent_codes:
            latent_codes[patient_name] = latent.cpu().numpy().ravel()


        if la_diagnostics:
            print("\n" + "=" * 70)
            print("LAX DIAGNOSTICS")
            print("=" * 70)

            raw_data_for_la = np.load(coords_and_sdf_file)

            (
                xyz_la,
                sdf_la_gt,
                sdf_la_pred,
                mask_la,
            ) = predict_sdf_on_la_points(
                decoder=decoder,
                latent=latent,
                raw_data=raw_data_for_la,
                decoder_input_scale=decoder_input_scale,
                enforce_minmax=enforce_minmax,
                clamp_distance=clamp_distance,
                n_la_points=n_la_points,
            )

            print("LAX points:", xyz_la.shape[0])
            print("xyz_la shape:", xyz_la.shape)
            print("sdf_la_gt shape:", sdf_la_gt.shape)
            print("sdf_la_pred shape:", sdf_la_pred.shape)
            print("mask_la shape:", mask_la.shape)

            if n_la_points % n_points_per_la_plane != 0:
                raise ValueError(
                    f"n_la_points={n_la_points} is not divisible by "
                    f"n_points_per_la_plane={n_points_per_la_plane}"
                )

            n_planes = n_la_points // n_points_per_la_plane

            print(
                f"Interpreting the last {n_la_points} points as "
                f"{n_planes} LAX plane(s) of "
                f"{n_points_per_la_plane} points each."
            )

            planes_data = []

            for la_plane_idx in range(n_planes):

                start = (
                    la_plane_idx
                    * n_points_per_la_plane
                )

                stop = (
                    start
                    + n_points_per_la_plane
                )

                xyz_plane = xyz_la[start:stop]
                sdf_gt_plane = sdf_la_gt[start:stop]
                sdf_pred_plane = sdf_la_pred[start:stop]
                mask_plane = mask_la[start:stop]

                plane_name = f"LAX {la_plane_idx + 1}"

                print_la_sdf_error_summary(
                    patient_name=patient_name,
                    plane_name=plane_name,
                    sdf_gt=sdf_gt_plane,
                    sdf_pred=sdf_pred_plane,
                    mask=mask_plane,
                )

                planes_data.append(
                    {
                        "xyz": xyz_plane,
                        "sdf_gt": sdf_gt_plane,
                        "sdf_pred": sdf_pred_plane,
                        "mask": mask_plane,
                        "name": plane_name,
                    }
                )

                if show_la_zero_levels or save_la_zero_levels:

                    if save_la_zero_levels:

                        la_image_dir = (
                            images_dir
                            / experiment_name
                            / "la_zero_levels"
                        )

                        la_save_path = (
                            la_image_dir
                            / f"{version}-{patient_name}-LAX-both-zero-level.png"
                        )

                    else:
                        la_save_path = None

                    plot_both_la_zero_levels(
                        planes_data=planes_data,
                        patient_name=patient_name,
                        save_path=la_save_path,
                        show=show_la_zero_levels,
                    )

            print("=" * 70)

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
                raise FileNotFoundError(f"Ground-truth mesh not found for {patient_name}, organ {organ_name}")
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
    parser.add_argument("--test_data_dir", type=Path, default=None, help="Directory containing the npy files used for testing",)
    parser.add_argument(
        "--la_diagnostics",
        action="store_true",
        help=(
            "After fitting the latent code, predict SDF only on the last "
            "N LAX points of the original npy and compare GT vs predicted zero level."
        ),
    )
    parser.add_argument(
        "--n_la_points",
        type=int,
        default=2000,
        help="Number of final rows in each npy belonging to the LAX planes.",
    )
    parser.add_argument(
        "--n_points_per_la_plane",
        type=int,
        default=1000,
        help="Number of points belonging to each individual LAX plane.",
    )
    parser.add_argument(
        "--no_show_la_zero_levels",
        action="store_true",
        help="Do not open interactive matplotlib windows for the LAX zero-level plots.",
    )
    parser.add_argument(
        "--save_la_zero_levels",
        action="store_true",
        help="Save LAX zero-level comparison plots under IMAGES_DIR.",
    )
    parser.add_argument(    "--surface",    type=str,    default="all",    choices=[
        "all",
        "epicardium",
        "lv_endo",
        "rv_endo",
    ], help=("Reconstruct, display and evaluate only the selected surface."),
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
        "test_data_dir": args.test_data_dir,
        "num_samp_per_scene_for_fit": args.num_samp_per_scene_for_fit,
        "num_epochs_fit_latent": args.num_epochs,
        "latent_reg_factor": args.latent_reg_factor,
        "lr_fit_latent": args.lr,
        "initialize_latent_from": args.init_latent_from,
        "use_mahalanobis_loss": args.use_mahalanobis_loss,
        "surface": args.surface,
        "la_diagnostics": args.la_diagnostics,
        "n_la_points": args.n_la_points,
        "n_points_per_la_plane": args.n_points_per_la_plane,
        "show_la_zero_levels": not args.no_show_la_zero_levels,
        "save_la_zero_levels": args.save_la_zero_levels,
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
