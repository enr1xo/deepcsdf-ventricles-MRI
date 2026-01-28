import json
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import pyvista as pv
from model.atria_deepsdf_decoder import Decoder
from model.atria_dataloader import SDFDataModule
from utils.metrics import chamfer_distance_L2, LDDMM_loss
from utils.surface_utils import remesh, make_trimesh_from_pv, scale_to_unit_sphere
from utils.reconstruction_utils import isosurface_from_sdf
from utils.visual_utils import plot_gt_vs_reconstructed_with_error
from vtk import vtkImplicitPolyDataDistance
from tqdm import tqdm
import pandas as pd

from config import (
    EXPERIMENTS_DIR,
    IMAGES_DIR,
    RECONSTRUCTED_MESHES_DIR,
    LATENTS_DIR,
    METRICS_DIR,
    PATIENT_MESHES_DIR
)


def get_dataset_patients_names(data: dict):
    patient_names = []

    # file names are <patient_name>-<suffix>.npy where suffix isn't supposed to have any - in it
    for fullfname in data:
        patient_name = fullfname.split("-")[0]
        patient_names.append(patient_name)

    return patient_names


def compute_chd_dists(mesh_orig, mesh_organ):

    samples_orig = make_trimesh_from_pv(mesh_orig).sample(count=50000)
    samples_rec = make_trimesh_from_pv(mesh_organ).sample(count=50000)

    chd = chamfer_distance_L2(samples_orig, samples_rec)

    # scale by some characteristic scale
    xmin, xmax, ymin, ymax, zmin, zmax = mesh_orig.bounds
    pmin = np.array([xmin, ymin, zmin])
    pmax = np.array([xmax, ymax, zmax])
    s_bbox = np.linalg.norm(pmax - pmin)

    return chd / s_bbox


# ======================== #
# RUN TESTS
# ======================== #
def run(
        experiment_name,
        version,
        override_with_test_dataset = None,
        hparams_file = None,
        num_epochs_fit_latent = 250,
        latent_reg_factor = 2e-3,
        lr_fit_latent = 5e-3,
        reconstruct_surface = True,
        reconstruct_from = "all",
        show_reconstruction_images = True,
        save_reconstruction_images = False,
        save_reconstructed_mesh = False,
        compute_metrics = True,
        save_latent_codes = True
    ):

    logger.info(f"Experiment: {experiment_name}")

    # get specifics for the wanted run
    version_dir = EXPERIMENTS_DIR / experiment_name / version

    if hparams_file is None:
        hparams_file =  version_dir / "hparams.json"

    specs = json.load( open(hparams_file) )
    
    # manual test dataset
    if override_with_test_dataset is not None:
        specs["TestSplit"] = override_with_test_dataset

    logger.info(f"Loaded specs from version: {version_dir.name}")

    # rebuild trained decoder and model: I do it with specs that contain everything alerady, no need for checkpoints now
    decoder_params = specs["Network_specs"]

    decoder = Decoder(**decoder_params)

    print("\n")
    print(decoder.description())
    print("\n")
    

    # get trained model parameters
    decoder_weights_path = version_dir / "decoder_weights.pth"
    if decoder_weights_path.is_file():
        state_dict = torch.load(decoder_weights_path)
        decoder.load_state_dict(state_dict)
    else:
        logger.warning(f"Decoder weights not found for {version}")
        return

    decoder.cuda()

    # dataset
    dataloader = SDFDataModule(specs = specs)

    dataloader.setup("test")

    dataset = dataloader.test_dataloader().dataset

    # retrieve original patient names in current dataset
    data_file = dataset.data_file
    patient_names = get_dataset_patients_names(json.load(open(data_file)))

    decoder_input_scale = specs.get("scale_spatial_inputs_by", 100)

    enforce_minmax = specs.get("enforce_minmax", False)
    clamp_distance = specs.get("clamp_distance", 0.1)

    chamfer_dists = {}

    LDDMM_losses = {}

    latent_codes = {}
        
    for shape_idx in range( len(dataset) ):

        patient_name = patient_names[shape_idx] # careful

        batch = dataset[shape_idx]

        data = batch[0]

        chamfer_dists[patient_name] = {}

        LDDMM_losses[patient_name] = {}

        xyz_gt = data["coords"]
        sdf_gt = data["sdf"]
        
        if reconstruct_from == "la":
            logger.warning("Fitting latent code using only points near left atrium")
            near_la = np.where(np.abs(sdf_gt[:,1]) <=  0.005)
            xyz_gt = xyz_gt[near_la]
            sdf_gt = sdf_gt[near_la]

        xyz_gt = xyz_gt.reshape(-1, 3) * decoder_input_scale
        sdf_gt = sdf_gt.reshape(-1, decoder.out_dim)
        if enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, min = -clamp_distance, max = clamp_distance)

        # print(f"##### ===== PATIENT {patient_name} : {shape_idx}/{len(dataset)} ===== #####")
        print("\033[48;2;30;30;30;0;38;2;255;200;0m" + f"##### ===== PATIENT {patient_name} : {shape_idx+1}/{len(dataset)} ===== #####" + "\033[0m")
     
        sdf_gt = sdf_gt.cuda()
        xyz = xyz_gt.cuda()

        # starting point for optimization (same I use initializing latent codes in training): zero
        # I could also save initial vectors when training and start with empirical mean and covariance,
        # sampling a latent using MultivariateNormal and rsample()
        latent_size = decoder.latent_size
        mean_code = torch.zeros(latent_size, device="cuda")  
        latent = mean_code
        latent.requires_grad = True
        
        loss_l1 = torch.nn.L1Loss(reduction="sum")

        code_reg_lambda = latent_reg_factor

        num_epochs = num_epochs_fit_latent

        optimizer = torch.optim.Adam(params=[latent], lr=lr_fit_latent)
        
        num_samp_per_scene = sdf_gt.shape[0]

        chunk_losses= []

        reg_losses = []

        losses = []

        # ==================================================== #
        # region fit latent
        # ==================================================== #
        for i in tqdm(range(num_epochs)):
            
            decoder.eval()
            
            optimizer.zero_grad()

            batch_vecs = latent.expand(num_samp_per_scene, -1)
            
            input_ = torch.cat([batch_vecs, xyz], dim=1)

            sdf_pred = decoder(input_)
            if enforce_minmax:
                sdf_pred = torch.clamp(sdf_pred, min = -clamp_distance, max = clamp_distance)
                
            # mahalanobis to train codes distribution

            # vanilla loss : same loss as in training
            reg_loss = torch.sum( torch.linalg.norm(latent) ) * code_reg_lambda

            chunk_loss = loss_l1(sdf_pred, sdf_gt) / num_samp_per_scene

            loss = chunk_loss + reg_loss

            loss.backward()

            optimizer.step()

            chunk_losses.append(chunk_loss.cpu().detach().numpy())
            reg_losses.append(reg_loss.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())

        latent.requires_grad = False

        if save_latent_codes:
            latent_codes[patient_name] = latent.cpu().numpy().ravel()  # keep fitted latent code, keep latent on gpu for reconstruction


        if reconstruct_surface:
            # ==================================================== #
            # region reconstruct surface from predicted sdf
            # ==================================================== #
            resolution = 128
            box_lim = decoder_input_scale * 1.05

            with torch.no_grad():
                
                logger.info("Computing SDF on grid for reconstruction")

                decoder.eval()

                #region SDF ON GRID FOR RECONSTRUCTION
                x = np.linspace(-box_lim, box_lim, resolution)
                y = np.linspace(-box_lim, box_lim, resolution)
                z = np.linspace(-box_lim, box_lim, resolution)
                xx, yy, zz = np.meshgrid(x, y, z)

                grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
                xyz_raw = grid

                n_batches = len(xyz_raw) // 250000

                sdf_preds = []

                for i in range(n_batches + 1):
                    if i < n_batches:
                        print(250000 * i, 250000 * (i + 1))
                        xyz = torch.from_numpy(xyz_raw[250000 * i : 250000 * (i + 1)]).cuda()
                    else:
                        print(250000 * i, ": ")
                        xyz = torch.from_numpy(xyz_raw[250000 * i :]).cuda()

                    if decoder.use_positional_encoding:
                        freqs = 2.0 ** torch.arange(decoder.pos_enc_dim)
                        x_proj = [xyz]
                        for freq in freqs:
                            x_proj.append(torch.sin(freq * xyz / 100))
                            x_proj.append(torch.cos(freq * xyz / 100))
                        xyz = torch.cat(x_proj, dim=1)

                    batch_vecs = latent.expand(xyz.shape[0], -1)

                    input_ = torch.cat([batch_vecs, xyz], dim=1)

                    sdf_pred_batch = decoder(input_.to(torch.float32)).cpu().data.numpy() 

                    sdf_preds.append(sdf_pred_batch)
                
            sdf_pred = np.concatenate(sdf_preds, axis=0)
                    
            
            # ==================================================== #
            # region process model prediction
            # ==================================================== #
            if decoder.out_dim == 3:
                # assume sdf is distance from epicardium, left endocardium, right endocardium                        
                sdfs_pred = {
                        "epicardium": sdf_pred[:, 0],
                        "la_endo": sdf_pred[:, 1],
                        "ra_endo": sdf_pred[:, 2]
                    }

                thresholds = {}
                
                for organ in ["epicardium", "la_endo", "ra_endo"]:

                    logger.info(f"Processing {organ} surface ")
                        
                    threshold = 0.0 

                    #TODO: optionally minimize for each surface with differential evolution
                    thresholds[organ] = threshold
                                           
                    try:
                        mesh_reconstructed = isosurface_from_sdf( x, y, z, sdf_pred=sdfs_pred[organ], level = threshold, box_lim = box_lim )
                    except:
                        logger.warning( f"Version {version}: skipping isosurface extraction: not found for current isovalue")
                        return

                    if save_reconstructed_mesh:
                        logger.info("Saving vtp file")
                        if reconstruct_from == "all":
                            fname = f"{version}-{patient_name}-{organ}.vtp"
                        if reconstruct_from == "la":
                            fname = f"{version}-{patient_name}-{organ}-from_la_only.vtp"

                        mesh_reconstructed.save(RECONSTRUCTED_MESHES_DIR / fname )

                    patient_dir = PATIENT_MESHES_DIR / patient_name
                    mesh_file = next( patient_dir.rglob(f"{organ}-processed.vtp"), None) # !!! these are assumed to be standardized, unit scale already.
                    mesh_gt = pv.read(mesh_file)
                    # bring reconstructed at the same scale 
                    # the training and fitting points are sampled from these meshes, then multiplied as input to the network optionally --> remove decoder scale from the reconstructed mesh
                    mesh_reconstructed.points /= decoder_input_scale 

                    if compute_metrics: 
                        # ======= COMPUTE METRICS ======= # 
                        # bring meshes back to either ORIGINAL scale, OR STANDARDIZED, UNIT SCALE before computing metrics !!!
                        logger.info("Remeshing")
                        mesh_gt = remesh(mesh_gt, n_points=30000)
                        mesh_reconstructed = remesh(mesh_reconstructed, n_points=30000)

                        logger.info(f"Computing chamfer")
                        chamfer_dists[patient_name][organ] = compute_chd_dists(mesh_gt, mesh_reconstructed)

                        logger.info(f"Computing LDDMM")
                        LDDMM_losses[patient_name][organ] = LDDMM_loss(mesh_gt, mesh_reconstructed, remeshing=False)

                    if show_reconstruction_images or save_reconstruction_images:
                        
                        # TODO: add again rescaling to actual original scale in micrometers
                        scale = 1.0
                        # ==== compute sdf of points on predicted surface to the original surface ==== #               
                        # compute (signed!) distances of ground truth points (on the true surface) from the nearest spot on the predicted mesh surface
                        implicit_distance = vtkImplicitPolyDataDistance()
                        implicit_distance.SetInput(mesh_gt)
                        points_pred = mesh_reconstructed.points
                        signed_distances = np.array([implicit_distance.EvaluateFunction(p) for p in points_pred])
                        mesh_reconstructed.point_data['error'] = signed_distances # save as point data in the predicted mesh

                        last_cam_pos = None
                        if show_reconstruction_images:
                            plotter = plot_gt_vs_reconstructed_with_error(
                                mesh_gt, mesh_reconstructed, patient_name, signed_distances, off_screen = False, scale = scale
                                )
                            plotter.show(interactive=True)
                            last_cam_pos = plotter.camera_position # this used if interactive AND saving images later
                            plotter.close()

                        if save_reconstruction_images:
                            logger.info("Saving plot screenshot")
                            save_fname = patient_name + f"_{organ}_gt_vs_reconstructed_with_error_" + version + ".png"
                            save_fname = IMAGES_DIR / save_fname
                            plotter = plot_gt_vs_reconstructed_with_error(
                                mesh_gt, mesh_reconstructed, patient_name, signed_distances, off_screen=True, scale = scale #decoder_input_scale
                            )
                            if last_cam_pos is not None: plotter.camera_position = last_cam_pos
                            plotter.screenshot(save_fname, transparent_background=True)
                            pv.close_all()

    if compute_metrics:
        logger.info("Saving metrics data to csv")

        # chamfer
        rows = []
        for name, organs in chamfer_dists.items():
            for organ, metric in organs.items():
                rows.append({
                    "version": int(version.split("_")[-1]),
                    "patient": name,
                    "organ": organ,
                    "metric": "chamfer",
                    "value": metric,
                })
        df = pd.DataFrame(rows)
        df.to_parquet(METRICS_DIR / f"{version}-chamfer.parquet", index=False)

        # LDDMM
        rows = []
        for name, organs in LDDMM_losses.items():
            for organ, metric in organs.items():
                rows.append({
                    "version": int(version.split("_")[-1]),
                    "patient": name,
                    "organ": organ,
                    "metric": "LDDMM",
                    "value": metric,
                })
        df = pd.DataFrame(rows)
        df.to_parquet(METRICS_DIR / f"{version}-LDDMM.parquet", index=False)

    if save_latent_codes:
        logger.info("Saving fitted latents")
        np.savez(LATENTS_DIR / f"latent_codes_{len(latent_codes.keys())}_patients_{version}-codereg={code_reg_lambda:.6f}-epochs={num_epochs}.npz", **latent_codes)

    return


       


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", type=str, default = "deepsdfatria_training_concurrent")
    parser.add_argument("--version", "-v", type=str, default = "version_114")
    parser.add_argument("--override_with_test_dataset", "-od", type=str, default=None)
    parser.add_argument("--mode", "-m", type=int, default=1, choices=[1, 2])
    parser.add_argument("--save_latent_codes", "-sc", action="store_true")
    parser.add_argument("--interactive_images", "-i", action="store_true")
    parser.add_argument("--save_images", "-si", action="store_true")
    parser.add_argument("--save_reconstructed_meshes", "-sm", action="store_true")
    parser.add_argument("--compute_chamfer", "-chd", action="store_true")
    parser.add_argument("--compute_lddmm", "-lddmm", action="store_true")
    args = parser.parse_args()

    exp_name = args.experiment_name
    vers = args.version
    test_datafnames = args.override_with_test_dataset if args.override_with_test_dataset is not None else "test/data_fnames-AF001-LEU_NORM_F004.json"
    mode = args.mode 
    num_epochs_fit_latent = 250
    latent_reg_factor = 2e-4
    lr_fit_latent = 5e-3

    kwargs = {
        "experiment_name" : exp_name,
        "version" : vers,
        "override_with_test_dataset" : test_datafnames,
        "num_epochs_fit_latent" : num_epochs_fit_latent,
        "latent_reg_factor" : latent_reg_factor,
        "lr_fit_latent" : lr_fit_latent,
    }

    match mode:

        case 1: # reconstruct surface, then visualize or compute metrics
            run_kwargs = {
                **kwargs,
                "reconstruct_surface" : True,
                "reconstruct_from" : "all",
                "show_reconstruction_images" : args.interactive_images,
                "save_reconstruction_images" : args.save_images,
                "save_reconstructed_mesh" : args.save_reconstructed_meshes,
                "compute_metrics" : args.compute_chamfer or args.compute_lddmm,
                "save_latent_codes" : args.save_latent_codes
            }

        case 2: # only fit latents and save
            run_kwargs = {
                **kwargs,
                "reconstruct_surface" : False,
                "show_reconstruction_images" : False,
                "save_reconstruction_images" : False,
                "save_reconstructed_mesh" : False,
                "compute_metrics" : False,
                "save_latent_codes" : True
            }

    run(**run_kwargs)