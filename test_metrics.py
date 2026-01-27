import json
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import pyvista as pv
from model.atria_deepsdf_decoder import Decoder, DeepSDF
from model.atria_dataloader import SDFDataModule
from utils.metrics import chamfer_distance_L2, LDDMM_loss
from utils.surface_utils import remesh, make_trimesh_from_pv, scale_to_unit_sphere
from utils.reconstruction_utils import isosurface_from_sdf
from tqdm import tqdm
import pandas as pd

from config import (
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    IMAGES_DIR,
    RECONSTRUCTED_MESHES_DIR,
    LATENTS_DIR,
    TEST_DATA_DIR,
    PATIENT_MESHES_DIR,
    PATIENTS_NPY_DATA_DIR,
    DATA_DIR
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
        override_with_dataset = None,
        all_train_shapes = False,
        all_test_shapes = False,
        reconstruct_from = "all",
        num_epochs_fit_latent = 250,
        latent_reg_factor = 2e-3,
        lr_fit_latent = 5e-3,
        save_metrics = True,
        save_latent_codes = True,
        return_metrics_results = False
    ):

    logger.info(f"Experiment: {experiment_name}")

    # get specifics for the wanted run
    version_dir = EXPERIMENTS_DIR / experiment_name / version

    hparams_file =  version_dir / "hparams.json"

    specs = json.load( open(hparams_file) )
    
    # manual test dataset
    if override_with_dataset is not None:
        specs["TestSplit"] = override_with_dataset

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
        model = DeepSDF(decoder=decoder, specs = specs).cuda()
    else:
        logger.warning(f"Decoder weights not found for {version}")
        return

    model.set_embedding( num_scenes = 1 ) # will process one shape at a time


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
        print("\033[48;2;30;30;30;0;38;2;255;200;0m" + f"##### ===== PATIENT {patient_name} : {shape_idx}/{len(dataset)} ===== #####" + "\033[0m")

        # region fit latent      
        sdf_gt = sdf_gt.cuda()
        xyz = xyz_gt.cuda()

        if decoder.use_positional_encoding:
            xyz = model.positional_encoding(xyz)

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

        for i in tqdm(range(num_epochs)):
            
            model.eval()
            
            model.decoder.eval()
            
            optimizer.zero_grad()

            batch_vecs = latent.expand(num_samp_per_scene, -1)
            
            input_ = torch.cat([batch_vecs, xyz], dim=1)

            sdf_pred = model.decoder(input_)
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


        if save_metrics or return_metrics_results: # if not, don't even bother processing the mesh
            # ==================================================== #
            # region reconstruct surface from predicted sdf
            # ==================================================== #
            resolution = 128
            box_lim = decoder_input_scale * 1.05

            with torch.no_grad():
                
                logger.info("Computing SDF on grid for reconstruction")

                model.eval()
                
                model.decoder.eval()

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

                for organ in [ "epicardium", "la_endo", "ra_endo"]: #

                    logger.info(f"Processing {organ} surface ")
                        
                    threshold = 0.0 
                    
                    try:
                        mesh_reconstructed = isosurface_from_sdf( x, y, z, sdf_pred=sdfs_pred[organ], level = threshold, box_lim = box_lim )
                    except:
                        logger.warning( f"Version {version}: skipping isosurface extraction: not found for current isovalue")
                        return
                    
                    patient_dir = PATIENT_MESHES_DIR / patient_name
                    mesh_file = next( patient_dir.rglob(f"{organ}-processed.vtp"), None) # !!! these are assumed to be standardized, unit scale already.
                    mesh_gt = pv.read(mesh_file)

                    # ======= COMPUTE METRICS ======= #
                    # bring meshes back to either ORIGINAL scale, OR STANDARDIZED, UNIT SCALE before computing metrics !!!
                    # to be consistent and not accumulate errors:
                    # the training and fitting points are sampled from these meshes, then multiplied as input to the network optionally --> remove decoder scale from the reconstructed mesh
                    mesh_reconstructed.points /= decoder_input_scale # bring reconstructed at the original scale TODO: does this numerically move vertices a bit so meshes are no more really watertight?
                    
                    logger.info("Remeshing")
                    mesh_gt = remesh(mesh_gt, n_points=10000)
                    mesh_reconstructed = remesh(mesh_reconstructed, n_points=10000)

                    logger.info(f"Computing chamfer")
                    chamfer_dists[patient_name][organ] = compute_chd_dists(mesh_gt, mesh_reconstructed)

                    logger.info(f"Computing LDDMM")
                    LDDMM_losses[patient_name][organ] = LDDMM_loss(mesh_gt, mesh_reconstructed, remeshing=False)


    if save_latent_codes:
        logger.info("Saving fitted latents")
        np.savez(LATENTS_DIR / f"{version}-latents_{len(latent_codes.keys())}_patients_codereg={code_reg_lambda:.6f}_epochs={num_epochs}.npz", **latent_codes)

    if save_metrics:
        logger.info("Saving metrics data to csv")

        if all_train_shapes:
            opt = "trainshapes" 
        elif all_test_shapes:
            opt = "testshapes"
        else:
            opt = "mixed"

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
        df.to_parquet(RESULTS_DIR / "metrics" / f"{version}-chamfer-{opt}.parquet", index=False)

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
        df.to_parquet(RESULTS_DIR / "metrics" / f"{version}-LDDMM-{opt}.parquet", index=False)

    if return_metrics_results:
        return {"chamfer" : chamfer_dists, "LDDMM": LDDMM_losses}
    
    return

   
       


if __name__ == "__main__":
    
    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--experiment_name", type=str, default = "deepsdfatria_training_local")
    # parser.add_argument("--version", type=str, default = "version_0")
    # parser.add_argument("--mode", type=str, default = "process_test_dataset")
    # args = parser.parse_args()

    experiment_name = "deepsdfatria_training_concurrent"

    versions = ["version_89"]

    test_dataset = "train/data_fnames_train-10patientsonly.json" #"test/data_fnames-AF001-LEU_NORM_F004-AF009_P2R-LEU_NORM_0032.json"

    for version in versions:

        run(
            experiment_name=experiment_name,
            version=version,
            override_with_dataset = test_dataset,
            reconstruct_from = "all",
            all_train_shapes=False,
            save_metrics = True,
            save_latent_codes = False,
            return_metrics_results = False
        )




"""
# def run_multiple(
#         experiment_name,
#         versions = [],
#         test_fnames=None,
#         save_latent_codes=False, 
#         save_metrics_for_each=False

#     ):

#     # run test for every version
#     versions_metrics = {}

#     for v_num in versions:

#         metrics = run(
#             experiment_name=experiment_name,
#             version=f"version_{v_num}",
#             override_with_test_dataset=test_fnames,
#             reconstruct_from = "all",
#             save_metrics = save_metrics_for_each,
#             save_latent_codes = save_latent_codes,
#             return_metrics_results = True
#         )

#         versions_metrics[v_num]["chamfer"] = metrics["chamfer"]
#         versions_metrics[v_num]["LDDMM"] = metrics["LDDMM"]


#     # =============== #
#     # build dataframe
#     # =============== #
#     for key, value in versions_metrics.items():
#         rows = []
#         for v_num, vals in value.items():
#             for patient, organs in vals.items():
#                 row = {"patient": patient, "version": v_num}
#                 row.update(organs)  # adds all organ:value pairs
#                 rows.append(row)
#         df = pd.DataFrame(rows)

#         # print and save top 3 versions per patient
#         organs = ["epicardium", "la_endo", "ra_endo"]
#         # Compute mean performance
#         df["mean_perf"] = df[organs].mean(axis=1)
#         # Reset index for safety
#         df_reset = df.reset_index(drop=True)
#         # Sort by patient and mean performance (lowest first)
#         df_sorted = df_reset.sort_values(["patient","mean_perf"], ascending=[True,True])
#         # Keep top 3 versions per patient
#         df_top3 = df_sorted.groupby("patient").head(3).copy()  # copy to be sure
#         # Get index of best version per patient among top 3
#         best_idx = df_top3.groupby("patient")["mean_perf"].idxmin()
#         table_rows = []
#         for i, row in df_top3.iterrows():
#             version_val = str(row["version"])
#             if i in best_idx.values:  # mark best version
#                 version_val += " *"
#             row_vals = [row["patient"], version_val] + [row[o] for o in organs]
#             table_rows.append(row_vals)
#         print(tabulate(table_rows, headers=["patient", "version"] + organs, tablefmt="grid"))

#         # Add a column for marked version
#         df_top3["version_marked"] = df_top3["version"].astype(str)
#         df_top3.loc[best_idx, "version_marked"] += " *"
#         # Save to CSV
#         df_top3.to_csv( RESULTS_DIR / "metrics" / f"{key}-markedtop3.csv", index=False, columns=["patient", "version_marked"] + organs)
            # def run_multiple(
#         experiment_name,
#         versions = [],
#         test_fnames=None,
#         save_latent_codes=False, 
#         save_metrics_for_each=False

#     ):

#     # run test for every version
#     versions_metrics = {}

#     for v_num in versions:

#         metrics = run(
#             experiment_name=experiment_name,
#             version=f"version_{v_num}",
#             override_with_test_dataset=test_fnames,
#             reconstruct_from = "all",
#             save_metrics = save_metrics_for_each,
#             save_latent_codes = save_latent_codes,
#             return_metrics_results = True
#         )

#         versions_metrics[v_num]["chamfer"] = metrics["chamfer"]
#         versions_metrics[v_num]["LDDMM"] = metrics["LDDMM"]


#     # =============== #
#     # build dataframe
#     # =============== #
#     for key, value in versions_metrics.items():
#         rows = []
#         for v_num, vals in value.items():
#             for patient, organs in vals.items():
#                 row = {"patient": patient, "version": v_num}
#                 row.update(organs)  # adds all organ:value pairs
#                 rows.append(row)
#         df = pd.DataFrame(rows)

#         # print and save top 3 versions per patient
#         organs = ["epicardium", "la_endo", "ra_endo"]
#         # Compute mean performance
#         df["mean_perf"] = df[organs].mean(axis=1)
#         # Reset index for safety
#         df_reset = df.reset_index(drop=True)
#         # Sort by patient and mean performance (lowest first)
#         df_sorted = df_reset.sort_values(["patient","mean_perf"], ascending=[True,True])
#         # Keep top 3 versions per patient
#         df_top3 = df_sorted.groupby("patient").head(3).copy()  # copy to be sure
#         # Get index of best version per patient among top 3
#         best_idx = df_top3.groupby("patient")["mean_perf"].idxmin()
#         table_rows = []
#         for i, row in df_top3.iterrows():
#             version_val = str(row["version"])
#             if i in best_idx.values:  # mark best version
#                 version_val += " *"
#             row_vals = [row["patient"], version_val] + [row[o] for o in organs]
#             table_rows.append(row_vals)
#         print(tabulate(table_rows, headers=["patient", "version"] + organs, tablefmt="grid"))

#         # Add a column for marked version
#         df_top3["version_marked"] = df_top3["version"].astype(str)
#         df_top3.loc[best_idx, "version_marked"] += " *"
#         # Save to CSV
#         df_top3.to_csv( RESULTS_DIR / "metrics" / f"{key}-markedtop3.csv", index=False, columns=["patient", "version_marked"] + organs)
            

"""