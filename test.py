import json
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import re
from tabulate import tabulate
import pyvista as pv
from model.atria_deepsdf_decoder import Decoder, DeepSDF
from model.atria_dataloader import SDFDataModule
from utils.visual_utils import plot_gt_vs_reconstructed_with_error, vtkImplicitPolyDataDistance
from utils.metrics import chamfer_distance
from utils.reconstruction_utils import isosurface_from_sdf, remesh
import trimesh
from scipy.spatial import KDTree
# from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
import pandas as pd

from config import EXPERIMENTS_DIR, RESULTS_DIR, IMAGES_DIR, RECONSTRUCTED_MESHES_DIR, LATENTS_DIR, TEST_DATA_DIR, PATIENT_MESHES_DIR, PATIENTS_NPY_DATA_DIR, DATA_DIR

def get_dataset_patients_names(data_file):
    patient_names = []
    if data_file.suffix == ".json":
        loaded = json.load( open(data_file) )
        
        # file names are <patient_name>-<suffix>.npy where suffix isn't supposed to have any - in it
        for fullfname in loaded:
            patient_name = fullfname.split("-")[0]
            patient_names.append(patient_name)

    return patient_names

scale_to_unit = lambda points: (points - np.mean(points, axis = 0)) / np.max( np.linalg.norm(points - np.mean(points, axis = 0), axis=1) )

def make_trimesh_from_pv(mesh):
    faces = mesh.faces.reshape((-1, 4))[:, 1:] 
    vertices = mesh.points
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def compute_chd_dists(mesh_orig, mesh_organ):
    # remesh first: may take some time!
    logger.info("Remeshing ...")

    mesh_orig_remeshed = remesh(mesh_orig, n_points=100000)
    mesh_reconstructed_remeshed = remesh(mesh_organ, n_points=100000)
    samples_orig = make_trimesh_from_pv(mesh_orig_remeshed).sample(count=50000)
    samples_rec = make_trimesh_from_pv(mesh_reconstructed_remeshed).sample(count=50000)

    chd = chamfer_distance(samples_orig, samples_rec)

    # scale by some characteristic scale
    xmin, xmax, ymin, ymax, zmin, zmax = mesh_orig_remeshed.bounds
    pmin = np.array([xmin, ymin, zmin])
    pmax = np.array([xmax, ymax, zmax])
    s_bbox = np.linalg.norm(pmax - pmin)
    # centroid = mesh_orig_remeshed.points.mean(axis=0)
    # s_mean_radial = np.linalg.norm(mesh_orig_remeshed.points - centroid, axis=1).mean()
    # tree = KDTree(mesh_orig_remeshed.points)
    # dists, idx = tree.query(mesh_orig_remeshed.points, k=2)
    # nn_dist = dists[:, 1]   # ignore self-match
    # s_median_nn = np.median(nn_dist)

    # chd_bbox = chd / s_bbox
    # chd_radial = chd / s_mean_radial
    # chd_nn = chd / s_median_nn

    # # scale to unit the meshes first
    # mesh_orig_remeshed.points = scale_to_unit(mesh_orig_remeshed.points)
    # mesh_orig_remeshed.points = scale_to_unit(mesh_orig_remeshed.points)
    # samples_orig = make_trimesh_from_pv(mesh_orig_remeshed).sample(count=50000)
    # samples_rec = make_trimesh_from_pv(mesh_reconstructed_remeshed).sample(count=50000)

    # chd_scale_to_unit = chamfer_distance(samples_orig, samples_rec)

    # # scale by some characteristic scale
    # xmin, xmax, ymin, ymax, zmin, zmax = mesh_orig_remeshed.bounds
    # pmin = np.array([xmin, ymin, zmin])
    # pmax = np.array([xmax, ymax, zmax])
    # s_bbox = np.linalg.norm(pmax - pmin)
    # centroid = mesh_orig_remeshed.points.mean(axis=0)
    # s_mean_radial = np.linalg.norm(mesh_orig_remeshed.points - centroid, axis=1).mean()
    # tree = KDTree(mesh_orig_remeshed.points)
    # dists, idx = tree.query(mesh_orig_remeshed.points, k=2)
    # nn_dist = dists[:, 1]   # ignore self-match
    # s_median_nn = np.median(nn_dist)

    # chd_bbox_scale_to_unit = chd / s_bbox
    # chd_radial_scale_to_unit = chd / s_mean_radial
    # chd_nn_scale_to_unit = chd / s_median_nn

    # chds = {
    #     "chd" : chd,
    #     "chd_scale_to_unit" : chd_scale_to_unit,
    #     "chd_bbox" : chd_bbox,
    #     "chd_radial" : chd_radial,
    #     "chd_nn" : chd_nn,
    #     "chd_bbox_scale_to_unit" : chd_bbox_scale_to_unit,
    #     "chd_radial_scale_to_unit" : chd_radial_scale_to_unit,
    #     "chd_nn_scale_to_unit" : chd_nn_scale_to_unit
    # }

    return chd / s_bbox

def plot_code_losses(l,cl,rl):
    
    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(l)), l, c='r',label='loss')
    plt.plot(np.arange(len(cl)), cl, c='b', label='chunk loss')
    plt.plot(np.arange(len(rl)), rl, c = 'orange', label='reg loss')
    plt.legend()
    plt.show()

    return

# ======================== #
# RUN TESTS
# ======================== #
def run(
        experiment_name,
        version,
        hparams_file = None,
        mode = "process_test_dataset",
        override_with_test_dataset = None,
        reconstruct_surface = True,
        reconstruct_from = "all",
        show_reconstruction_images = True,
        save_reconstruction_images = False,
        save_reconstructed_mesh = False,
        save_chd_results = True,
        save_latent_codes = True,
        return_chd_results = False
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
        model = DeepSDF(decoder=decoder, specs = specs).cuda()
    else:
        logger.warning(f"Decoder weights not found for {version}")
        return

    # else: # load from last checkpoint
    #     checkpoint_dir = version_dir / "checkpoints"
    #     ckpt_file = max( checkpoint_dir.glob("*.ckpt"), key = lambda file: int( re.search(r"epoch_([0-9]+)", file.name ).group(1) ) )
    #     checkpoint = torch.load(ckpt_file, map_location="cpu")
    #     state_dict = checkpoint["state_dict"]  
    #     logger.warning(f"Loading state dict from ckpt file: {ckpt_file}")
    #     model = DeepSDF(decoder=decoder, specs = specs).cuda()
    #     model.load_state_dict(state_dict)    

    
    model.set_embedding( num_scenes = 1 ) # will process one shape at a time

    # region mode
    match mode:

        case "process_test_dataset":

            dataloader = SDFDataModule(specs = specs)

            dataloader.setup("test")

            dataset = dataloader.test_dataloader().dataset

            # retrieve original patient names in current dataset
            data_file = dataset.data_file
            patient_names = get_dataset_patients_names(data_file)

            decoder_input_scale = specs.get("scale_spatial_inputs_by", 100)

            enforce_minmax = specs.get("enforce_minmax", False)
            clamp_distance = specs.get("clamp_distance", 0.1)

            chamfer_dists = {}

            latent_codes = {}
            
            for shape_idx in range( len(dataset) ):

                patient_name = patient_names[shape_idx] # careful

                batch = dataset[shape_idx]

                data = batch[0]

                chamfer_dists[patient_name] = {}

                # TODO: just pick points from only LA or RA to see the reconstruction power then ...
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

                print("PATIENT: ", patient_name)

                # latent = fit_latent_to_single_shape(xyz_gt, sdf_gt, model, decoder)

                # region fit latent      
                sdf_gt = sdf_gt.cuda()
                xyz = xyz_gt.cuda()

                if decoder.use_positional_encoding:
                    xyz = model.positional_encoding(xyz)

                # starting point for optimization (same I use initializing latent codes in training)
                # I could also save initial vectors when training and start with empirical mean and covariance,
                # sampling a latent using MultivariateNormal and rsample()
                latent_size = decoder.latent_size
                mean_code = torch.zeros(latent_size, device="cuda")  
                latent = mean_code
                latent.requires_grad = True
                
                loss_l1 = torch.nn.L1Loss(reduction="sum")

                code_reg_lambda = 2e-4

                num_epochs = 300

                optimizer = torch.optim.Adam(params=[latent], lr=5e-3)
                
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

                # plot_code_losses(losses, chunk_losses, reg_losses)

                latent.requires_grad = False

                latent_codes[patient_name] = latent.cpu().numpy().ravel()  # save fitted latent code, keep latent on gpu for reconstruction


                if reconstruct_surface:
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

                        thresholds = {}
                        
                        for organ in [ "epicardium" ]: #, "la_endo", "ra_endo"]: #

                            logger.info(f"Processing {organ} surface ")
                                
                            threshold = 0.0 

                            #TODO: optionally minimize for each surface with differential evolution
                            thresholds[organ] = threshold
                            
                            try:
                                mesh_organ = isosurface_from_sdf( x, y, z, sdf_pred=sdfs_pred[organ], level = thresholds[organ], box_lim = box_lim )
                            except:
                                logger.warning( f"Version {version}: skipping isosurface extraction: not found for current isovalue")
                                return
                                                        
                            patient_dir = PATIENT_MESHES_DIR / patient_name
                            mesh_file = next( patient_dir.rglob(f"{organ}-processed.vtp"), None)
                            mesh_orig = pv.read(mesh_file)
                            mesh_orig.points *= specs.get("scale_spatial_inputs_by", 100) #  scale to decoder range !
                            
                            # compute chamfer distance between point clouds near the surface
                            if save_chd_results:
                                chamfer_dists[patient_name][organ] = compute_chd_dists(mesh_orig, mesh_organ)

                            # region plots
                            if show_reconstruction_images or save_reconstructed_mesh or save_reconstruction_images:
                                mesh_pred = pv.wrap(mesh_organ)   

                                # RETRIEVE ORIGINAL MESH                             
                                # # ===== this is original mesh for point files like curv_xxxxx_coords_and_sdfs : every mesh been scaled with its own value ====== #
                                # patient_dir = PATIENT_MESHES_DIR / patient_name
                                # mesh_file = next( patient_dir.rglob(f"{organ}-processed.vtp"), None)
                                # mesh_orig = pv.read(mesh_file)

                                # ===== this is for point files like <name>_epi_la_ra_xxxxxpts_coords_and_sdfs.npy, for which
                                # meshes had been scaled all with the same value to maintain relative dimension ===== #
                                mesh_file = next( patient_dir.rglob(f"{organ}-processed.vtp"), None)
                                mesh_orig = pv.read(mesh_file)
                                scale = mesh_orig.field_data["scale_to_original_range-oneforall"]
                                mesh_orig.points *= scale

                                mesh_pred.points /= 100 # back to unit scale
                                mesh_pred.points *= scale
                            
                                # ==== compute sdf of points on predicted surface to the original surface ==== #               
                                # compute (signed!) distances of ground truth points (on the true surface) from the nearest spot on the predicted mesh surface
                                implicit_distance = vtkImplicitPolyDataDistance()
                                implicit_distance.SetInput(mesh_orig)
                                points_pred = mesh_pred.points
                                signed_distances = np.array([implicit_distance.EvaluateFunction(p) for p in points_pred])
                                # errors = np.abs(signed_distances)
                                mesh_pred.point_data['error'] = signed_distances # save as point data in the predicted mesh

                            if show_reconstruction_images: 
                                plotter = plot_gt_vs_reconstructed_with_error(
                                    mesh_orig, mesh_pred, patient_name, signed_distances, off_screen = False, scale = scale #decoder_input_scale
                                    )
                                plotter.show(interactive=True)
                                last_cam_pos = plotter.camera_position
                                plotter.close()

                            if save_reconstruction_images:
                                    logger.info("Saving plot screenshot")
                                    save_fname = patient_name + f"_{organ}_gt_vs_reconstructed_with_error_" + version + ".png"
                                    save_fname = IMAGES_DIR / save_fname
                                    plotter = plot_gt_vs_reconstructed_with_error(
                                        mesh_orig, mesh_pred, patient_name, signed_distances, off_screen=True, scale = scale #decoder_input_scale
                                    )
                                    plotter.camera_position = last_cam_pos
                                    plotter.screenshot(save_fname, transparent_background=True)
                                    pv.close_all()
                          
                            if save_reconstructed_mesh:
                                logger.info("Saving vtp file")
                                if reconstruct_from == "all":
                                    fname = f"reconstructed_{patient_name}_{version}_{organ}-res={resolution}.vtp"
                                if reconstruct_from == "la":
                                    fname = f"reconstructed_from_LA_{patient_name}_{version}_{organ}-res={resolution}.vtp"

                                mesh_organ.save(RECONSTRUCTED_MESHES_DIR / fname )

            if save_latent_codes:
                logger.info("Saving fitted latents")
                np.savez(LATENTS_DIR / f"latent_codes_{len(latent_codes.keys())}_patients_{version}-codereg={code_reg_lambda:.6f}-epochs={num_epochs}.npz", **latent_codes)

            # if save_chd_results and chamfer_dists:
            #     logger.info("Saving chamfer distances data")
            #     # pd.DataFrame.from_dict(chamfer_dists).to_csv(
            #     #     RESULTS_DIR / "metrics" / f"chamfer_distances_{version}.csv"
            #     # )
            #     rows = []
            #     for name, organs in chamfer_dists.items():
            #         for organ, metrics in organs.items():
            #             row = {"name": name, "organ": organ, **metrics}
            #             rows.append(row)
            #     df = pd.DataFrame(rows)
            #     df.to_csv(RESULTS_DIR / "metrics" / f"chamfer_distances_{version}.csv", index=False)
            # elif return_chd_results and chamfer_dists:
            #     return chamfer_dists

                # pd.DataFrame.from_dict(latent_codes).to_csv(RESULTS_DIR/ "fitted_latents" / f"latent_codes_{version}.csv")

    return

def run_multiple(
        experiment_name,
        versions = [],
        mode = "process_test_dataset",
        test_patient_names=[],
        save_latent_codes=False, 
        reconstruct_surface=True,
        save_chd_results=False,
        save_reconstruction_images=False,
        return_chd_results=True
    ):

    # create json filename to test on wanted patients, if it doesn't exists yet
    names = ""
    for name in test_patient_names:
        names += f"{name}_"
    test_fnames = "all_train_data_fnames.json" # this is the structure in which I saved specific patients filenames
    test_fnames = TEST_DATA_DIR / test_fnames
    if not test_fnames.is_file(): # create it
        patient_files = [file.name for file in PATIENTS_NPY_DATA_DIR.iterdir() if any( name in file.name for name in test_patient_names)]
        data = []
        for file in patient_files:
            fname = PATIENTS_NPY_DATA_DIR.name + "/" + file
            pf = DATA_DIR / fname
            if Path(pf).stat().st_size > 0:  
                data.append(str(fname))
        with open(TEST_DATA_DIR / test_fnames, "w") as f:
            json.dump(data, f)

    # run test for every version
    versions_chd = {}

    for v_num in versions:

        chamfer_dists = run(
            experiment_name=experiment_name,
            version=f"version_{v_num}",
            mode=mode,
            override_with_test_dataset=test_fnames,
            save_latent_codes=save_latent_codes, 
            reconstruct_surface=reconstruct_surface,
            save_chd_results=save_chd_results,
            save_reconstruction_images=save_reconstruction_images,
            return_chd_results=return_chd_results
        )

        versions_chd[v_num] = chamfer_dists


    # =============== #
    # build dataframe
    # =============== #
    rows = []
    for v_num, chamfer_dists in versions_chd.items():
        for patient, organs in chamfer_dists.items():
            row = {"patient": patient, "version": v_num}
            row.update(organs)  # adds all organ:value pairs
            rows.append(row)
    df = pd.DataFrame(rows)

    # print and save top 3 versions per patient
    organs = ["epicardium", "la_endo", "ra_endo"]
    # Compute mean performance
    df["mean_perf"] = df[organs].mean(axis=1)
    # Reset index for safety
    df_reset = df.reset_index(drop=True)
    # Sort by patient and mean performance (lowest first)
    df_sorted = df_reset.sort_values(["patient","mean_perf"], ascending=[True,True])
    # Keep top 3 versions per patient
    df_top3 = df_sorted.groupby("patient").head(3).copy()  # copy to be sure
    # Get index of best version per patient among top 3
    best_idx = df_top3.groupby("patient")["mean_perf"].idxmin()
    table_rows = []
    for i, row in df_top3.iterrows():
        version_val = str(row["version"])
        if i in best_idx.values:  # mark best version
            version_val += " *"
        row_vals = [row["patient"], version_val] + [row[o] for o in organs]
        table_rows.append(row_vals)
    print(tabulate(table_rows, headers=["patient", "version"] + organs, tablefmt="grid"))

    # Add a column for marked version
    df_top3["version_marked"] = df_top3["version"].astype(str)
    df_top3.loc[best_idx, "version_marked"] += " *"
    # Save to CSV
    df_top3.to_csv( RESULTS_DIR / "metrics" / str("all_train" + "_chamfer_top3_marked.csv"), index=False, columns=["patient", "version_marked"] + organs)
            

    
       


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default = "deepsdfatria_training_local")
    parser.add_argument("--version", type=str, default = "version_0")
    parser.add_argument("--mode", type=str, default = "process_test_dataset")
    args = parser.parse_args()

    run(
        experiment_name=args.experiment_name,
        version=args.version,
        # hparams_file="/home/davidenava_linux/AtriaProject/deepcsdf_fork/deepcsdf/deepcsdf/deepcsdfatria/specs_files_concurrent/specs_5.json",
        mode=args.mode,
        override_with_test_dataset= "test/AF009_P2R-LEU_NORM_F004.json",
        reconstruct_surface=True,
        reconstruct_from="all",
        show_reconstruction_images=True,
        save_reconstruction_images=False,
        save_reconstructed_mesh = False,
        save_chd_results=False, 
        save_latent_codes=False,
    )

    # # run multiple version to compare on the same patients
    # run_multiple(
    #     experiment_name="deepsdf_atria_training_concurrent",
    #     test_patient_names=["AF001"],
    #     versions=[int(n) for n in np.arange(60,96+1)]
    # )

    # files = list( Path("/home/davidenava_linux/AtriaProject/deepcsdf_fork/deepcsdf/deepcsdf/deepcsdfatria/results/metrics").iterdir())
    # for file in files:

    #     # Load CSV
    #     df = pd.read_csv(file, index_col=0)

    #     v_num = (str(file.name).split(".")[0]).split("_")[-1]

    #     chamfer_dists = {}
    #     # Iterate over columns (patients)
    #     for patient in df.columns:
    #         chamfer_dists[patient] = df[patient].to_dict()

    #     versions_chd[v_num] = chamfer_dists

    # versions = [int(n) for n in np.arange(60,96+1)]

    # for v_num in versions:

    #     run(
    #         experiment_name="deepsdf_atria_training_concurrent",
    #         version=f"version_{v_num}",
    #         override_with_test_dataset="test/AF001_data_fname.json",
    #         save_latent_codes=False, 
    #         reconstruct_surface=True,
    #         show_reconstruction_images=False,
    #         save_chd_results=False,
    #         save_reconstruction_images=True,
    #         return_chd_results=False
    #     )