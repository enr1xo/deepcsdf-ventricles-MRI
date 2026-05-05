# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import json
from pathlib import Path
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import pyvista as pv
from model.deepsdf_decoder import Decoder, DeepSDF
from model.deepsdf_dataloader import SDFDataModule
from utils.metrics import chamfer_distance_L2, LDDMM_loss, haussdorff
from utils.surface_utils import remesh, make_trimesh_from_pv
from utils.reconstruction_utils import isosurface_from_sdf
from utils.visual_utils import plot_gt_vs_reconstructed_with_error
from vtk import vtkImplicitPolyDataDistance
from scipy.interpolate import griddata
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import math

from config import (
    EXPERIMENTS_DIR,
    IMAGES_DIR,
    RECONSTRUCTED_MESHES_DIR,
    LATENTS_DIR,
    METRICS_DIR,
    PATIENT_MESHES_DIR,
    PATIENTS_NPY_DATA_DIR
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using GPU:", torch.cuda.get_device_name(DEVICE))


# ======================== #
# Helpers
# ======================== #
def get_dataset_patients_names(data: dict):
    patient_names = []

    # file names are <patient_name>-<suffix>.npy
    for fullfname in data:
        patient_name = fullfname.split("-")[0]
        patient_names.append(patient_name)

    return patient_names

def save_metrics_csv(metric_name, experiment_name, version, which_shapes, metric_data : dict):
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
    output_path = METRICS_DIR / experiment_name / f"{experiment_name}-{version}-{metric_name}-{which_shapes}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def save_latents_npz(experiment_name, version, which_shapes, code_reg_lambda, num_epochs, init_from, loss_type, latent_codes : dict):
    fname = f"{exp_name}-{version}-latents_{len( set(latent_codes.keys()) )}_{which_shapes}_patients-codereg={code_reg_lambda:.0e}-epochs={num_epochs}-init={init_from}-loss={loss_type}.npz"
    output_path = LATENTS_DIR / experiment_name / fname
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **latent_codes)


def find_pointcloud_noise(
        decoder: Decoder,
        model: DeepSDF,
        xyz_gt,
        sdf_gt,
        num_epochs_fit_latent,
        lr_fit_latent,
        code_reg_lambda,
        max_iter = 10
    ):
    """
    Find noise of input point cloud as in ShapeOfMyHeart, to define regularization strength in reconstruction loss
    """

    latent_size = decoder.latent_size
    mean_code = torch.zeros(latent_size, device=DEVICE)  
    latent = mean_code

    latent.requires_grad = True

    loss_fn = torch.nn.MSELoss(reduction="sum")

    num_epochs = num_epochs_fit_latent

    optimizer = torch.optim.Adam(params=[latent], lr=lr_fit_latent)
    
    num_samp_per_scene = sdf_gt.shape[0]

    epsilon = 0.0

    epsilons = [epsilon]

    # find variance of the noise iteratively
    for it in range(max_iter):

        mean_code = torch.zeros(latent_size, device=DEVICE, requires_grad=True)
        latent = mean_code

        optimizer = torch.optim.Adam(params=[latent], lr=lr_fit_latent)

        # reconstruct
        for i in range(num_epochs):
            
            decoder.eval()
            
            optimizer.zero_grad()

            batch_vecs = latent.expand(num_samp_per_scene, -1)
            
            input_ = torch.cat([batch_vecs, xyz_gt], dim=1)

            sdf_pred = decoder(input_)
            if model.enforce_minmax:
                sdf_pred = torch.clamp(sdf_pred, min = -model.clamp_distance, max = model.clamp_distance)
        
            # vanilla loss
            reg_loss = torch.linalg.norm(latent) ** 2 
            recon_loss = loss_fn(sdf_pred, sdf_gt) 
            chunk_loss = recon_loss / num_samp_per_scene

            loss = chunk_loss + 100 * epsilon * code_reg_lambda * reg_loss

            loss.backward()

            optimizer.step()
        
            if i == num_epochs - 1: # last epoch
                epsilon = np.sqrt( recon_loss.detach().item() / (num_samp_per_scene - 1) ) # detach or on the next epoch it is still attached to the computational graph, instead like this is just a scalar to be reused
                epsilons.append(epsilon)

        # stopping criterion   ...
        tol = 1e-7 
        if abs(epsilons[-1]**2 - epsilons[-2]**2) < tol:
            break
    
    # for i,ep in enumerate(epsilons):
    #     print(f"eps_{i} = ", ep)

    return epsilons[-1]


# ======================== #
# RUN TESTS
# ======================== #
def run(
    experiment_name,
    version,
    override_with_dataset = None,
    num_samp_per_scene_for_fit = None,
    hparams_file = None,
    num_epochs_fit_latent = 250,
    latent_reg_factor = None,
    lr_fit_latent = 5e-3,
    initialize_latent_from = "zero",
    use_mahalanobis_loss=False,
    reconstruct_surface = True,
    reconstruct_from = "all",
    show_reconstruction_images = True,
    save_reconstruction_images = False,
    save_reconstructed_mesh = False,
    compute_chamfer = False,
    compute_lddmm = False,
    compute_haussdorff = False,
    save_latent_codes = True,
    use_old_chamfer_surface_metric = False
):

    # region specs: get specifics for the wanted run
    version_dir = EXPERIMENTS_DIR / experiment_name / version
    # DEBUG
    # version_dir = Path("deepcsdf-atria/experiments/enrico_preliminary_train/version_0")
    # version_dir = Path("experiments/enrico_preliminary_2/version_1")
    # fine debug

    if hparams_file is None:
        hparams_file =  version_dir / "hparams.json"

    specs = json.load( open(hparams_file) )

    # region manual test dataset
    if override_with_dataset is not None:
        specs["TestSplit"] = override_with_dataset

    print(f"\n\n")
    print(f"Experiment: {experiment_name}")
    print(f"Version: {version}")
    print("Specs:")
    pprint(specs)
    print(f"\n")

    # region decoder
    # rebuild trained decoder and model: I do it with specs that contain everything alerady, no need for checkpoints now
    decoder_params = specs["Network_specs"]

    decoder = Decoder(**decoder_params)

    # print("\n")
    # print(decoder.description())
    # print("\n")
    
    # get trained model parameters
    decoder_weights_path = version_dir / "decoder_weights.pth"
    if decoder_weights_path.is_file():
        state_dict = torch.load(decoder_weights_path)
        decoder.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Decoder weights .pth file not found in {version_dir}")

    decoder.to(DEVICE)

    model = DeepSDF(decoder=decoder, specs=specs) 

    # region load dataset
    # optionally set num of subsamples to use --> the dataloader will return scenes with specs["num_samp_per_scene"] points !!!
    if num_samp_per_scene_for_fit is not None:
        specs["num_samp_per_scene"] = num_samp_per_scene_for_fit

    dataloader = SDFDataModule(specs = specs)

    dataloader.setup("test")

    dataset = dataloader.test_dataloader().dataset

    # retrieve original patient names in current dataset
    data_file = dataset.data_file
    which_shapes = "test" if "test" in Path(data_file).name else "train"
    patient_names = get_dataset_patients_names( json.load(open(data_file)) )

    decoder_input_scale = specs.get("scale_spatial_inputs_by", 100)

    enforce_minmax = specs.get("enforce_minmax", False)
    clamp_distance = specs.get("clamp_distance", 0.1)

    if latent_reg_factor is None:
        latent_reg_factor = specs["code_reg_lambda"]

    loss_fn = torch.nn.MSELoss(reduction="sum")

    chamfer_dists = {}

    LDDMM_losses = {}

    haussdorff_dists = {}

    latent_codes = {}

    print(f"\n")
    print("RECONSTRUCTION PARAMETERS")
    print(f"    - {specs['num_samp_per_scene']} samples per scene to fit latents")
    if reconstruct_from == "la":
        print(f"    - Fitting latent code using only points near left atrium")        
    if initialize_latent_from == "zero":
        print(f"    - Latent code initialized from zero")
    elif initialize_latent_from == "normal":
        print(f"    - Latent code initialized from random normal, mean = 0, std = {1.0 / math.sqrt(decoder.latent_size)}")
    elif initialize_latent_from == "empirical":
        print(f"    - Latent code initialized by sampling trained latents distribution")
    if use_mahalanobis_loss:
        print(f"    - Using mahalanobis regularization")
    else:
        print(f"    - Using L2 squared regularization")

    print(f"\n")

    for shape_idx in range( len(dataset) ): # --> the dataloader already returns scenes with specs["num_samp_per_scene"] points each. I call it here only ONE time per shape, so latents are effectively fitted using only these points

        patient_name = patient_names[shape_idx] # careful

        # print(f"##### ===== PATIENT {patient_name} : {shape_idx}/{len(dataset)} ===== #####")
        print("\n \033[48;2;30;30;30;0;38;2;255;200;0m" + f"# {'='*10} PATIENT {patient_name} : {shape_idx+1}/{len(dataset)} {'='*10} #" + "\033[0m")
        
        coords_and_sdf_file = next( PATIENTS_NPY_DATA_DIR.glob(f"{patient_name}*.npy"), None )
        if coords_and_sdf_file is None:
            raise FileNotFoundError(f"Tried to load original points and sdf data from directory \
                                    {PATIENTS_NPY_DATA_DIR} for patient {patient_name}, found None. \
                                    These should be used for chamfer distance computations.")
    
        chamfer_dists[patient_name] = {}

        haussdorff_dists[patient_name] = {}

        LDDMM_losses[patient_name] = {}

        batch = dataset[shape_idx]

        data = batch[0]

        xyz_gt = data["coords"]
        sdf_gt = data["sdf"]

        if reconstruct_from == "la":
            near_la = np.where(np.abs(sdf_gt[:,1]) <=  0.005)
            xyz_gt = xyz_gt[near_la]
            sdf_gt = sdf_gt[near_la]
            
        xyz_gt = xyz_gt.reshape(-1, 3) * decoder_input_scale

        sdf_gt = sdf_gt.reshape(-1, decoder.out_dim)
        if enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, min = -clamp_distance, max = clamp_distance)

        sdf_gt = sdf_gt.to(DEVICE)
        xyz = xyz_gt.to(DEVICE)

        latent_size = decoder.latent_size

        # retrieve trained latents to use in loss and/or to initialize latent code
        if use_mahalanobis_loss or initialize_latent_from == "empirical":
            trained_latents_file = next( version_dir.glob("latents.npy"), None)

            if trained_latents_file is None:
                raise ValueError(f"Requested initialize_latent_from_mean_empirical=True, but trained latents file latents.npy not found in version dir {version_dir}.")
        
            trained_latents = torch.from_numpy( np.load(trained_latents_file)).to(device=DEVICE)
            mean_code = torch.mean(trained_latents, axis=0)
            cov = torch.cov(trained_latents.T)
            cov_inv = cov.inverse()

        # initialize latent
        if initialize_latent_from == "zero":
            latent = torch.zeros(latent_size, device=DEVICE)  
        elif initialize_latent_from == "normal":
            latent = torch.randn(latent_size, device=DEVICE) * ( 1.0 / math.sqrt(latent_size) ) # same std as in training 
        elif initialize_latent_from == "empirical":
            distrib = MultivariateNormal(loc=mean_code, covariance_matrix=cov)
            latent = distrib.rsample()

        latent.requires_grad = True
        
        code_reg_lambda = latent_reg_factor 

        beta = 100 * find_pointcloud_noise(decoder, model, xyz, sdf_gt, code_reg_lambda=code_reg_lambda, num_epochs_fit_latent=250, lr_fit_latent=0.005)

        num_epochs = num_epochs_fit_latent

        optimizer = torch.optim.Adam(params=[latent], lr=lr_fit_latent)
        
        num_samp_per_scene = sdf_gt.shape[0]

        chunk_losses= []

        reg_losses = []

        losses = []

        # ==================================================== #
        # region fit latent
        # ==================================================== #
        print(f"\n Fitting code ... \n")

        for i in tqdm(range(num_epochs)):
            
            decoder.eval()
            
            optimizer.zero_grad()

            batch_vecs = latent.expand(num_samp_per_scene, -1)
            
            input_ = torch.cat([batch_vecs, xyz], dim=1)

            sdf_pred = decoder(input_)
            if enforce_minmax:
                sdf_pred = torch.clamp(sdf_pred, min = -clamp_distance, max = clamp_distance)
            
            if use_mahalanobis_loss:    
                diff = latent - mean_code
                reg_loss = diff @ cov_inv @ diff # mahalanobis to train codes distribution
            else:                       
                reg_loss = latent.pow(2).sum()   # vanilla loss : same loss as in training (pow avoids creating intermediate tensors when doing like latents ** 2)

            chunk_loss = loss_fn(sdf_pred, sdf_gt) / (num_samp_per_scene * decoder.out_dim)

            loss = chunk_loss + beta * code_reg_lambda * reg_loss

            loss.backward()

            optimizer.step()

            chunk_losses.append(chunk_loss.cpu().detach().numpy())
            reg_losses.append(reg_loss.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())
    
        latent.requires_grad = False    

        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(len(losses)), losses)
        # plt.show()


        if save_latent_codes:
            latent_codes[patient_name] = latent.cpu().numpy().ravel()  # keep fitted latent code, keep latent on gpu for reconstruction


        if reconstruct_surface:
            # ==================================================== #
            # region reconstruct surface from predicted sdf
            # ==================================================== #
    
            resolution = 128
            box_lim = 1.05

            with torch.no_grad():
                
                print("\n Computing prediction on grid ...")

                decoder.eval()

                #region SDF ON GRID FOR RECONSTRUCTION
                x = np.linspace(-box_lim, box_lim, resolution)
                y = np.linspace(-box_lim, box_lim, resolution)
                z = np.linspace(-box_lim, box_lim, resolution)
                xx, yy, zz = np.meshgrid(x, y, z)

                grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

                ppb = 500000
                n_batches = len(grid) // ppb

                sdf_preds = []

                for i in range(n_batches + 1):
                    if i < n_batches:
                        # print(250000 * i, 250000 * (i + 1))
                        xyz = torch.from_numpy(grid[ppb * i : ppb * (i + 1)]).to(DEVICE)
                    else:
                        # print(250000 * i, ": ")
                        xyz = torch.from_numpy(grid[ppb * i :]).to(DEVICE)

                    xyz *= decoder_input_scale

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
                sdf_grid_pred = {
                        "epicardium": sdf_pred[:, 0],
                        "lv_endo": sdf_pred[:, 1],
                        "rv_endo": sdf_pred[:, 2]
                    }
                
                data_ = np.load( coords_and_sdf_file )

                points_all_in_scene = data_[:,:3] # they have to be in the same scale I create grid on which I interpolate the SDF !

                sdfs_all_gt = {
                        "epicardium": data_[:, 0 + 3],
                        "lv_endo": data_[:, 1 + 3],
                        "rv_endo": data_[:, 2 + 3]
                    }

                thresholds = {}
                
                organs_to_process = ["epicardium", "lv_endo", "rv_endo"]

                mesh_gt_dict = {}
                patient_dir = PATIENT_MESHES_DIR / patient_name
                for organ_name in organs_to_process:
                    mesh_file = next(patient_dir.rglob(f"{organ_name}-processed.vtp"), None)
                    mesh_gt_dict[organ_name] = pv.read(mesh_file)


                for i,organ in enumerate(organs_to_process):

                    print(f"\n > > > Processing {organ} surface ")
                    
                    # mesh_gt = pv.read(mesh_file)
                    mesh_gt = mesh_gt_dict[organ]

                    # now I compute chamfer without needing the reconstructed mesh, so I do it only when needed
                    if save_reconstructed_mesh or show_reconstruction_images or save_reconstruction_images or compute_lddmm or compute_chamfer or compute_haussdorff:
                        print("\n Running marching cubes to reconstruct surface ...")

                        threshold = 0.0 

                        #TODO: optionally minimize for each surface with differential evolution
                        thresholds[organ] = threshold
                                            
                        try: # mesh is now in the same scale of the grid points I reconstructed it on !!
                            mesh_reconstructed = isosurface_from_sdf( x, y, z, sdf_pred=sdf_grid_pred[organ], level = threshold, box_lim = box_lim )
                        except:
                            print( f" ##### Version {version}: skipping {organ} isosurface extraction: not found for current isovalue !!!")
                            if i == len(organs_to_process)-1:
                                return
                            else:
                                continue

                        if save_reconstructed_mesh:
                            if reconstruct_from == "all":
                                fname = f"{version}-{patient_name}-{organ}.vtp"
                            if reconstruct_from == "la":
                                fname = f"{version}-{patient_name}-{organ}-from_la_only.vtp"

                            mesh_reconstructed.save(RECONSTRUCTED_MESHES_DIR / fname )

                            print("\n Saved reconstructed mesh file")

                        # patient_dir = PATIENT_MESHES_DIR / patient_name

                        # DEBUG
                        # print("patient dir:", patient_dir)

                        # print(patient_dir.rglob(f"{organ}-processed.vtp"))
                        #  fine debug


                        # mesh_file = next( patient_dir.rglob(f"{organ}-processed.vtp"), None) # !!! these are assumed to be standardized, unit scale already.                    
                        
                        
                        # DEBUG
                        # print("mesh file:", mesh_file) # il debug dice che è None
                        # fine debug


                        # mesh_gt = pv.read(mesh_file)
                        
                        ## !!! NOW BOTH MESHES ARE AT THE UNIT SCALE !!! 

                        # ======= region PLOTS ======= # 
                        if show_reconstruction_images or save_reconstruction_images:
                            # copy meshes so I don't modify originals, less of a pain to keep track of
                            mesh_gt_show = mesh_gt.copy()
                            mesh_reconstructed_show = mesh_reconstructed.copy()

                            # bring to original range for visualization
                            scale = mesh_gt.field_data["scale-tooriginalrange"]
                            mesh_gt_show.points *= scale
                            mesh_reconstructed_show.points *= scale

                            # ==== compute (signed!) distances of points on the predicted surface from the nearest spot on the original surface ==== #               
                            implicit_distance = vtkImplicitPolyDataDistance()
                            implicit_distance.SetInput(mesh_gt_show)
                            points_pred = mesh_reconstructed_show.points
                            signed_distances = np.array([implicit_distance.EvaluateFunction(p) for p in points_pred]) # relies on normal orientation, less accurate maybe than libigl, but I'm using it just for plots ...
                            mesh_reconstructed_show.point_data['error'] = signed_distances # save as point data in the predicted mesh

                            last_cam_pos = None
                            if show_reconstruction_images:
                                plotter = plot_gt_vs_reconstructed_with_error(
                                    mesh_gt_show, mesh_reconstructed_show, patient_name, signed_distances, off_screen = False
                                    )
                                plotter.show(interactive=True)
                                last_cam_pos = plotter.camera_position
                                plotter.close()

                            if save_reconstruction_images:
                                save_fname = patient_name + f"_{organ}_gt_vs_reconstructed_with_error_" + version + ".png"
                                save_fname = IMAGES_DIR / save_fname
                                plotter = plot_gt_vs_reconstructed_with_error(
                                    mesh_gt_show, mesh_reconstructed_show, patient_name, signed_distances, off_screen=True 
                                )
                                if last_cam_pos is not None: plotter.camera_position = last_cam_pos
                                plotter.screenshot(save_fname, transparent_background=True)
                                pv.close_all()
                                print("\n Saved plot screenshot")
                        
                    # ======= region METRICS ======= # 
                    if compute_chamfer or compute_haussdorff or compute_lddmm: 
                        
                        

                        if compute_chamfer or compute_haussdorff:
                            scale = mesh_gt.field_data["scale-tooriginalrange"][0]
                            scale_mm = scale * 0.001

                            # old code, lo teniamo per sicurezza
                            #print("\n Retrieving point clouds for metrics computation ... ")
                            # print("Sampling points ...")
                            # samples_orig = make_trimesh_from_pv(mesh_gt).sample(count=100000)
                            # samples_rec = make_trimesh_from_pv(mesh_reconstructed).sample(count=100000)

                            # Compute chamfer in another way: interpolate gt sdf values onto the same grid as prediction,
                            # threshold SDF values and compute chamfer on the resulting point clouds.
                            # OSS: If I used clamping in training, the SDF will be worse outside the clamp value !!
                            # so when deciding the shell width in mm, if it correspond at the unit scale to a shell wider 
                            # than the clamp value, the predicted SDFs are not faithful to the real SDF, so when thresholding points in
                            # that region, I might get pointclouds with many more or many less points than the real gt shell has, 
                            # this may bias the chamfer distance. 
                            # 
                            # (This happens when shell_threshold = shell_thick_mm / scale_mm becomes larger than clamp value used in training)
                            # 
                            # What I am computing here is basically how close the shells are to one
                            # another, but larger point clouds may lower the chamfer, when in reality the predicted shell is not "better" per say
                            # so choose shell_thick_mm low enough. 
                            # --> OR, train without clamp to regress the SDF correctly in more space around the surface

                            # here grid and SDF values are at the unit-sphere scale.
                            # Interpolate GT SDF onto same grid points --> use the all the original available points
                            # sdf_grid_gt_organ = griddata(
                            #     points_all_in_scene,
                            #     sdfs_all_gt[organ],
                            #     grid,
                            #     method='linear'   
                            # )

                            # # select points: SDF in a shell around the surface: make it meaningful in millimeters !
                            # # but still I want to compute chamfer at standardized scale for stability
                            # # Chamfer decreases as shell widens, not because reconstruction is better, 
                            # # but because more points farther from high-error regions dilute the average.

                            # patient_dir = PATIENT_MESHES_DIR / patient_name
                            # mesh_file = next( patient_dir.rglob(f"{organ}-processed.vtp"), None) # !!! these are assumed to be standardized, unit scale already.
                            # mesh_gt = pv.read(mesh_file)
                            # scale = mesh_gt.field_data["scale-tooriginalrange"] # this rescales the standardized mesh to its original scale, in micrometers
                            # scale_mm = scale * 0.001

                            # shell_thick_mm = 0.5 # this means the points AT MILLIMETERS SCALE will be thresholded where their distance to the surface is less than shell_thick_mm mm 
                            # shell_threshold = shell_thick_mm / scale_mm # pick this to be used at the unit scale, but so that the samples rescaled at mm scale are in fact in a shell of thickness 2*shell_thick_mm around the surface

                            # # now grid is still at the standardized scale, where I compute chamfer
                            # samples_gt = grid[ np.where( np.abs(sdf_grid_gt_organ) <= shell_threshold) ]

                            # samples_pred = grid[ np.where( np.abs(sdf_grid_pred[organ]) <= shell_threshold) ]

                            # # resample to same number of points : uniform
                            # num_points = 20000
                            # if samples_gt.shape[0] > num_points:
                            #     samples_gt = samples_gt[np.random.choice(samples_gt.shape[0], num_points, replace=False)]
                            # if samples_pred.shape[0] > num_points:
                            #     samples_pred = samples_pred[np.random.choice(samples_pred.shape[0], num_points, replace=False)]

                            # if compute_chamfer: # average nearest-neighbor distances in millimeters
                            #     print("\n Computing chamfer")
                            #     chamfer_dists[patient_name][organ] = chamfer_distance_L2(samples_gt, samples_pred) * scale_mm # compute at unit scale, rescale to mm
                        
                            # if compute_haussdorff:
                            #     print("\n Computing Haussdorff")
                            #     haussdorff_dists[patient_name][organ] = haussdorff(samples_gt, samples_pred) * scale_mm 
                            
                            if use_old_chamfer_surface_metric:
                                print("\n Retrieving point clouds for OLD chamfer metrics computation ...")

                                # =================
                                # OLD METHOD
                                # compare shell point clouds by thresholding sdf on the grid
                                # =================

                                sdf_grid_gt_organ = griddata(
                                    points_all_in_scene,
                                    sdfs_all_gt[organ],
                                    grid,
                                    method='linear'   
                                )

                                shell_thick_mm = 0.5
                                shell_threshold = shell_thick_mm / scale_mm

                                samples_gt = grid[np.where(np.abs(sdf_grid_gt_organ) <= shell_threshold)]
                                samples_pred = grid[np.where(np.abs(sdf_grid_pred[organ]) <= shell_threshold)]

                                num_points = 20000

                                if samples_pred.shape[0] > num_points:
                                    samples_pred = samples_pred[np.random.choice(samples_pred.shape[0], num_points, replace=False)]

                                if compute_chamfer: # average nearest-neighbor distances in millimeters
                                    print("\n Computing chamfer")
                                    chamfer_dists[patient_name][organ] = chamfer_distance_L2(samples_gt, samples_pred) * scale_mm # compute at unit scale, rescale to mm
                            
                                if compute_haussdorff:
                                    print("\n Computing Haussdorff")
                                    haussdorff_dists[patient_name][organ] = haussdorff(samples_gt, samples_pred) * scale_mm 
                            
                            else:
                                print("\n point clouds for new chamfer metrics computation")   
                                # =========================
                                # NEW METHOD
                                # Sample points directly from GT and reconstructed meshes
                                # =========================

                                if mesh_reconstructed is None:
                                    try:
                                        mesh_reconstructed = isosurface_from_sdf(
                                                                x, y, z,
                                                                sdf_pred=sdf_grid_pred[organ],
                                                                level=0.0,
                                                                box_lim=box_lim
                                        )
                                    except:
                                        print(f"!!! Version {version}: skipping {organ} isosurface extraction: not found for current isovalue !!! ")
                                        continue
                                    
                                samples_count = 50000
                                samples_gt = make_trimesh_from_pv(mesh_gt).sample(count=samples_count)
                                samples_pred = make_trimesh_from_pv(mesh_reconstructed).sample(count=samples_count)
                            
                                if compute_chamfer:
                                    print("\n Computing chamfer")
                                    chamfer_dist = chamfer_distance_L2(samples_gt, samples_pred) * scale_mm
                                    chamfer_dists[patient_name][organ] = chamfer_dist
                                
                                if compute_haussdorff:
                                    print("\n Computing Hausdorff")
                                    haussdorff_dist = haussdorff(samples_gt, samples_pred) * scale_mm
                                    haussdorff_dists[patient_name][organ] = haussdorff_dist



                        if compute_lddmm:
                            print("\n Computing LDDMM")

                            # check if already present in mesh directory and load, so I save a little bit of time instead of remeshing always the original ones
                            mesh_gt_remeshed_file = next( patient_dir.rglob(f"{organ}-processed-remeshed.vtp"), None)
                            if mesh_gt_remeshed_file is not None:
                                mesh_gt = pv.read(mesh_gt_remeshed_file)
                            else:
                                mesh_gt = remesh(mesh_gt, n_points=50000) 

                            mesh_reconstructed = remesh(mesh_reconstructed, n_points=50000)

                            # # pick gamma based on mesh size, so loss doesn't over or underflow IF I compute it on meshes at huge scales like micrometers
                            # typical_distance = utils.metrics.estimate_typical_distance(mesh1) # both meshes are remeshed to same resolution, so doesn't matter which I use
                            # gamma = 1.0 / typical_distance**2

                            # compute LDDMM at STANDARDIZED scale for stability, otherwise gamma should be probably picked differently
                            LDDMM_losses[patient_name][organ] = LDDMM_loss(mesh_gt, mesh_reconstructed, remeshing=False, gamma = 1.0, device = DEVICE)
    

    if compute_chamfer:
        exp_name = experiment_name.split("/")[-1]
        save_metrics_csv("chamfer", exp_name, version, which_shapes, chamfer_dists)
        print("Saved chamfer distances.")

    if compute_haussdorff:
        exp_name = experiment_name.split("/")[-1]
        save_metrics_csv("haussdorff", exp_name, version, which_shapes, haussdorff_dists)
        print("Saved haussdorff distances.")

    if compute_lddmm:
        exp_name = experiment_name.split("/")[-1]
        save_metrics_csv("LDDMM", exp_name, version, which_shapes, LDDMM_losses)
        print("Saved LDDMM distances.")

    if save_latent_codes:
        exp_name = experiment_name.split("/")[-1]
        loss_type = "L2" if not use_mahalanobis_loss else "Maha"
        save_latents_npz(
            exp_name, version, which_shapes,
            code_reg_lambda, num_epochs, initialize_latent_from,
            loss_type,
            latent_codes
        )
        print("Saved fitted latents.")

    print("Done.")

    return


       


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--experiment_name", "-e", type=str, default = "training_sweeps/LipAndAct")
    parser.add_argument("--experiment_name", "-e", type=str, default = "enrico_preliminary_train")
    parser.add_argument("--version", "-v", type=str, default = "version_0")
    parser.add_argument("--override_with_dataset", "-od", type=str, default=None)
    parser.add_argument("--override_with_patients_list", "-opl", type=str, nargs="+", default=None, help="List of patient IDs to process")
    parser.add_argument("--mode", "-m", type=int, default=1, choices=[1, 2])
    parser.add_argument("--reconstruct_from", "-r", type=str, default="all", choices=["la","ra","all"])
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
    args = parser.parse_args()
    

    json_file = None
    if args.override_with_patients_list:
        # Build temporary json with the wanted patients, delete it at the end
        # for now, I build by hand the file names I KNOW are in PATIENTS_NUMPY_DATA_DIR ... just to test this
        names = [f"{p}-epi_la_ra_100000_coords_and_sdf.npy" for p in args.override_with_patients_list]
        test_datafnames = "patients_list_temp.json"
        json_file = Path(f"data/{test_datafnames}")
        with json_file.open("w") as f:
            json.dump(names, f)
    else:
        test_datafnames = args.override_with_dataset or "test/data_fnames_test.json"

    exp_name = args.experiment_name
    vers = args.version
    mode = args.mode 

    kwargs = {
        "experiment_name" : exp_name,
        "version" : vers,
        "override_with_dataset" : test_datafnames,
        "num_samp_per_scene_for_fit" : args.num_samp_per_scene_for_fit,
        "num_epochs_fit_latent" : args.num_epochs,
        "latent_reg_factor" : args.latent_reg_factor,
        "lr_fit_latent" : args.lr,
        "initialize_latent_from" : args.init_latent_from,
        "use_mahalanobis_loss" : args.use_mahalanobis_loss
    }

    match mode:

        case 1: # reconstruct surface, then visualize or compute metrics
            run_kwargs = {
                **kwargs,
                "reconstruct_surface" : True,
                "reconstruct_from" : args.reconstruct_from,
                "show_reconstruction_images" : args.interactive_images,
                "save_reconstruction_images" : args.save_images,
                "save_reconstructed_mesh" : args.save_reconstructed_meshes,
                "compute_chamfer" : args.compute_chamfer,
                "compute_lddmm" : args.compute_lddmm,
                "compute_haussdorff" : args.compute_haussdorff,
                "save_latent_codes" : args.save_latent_codes
            }

        case 2: # only fit latents and save
            run_kwargs = {
                **kwargs,
                "reconstruct_surface" : False,
                "reconstruct_from" : args.reconstruct_from,
                "show_reconstruction_images" : False,
                "save_reconstruction_images" : False,
                "save_reconstructed_mesh" : False,
                "compute_chamfer" : False,
                "compute_lddmm" : False,
                "compute_haussdorff" : False,
                "save_latent_codes" : True
            }

    run(**run_kwargs)

    if json_file and json_file.exists():
        json_file.unlink()
