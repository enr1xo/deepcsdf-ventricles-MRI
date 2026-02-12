import json
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import re
import gc
from collections import defaultdict
import pyvista as pv
from model.deepsdf_decoder import Decoder
from utils.reconstruction_utils import isosurface_from_sdf
# from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
import pandas as pd

from config import RESULTS_DIR, IMAGES_DIR, EXPERIMENTS_DIR, ANIMATIONS_DIR

def animate_meshes_sequence(timed_meshes, animation_fname):

    ts = timed_meshes["times"]
    meshes = timed_meshes["meshes"]
    
    print(f"Received {len(meshes)} meshes")

    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(animation_fname, framerate=10)

    actor = plotter.add_mesh(meshes[0], color="white")
    text_actor = plotter.add_text(f"", font_size=20, position='upper_edge')

    #plotter.show(auto_close=False)
    camera_pos = [
        (300.0, 300.0, 300.0),   # eye position
        (0.0, 0.0, 0.0),         # focal point
        (0.0, 0.0, 1.0)          # up direction
    ]

    print("Animating mesh evolution ...")

    for i, mesh in enumerate(meshes):
        actor.mapper.SetInputData(mesh)
        plotter.camera_position = camera_pos
        # update existing text actor
        text_actor.SetText(0, f"t={ts[i]:.2f}") # <- this updates the text line
        plotter.render()
        plotter.write_frame()

    plotter.close()

    # then turn into .gif from .mp4 with ffmpeg:
    # $ ffmpeg -y -i input.mp4 -vf "fps=10,scale=720:-1:flags=lanczos,palettegen" palette.png
    # $ ffmpeg -i input.mp4 -i palette.png -filter_complex "fps=10,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none" output.gif

    return

def interpolate_latents(
        z1, z2, decoder: Decoder, num_interp = 10,
        extract_surface = ["epicardium"]
    ):

    ts  = np.linspace(0, 1, num_interp)

    # sdfs_preds = defaultdict(list)

    meshes = []

    times = []

    resolution = 128

    #region SDF ON GRID FOR RECONSTRUCTION
    boxlim = 100
    x = np.linspace(-boxlim, boxlim, resolution)
    y = np.linspace(-boxlim, boxlim, resolution)
    z = np.linspace(-boxlim, boxlim, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)

    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    xyz_raw = grid
    num_grid_points = xyz_raw.shape[0]

    sdf_pred = np.empty((xyz_raw.shape[0], 3), dtype=np.float32)

    n_batches = len(xyz_raw) // 250000
    
    decoder.to(device="cuda")

    for t in ts:

        # Allocate fresh big SDF buffer per t
        sdf_pred = np.empty((num_grid_points, 3), dtype=np.float32)

        latent_interp = torch.from_numpy((1 - t) * z1 + t * z2).float()

        logger.info(f"Computing SDF on grid for reconstruction, evolution progress: {(t/(ts[-1]-ts[0]))*100:.1f} %")

        decoder.eval()

        with torch.no_grad():

            for i in range(n_batches+1):
                start = i * 250000
                end = min(num_grid_points, (i + 1) * 250000)

                xyz = torch.from_numpy(xyz_raw[start:end]).float()
                batch_vecs = latent_interp.expand(end - start, -1)
                input_ = torch.cat([batch_vecs, xyz], dim=1).to("cuda")

                sdf_batch = decoder(input_).detach().cpu().numpy()

                sdf_pred[start:end] = sdf_batch  
                del xyz, batch_vecs, input_, sdf_batch
                torch.cuda.empty_cache()
        
        if "epicardium" in extract_surface:
            sdf_use = sdf_pred[:, 0]
        elif "la_endo" in extract_surface:
            sdf_use = sdf_pred[:, 1]
        elif "ra_endo" in extract_surface:
            sdf_use = sdf_pred[:, 2]
        else:
            raise ValueError("Unknown surface")

        try:
            mesh_organ = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)
            meshes.append(mesh_organ)
            times.append(t)
        except:
            break

        del latent_interp, sdf_pred, mesh_organ
        gc.collect()
        torch.cuda.empty_cache()
        
    return {"times" : times, "meshes" : meshes }

def extrapolate_latents(
        z1, z2, decoder: Decoder, t_max, num_interp = 10,
        extract_surface = ["epicardium"],
    ):

    if t_max < 0:
        ts  = np.linspace(0, t_max, num_interp)
    else:
        ts  = np.linspace(1, t_max, num_interp)

    meshes = []

    times = []

    resolution = 128

    #region SDF ON GRID FOR RECONSTRUCTION
    boxlim = 100
    x = np.linspace(-boxlim, boxlim, resolution)
    y = np.linspace(-boxlim, boxlim, resolution)
    z = np.linspace(-boxlim, boxlim, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)

    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    xyz_raw = grid
    num_grid_points = xyz_raw.shape[0]

    sdf_pred = np.empty((xyz_raw.shape[0], 3), dtype=np.float32)

    n_batches = len(xyz_raw) // 250000
    
    decoder.to(device="cuda")

    for t in ts:

        # Allocate fresh big SDF buffer per t
        sdf_pred = np.empty((num_grid_points, 3), dtype=np.float32)

        latent_interp = torch.from_numpy( z1 +  t * (z2 - z1) ).float()

        logger.info(f"Computing SDF on grid for reconstruction, evolution progress: {np.abs((t-ts[0])/(ts[-1]-ts[0])) * 100:.1f} %")

        decoder.eval()

        with torch.no_grad():

            for i in range(n_batches+1):
                start = i * 250000
                end = min(num_grid_points, (i + 1) * 250000)

                xyz = torch.from_numpy(xyz_raw[start:end]).float()
                batch_vecs = latent_interp.expand(end - start, -1)
                input_ = torch.cat([batch_vecs, xyz], dim=1).to("cuda")

                sdf_batch = decoder(input_).detach().cpu().numpy()

                sdf_pred[start:end] = sdf_batch  
                del xyz, batch_vecs, input_, sdf_batch
                torch.cuda.empty_cache()
        
        if "epicardium" in extract_surface:
            sdf_use = sdf_pred[:, 0]
        elif "la_endo" in extract_surface:
            sdf_use = sdf_pred[:, 1]
        elif "ra_endo" in extract_surface:
            sdf_use = sdf_pred[:, 2]
        else:
            raise ValueError("Unknown surface")

        try:
            mesh_organ = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)
            meshes.append(mesh_organ)
            times.append(t)
        except:
            break

        del latent_interp, sdf_pred, mesh_organ
        gc.collect()
        torch.cuda.empty_cache()
        
    return {"times" : times, "meshes" : meshes}

def _decode_mesh(decoder : Decoder, latent, extract_surface = ["epicardium"]):

    resolution = 128

    boxlim = 100
    x = np.linspace(-boxlim, boxlim, resolution)
    y = np.linspace(-boxlim, boxlim, resolution)
    z = np.linspace(-boxlim, boxlim, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)

    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    xyz_raw = grid
    num_grid_points = xyz_raw.shape[0]

    n_batches = len(xyz_raw) // 250000
    
    latent = torch.from_numpy(latent).to(device="cuda")

    decoder.to(device="cuda")

    decoder.eval()

    sdf_pred = np.empty((xyz_raw.shape[0], 3), dtype=np.float32)

    with torch.no_grad():

        for i in range(n_batches+1):
            start = i * 250000
            end = min(num_grid_points, (i + 1) * 250000)

            xyz = torch.from_numpy(xyz_raw[start:end]).float().to(device="cuda")
            batch_vecs = latent.expand(end - start, -1)
            input_ = torch.cat([batch_vecs, xyz], dim=1).to(device="cuda")

            sdf_batch = decoder(input_).detach().cpu().numpy()

            sdf_pred[start:end] = sdf_batch  
            del xyz, batch_vecs, input_, sdf_batch
            torch.cuda.empty_cache()
    
    if "epicardium" in extract_surface:
        sdf_use = sdf_pred[:, 0]
    elif "la_endo" in extract_surface:
        sdf_use = sdf_pred[:, 1]
    elif "ra_endo" in extract_surface:
        sdf_use = sdf_pred[:, 2]
    else:
        raise ValueError("Unknown surface")

    mesh_organ = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)

    return mesh_organ    







if __name__ == "__main__":

    pass

    # # region interpolate two latents

    # version = "version_114"
    # experiment_name = "deepsdf_atria_training_concurrent"
    # codereg = 0.0002
    # latents_fname = f"results/fitted_latents/latent_codes_50_patients_{version}-codereg={codereg:.6f}-epochs=300.npz"

    # latents = np.load(latents_fname)

    # patient1 = "AF009_P2R"
    # patient2 = "LEU_NORM_F004"
    # z1 = latents[patient1]
    # z2 = latents[patient2]

    # # load decoder
    # version_dir = EXPERIMENTS_DIR / experiment_name / version
    # hparams_file = version_dir / "hparams.json"
    # specs = json.load( open(hparams_file) )

    # decoder_params = specs["Network_specs"]

    # decoder = Decoder(**decoder_params)
    # # get trained model parameters
    # decoder_weights_path = version_dir / "decoder_weights.pth"
    # state_dict = torch.load(decoder_weights_path)
    # decoder.load_state_dict(state_dict)

    # print("\n")
    # print(decoder.description())
    # print("\n")

    # # meshes = extrapolate_latents(z1, z3, decoder, t_max=-2, num_interp=30)

    # # animate_meshes_sequence(meshes, ANIMATIONS_DIR / f"anim-latents-{version}-codereg={codereg}-extrapolate-before.mp4")

    # timed_meshes = interpolate_latents(z1, z2, decoder, num_interp=30)

    # animate_meshes_sequence(timed_meshes, ANIMATIONS_DIR / f"anim-latents-{version}-{patient1}-to-{patient2}-interpolate.mp4")

    # # meshes = extrapolate_latents(z1, z3, decoder, t_max=2, num_interp=30)

    # # animate_meshes_sequence(meshes, ANIMATIONS_DIR / f"anim-latents-{version}-codereg={codereg}-extrapolate-beyond.mp4")
    
    # meshes = timed_meshes["meshes"]

    # t = 0
    # mesh_organ = meshes[0]
    # mesh_organ.save( RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient1}-to-{patient2}-time={t}.vtp" )

    # t = 1/6
    # mesh_organ = meshes[4]
    # mesh_organ.save( RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient1}-to-{patient2}-time={t}.vtp" )

    # t = 2/6
    # mesh_organ = meshes[9]
    # mesh_organ.save( RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient1}-to-{patient2}-time={t}.vtp" )

    # t = 3/6
    # mesh_organ = meshes[14]
    # mesh_organ.save( RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient1}-to-{patient2}-time={t}.vtp" )

    # t = 4/6
    # mesh_organ = meshes[19]
    # mesh_organ.save( RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient1}-to-{patient2}-time={t}.vtp" )

    # t = 5/6
    # mesh_organ = meshes[24]
    # mesh_organ.save( RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient1}-to-{patient2}-time={t}.vtp" )

    # t = 1
    # mesh_organ = meshes[-1]
    # mesh_organ.save( RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient1}-to-{patient2}-time={t}.vtp" )

    # # # # from sklearn.decomposition import PCA

    # # # # latent_codes = [l for l in latents.values()]
    # # # # latent_codes = np.array(latent_codes)

    # # # # pca = PCA(n_components=2)
    # # # # pca.fit(latent_codes)

    # # # # pc1 = pca.components_[0]                # unit vector
    # # # # scale = np.sqrt(pca.explained_variance_[0])
    # # # # latent_pc1 = scale * pc1               # 64-D, correct magnitude

    # # # # pc2 = pca.components_[1]                # unit vector
    # # # # scale = np.sqrt(pca.explained_variance_[1])
    # # # # latent_pc2 = scale * pc2               # 64-D, correct magnitude

    # # # # mesh_pc1 = _decode_mesh(decoder, latent_pc1)

    # # # # mesh_pc2 = _decode_mesh(decoder, latent_pc2)

    # # # # plotter = pv.Plotter()
    # # # # plotter.add_mesh(mesh_pc1)
    # # # # plotter.show()

    # # # # plotter = pv.Plotter()
    # # # # plotter.add_mesh(mesh_pc2)
    # # # # plotter.show()
    



    # # region triangle interpolation

    # # from config import RESULTS_DIR

    # # version = "version_13"
    # # experiment_name = "deepsdf_atria_training_concurrent"

    # # latents_fname = f"/home/davidenava_linux/AtriaProject/deepcsdf_fork/deepcsdf/deepcsdf/deepcsdfatria/results/fitted_latents/latent_vectors_3_patients_{version}.npz"

    # # latents = np.load(latents_fname)

    # # patient1 = "AF001"
    # # patient2 = "AF069"
    # # patient3 = "LEU_NORM_0032"
    # # z1 = latents[patient1]
    # # z2 = latents[patient2]
    # # z3 = latents[patient3]

    # # # load decoder
    # # version_dir = EXPERIMENTS_DIR / experiment_name / version

    # # hparams_file = version_dir / "hparams.json"

    # # specs = json.load( open(hparams_file) )

    # # decoder_params = specs["Network_specs"]

    # # decoder = Decoder(**decoder_params)
    # # # get trained model parameters
    # # decoder_weights_path = version_dir / "decoder_weights.pth"
    # # state_dict = torch.load(decoder_weights_path)
    # # decoder.load_state_dict(state_dict)

    # # print("\n")
    # # print(decoder.description())
    # # print("\n")

    # # ts  = np.linspace(0, 1, 5)
     
    # # z_ij_t = lambda z_i,z_j,t: (1-t)*z_i + t*z_j
    
    # # z_12_t = []
    # # z_13_t = []
    # # z_23_t = []

    # # for t in ts[1:-1]:
    # #     z_12_t.append( z_ij_t(z1,z2,t) )
    # #     z_13_t.append( z_ij_t(z1,z3,t) )
    # #     z_23_t.append( z_ij_t(z2,z3,t) )

    # # t1 = 1/3
    # # t2 = 2/3

    # # z_inside = []
    # # z_inside.append( z_ij_t(z_12_t[0], z_23_t[-1], t1) )
    # # z_inside.append( z_ij_t(z_12_t[0], z_23_t[-1], t2) )
    # # z_inside.append( z_ij_t(z_12_t[-1], z_13_t[-1], t1) )

    # # resolution = 128

    # # #region SDF ON GRID FOR RECONSTRUCTION
    # # boxlim = 100
    # # x = np.linspace(-boxlim, boxlim, resolution)
    # # y = np.linspace(-boxlim, boxlim, resolution)
    # # z = np.linspace(-boxlim, boxlim, resolution)
    # # xx, yy, zz = np.meshgrid(x, y, z)

    # # grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    # # xyz_raw = grid
    # # num_grid_points = xyz_raw.shape[0]

    # # sdf_pred = np.empty((xyz_raw.shape[0], 3), dtype=np.float32)

    # # n_batches = len(xyz_raw) // 250000
    
    # # decoder.to(device="cuda")

    # # name = [f"{patient1}_to_{patient3}_epi_inside_t={int(t1*100)}",
    # #         f"{patient1}_to_{patient3}_epi_inside_t={int(t2*100)}",
    # #         f"{patient2}_to_{patient3}_epi_inside_t={int(t1*100)}"
    # # ]

    # # for num,code in enumerate(z_inside):

    # #     code = torch.from_numpy(code).float()

    # #     # Allocate fresh big SDF buffer per t
    # #     sdf_pred = np.empty((num_grid_points, 3), dtype=np.float32)

    # #     logger.info(f"Computing SDF on grid for reconstruction")

    # #     decoder.eval()

    # #     with torch.no_grad():

    # #         for i in range(n_batches+1):
    # #             start = i * 250000
    # #             end = min(num_grid_points, (i + 1) * 250000)

    # #             xyz = torch.from_numpy(xyz_raw[start:end]).float()
    # #             batch_vecs = code.expand(end - start, -1)
    # #             input_ = torch.cat([batch_vecs, xyz], dim=1).to("cuda")

    # #             sdf_batch = decoder(input_).detach().cpu().numpy()

    # #             sdf_pred[start:end] = sdf_batch  
    # #             del xyz, batch_vecs, input_, sdf_batch
    # #             torch.cuda.empty_cache()
        
    # #     sdf_use = sdf_pred[:, 0]

    # #     mesh_organ = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)

    # #     mesh_organ = pv.wrap(mesh_organ)
    # #     mesh_organ.save( RESULTS_DIR / f"reconstructed/interpolated/{version}_{name[num]}.vtp" )

    # # meshes = interpolate_latents(z1, z2, extract_surface = "epicardium", decoder=decoder, num_interp=5)

    # # for m, t in zip(meshes, ts):
    # #     m = pv.wrap(m)
    # #     m.save(RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient1}_to_{patient2}_epi_t={int(t*100)}.vtp")

    # # del meshes

    # # meshes = interpolate_latents(z1, z3, extract_surface = "epicardium", decoder=decoder, num_interp=5)

    # # for m, t in zip(meshes, ts):
    # #     m = pv.wrap(m)
    # #     m.save(RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient1}_to_{patient3}_epi_t={int(t*100)}.vtp")

    # # del meshes

    # # meshes = interpolate_latents(z2, z3, extract_surface = "epicardium", decoder=decoder, num_interp=5)

    # # for m, t in zip(meshes, ts):
    # #     m = pv.wrap(m)
    # #     m.save(RESULTS_DIR / f"reconstructed/interpolated/{version}_{patient2}_to_{patient3}_epi_t={int(t*100)}.vtp")


    # # meshes_files = list( Path("results/reconstructed/interpolated").iterdir() )

    # # for mfile in meshes_files:
    # #     m = pv.read(mfile)
    # #     plotter = pv.Plotter(window_size=[2000,2000])
    # #     plotter.add_mesh(m, color = 'lightgray')
    # #     #plotter.add_title(f"{mfile.name}")
    # #     plotter.camera_position = [
    # #         (-81.04493993231137, 260.8156792438555, 432.9223371958522),
    # #         (3.010909313291826, -13.338784045121962, -6.983822800043235),
    # #         (-0.7534632562880871, 0.48365499582887966, -0.4453885566935002)
    # #     ]
    # #     plotter.show(interactive=False)   
    # #     name = mfile.name
    # #     name = name.replace(".vtp","")
    # #     plotter.screenshot(f"{name}.png")
    # #     plotter.close()








