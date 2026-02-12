import json
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import gc
import pyvista as pv
from model.deepsdf_decoder import Decoder
from utils.reconstruction_utils import isosurface_from_sdf
import matplotlib.pyplot as plt

from config import RESULTS_DIR, IMAGES_DIR, EXPERIMENTS_DIR

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
        ts  = np.linspace(t_max, 0, num_interp)
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

def _decode_mesh(decoder : Decoder, latent, extract_surface = None):

    resolution = 128

    boxlim = 101
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
    
    if extract_surface == "epicardium":
        sdf_use = sdf_pred[:, 0]
        mesh_organ = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)
        return mesh_organ
    
    elif extract_surface =="la_endo":
        sdf_use = sdf_pred[:, 1]
        mesh_organ = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)
        return mesh_organ
    
    elif extract_surface == "ra_endo":
        sdf_use = sdf_pred[:, 2]
        mesh_organ = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)
        return mesh_organ
    
    elif extract_surface == "all":
        mesh_organs = {}
        sdf_use = sdf_pred[:, 0]
        mesh_organs["epicardium"] = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)
        sdf_use = sdf_pred[:, 1]
        mesh_organs["la_endo"] = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)
        sdf_use = sdf_pred[:, 2]
        mesh_organs["ra_endo"] = isosurface_from_sdf(x, y, z, sdf_pred=sdf_use, level=0.0)    
        return mesh_organ
    
    else:
        raise ValueError("Unknown surface, available: 'epicardium' 'la_endo' 'ra_endo' 'all'")

def plot_latents_components_magnitude(latents):

    for i in range(latents.shape[0]):
        plt.plot(np.arange(latents.shape[-1]), latents[i], c = 'k', linewidth = 0.1)
    plt.show()







if __name__ == "__main__":

    latents = np.load("experiments/training_sweeps/RegLambda/version_0/latents.npy")

    plot_latents_components_magnitude(latents)

    pass









