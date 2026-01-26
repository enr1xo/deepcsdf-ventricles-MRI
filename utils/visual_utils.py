import pyvista as pv
import vtk
import numpy as np
import os
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance



COLORS_PALETTE = {
    # --- Pastel tones ---
    "pastel_blue": "#AECBFA",
    "pastel_green": "#B7E1A1",
    "pastel_pink": "#F4C2C2",
    "pastel_orange": "#FFD1A9",
    "pastel_yellow": "#FFF4A3",
    "pastel_purple": "#CBA3F4",
    "pastel_teal": "#A3E4D7",
    "pastel_red": "#F7A1A1",
    "pastel_gray": "#D8D8D8",
    "pastel_brown": "#E3C7A1",

    # --- Fluorescent / Neon tones ---
    "neon_green": "#39FF14",      # Matrix green
    "neon_blue": "#04D9FF",       # Cyan-blue glow
    "neon_pink": "#FF10F0",       # Magenta pink
    "neon_orange": "#FF6700",     # Bright orange
    "neon_purple": "#BF00FF",     # Electric purple
    "neon_yellow": "#F5FF00",     # Highlighter yellow
    "neon_red": "#FF073A",        # Vibrant red
    "neon_turquoise": "#00FFEF",  # Fluorescent aqua
    "neon_lime": "#CFFF04",       # Lime acid green
    "neon_magenta": "#FF00C8",    # Deep magenta
    "neon_orange2" : "#FF6464",
    "neon_cyan" : '#00FFFF',
    "electric_blue" : '#007BFF',

    # --- Extra vivid but not eye-burning ---
    "sky_blue": "#4FC3F7",
    "mint_green": "#98FF98",
    "coral": "#FF7F50",
    "violet": "#EE82EE",
    "sun_yellow": "#FFD300",
    "aqua": "#00FFFF",
    "hot_pink": "#FF69B4",
    "light_lavender": "#D8B7FF",
    "apple_green": "#8DB600",
    "deep_cerulean": "#007BA7",
}


# ================================================================ #
# region meshes visualization
# ================================================================ #
def world_length_to_pixels(plotter, length):
    ren = plotter.renderer
    fx, fy, fz = plotter.camera_position[1]

    ren.SetWorldPoint(fx, fy, fz, 1.0)
    ren.WorldToView()
    v0 = ren.GetViewPoint()

    ren.SetWorldPoint(fx + length, fy, fz, 1.0)
    ren.WorldToView()
    v1 = ren.GetViewPoint()

    ren.SetViewPoint(*v0)
    ren.ViewToDisplay()
    x0, y0, _ = ren.GetDisplayPoint()

    ren.SetViewPoint(*v1)
    ren.ViewToDisplay()
    x1, y1, _ = ren.GetDisplayPoint()

    return abs(x1 - x0)
    
def add_static_scale_bar(plotter, length, label, pos=(60, 60)):
    px = world_length_to_pixels(plotter, length)


    # Line in display coordinates
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(2)
    points.SetPoint(0, pos[0], pos[1], 0)
    points.SetPoint(1, pos[0] + px, pos[1], 0)

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(2)
    lines.InsertCellPoint(0)
    lines.InsertCellPoint(1)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 0)
    actor.GetProperty().SetLineWidth(4)

    plotter.renderer.AddActor2D(actor)

    plotter.add_text(
        label,
        position=(pos[0], pos[1] + 15),
        font_size=12,
        color="black",
    )

def plot_surface_with_color(
        surface_mesh,
        surf_mesh_file = None, 
        color = 'white',
        opacity = 1.0,
        show_edges = False,
        color_by_curv = False,
        cmap = 'magma',
        save_ply = False,
        save_name = None,
        save_dir = None):

    if surf_mesh_file is not None:
        surface = pv.read(surf_mesh_file)
    else:
        surface = surface_mesh

    if color_by_curv:

        if not isinstance(surface, pv.PolyData):
            surface = surface.extract_surface()

        curv = surface.curvature(curv_type='mean')
        curv_abs = np.abs(curv)
        epsilon = 1e-6
        curv_log = np.log10(curv_abs + epsilon)

        curv_rescaled = (curv_log - curv_log.min()) / (curv_log.max() - curv_log.min())
        surface['CurvatureLog'] = curv_rescaled

        plotter = pv.Plotter()
        plotter.add_mesh(
            surface,
            scalars='CurvatureLog',
            cmap=cmap,
            show_edges=show_edges,
            show_scalar_bar=False
        )
        plotter.show()
    else:
        plotter = pv.Plotter()
        plotter.add_mesh(
            surface,
            color = color,
            show_edges=show_edges,
            opacity = opacity
        )
        plotter.show()

    if save_ply:
        surface.save(os.path.join(save_dir, save_name + ".ply"))
    
    return

def plot_surface_with_normals(mesh: pv.PolyData):
    
    mesh.compute_normals(
        cell_normals=False,       # we want point normals for glyphs
        point_normals=True,
        auto_orient_normals=True,
        split_vertices=False,
        inplace=True
    )

    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    diag = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    glyphs = mesh.glyph(orient="Normals", scale=False, factor=0.025*diag)

    p = pv.Plotter()
    p.add_mesh(mesh, color="lightgrey", show_edges=True, opacity=0.7)
    p.add_mesh(glyphs, color="red")  # draw the normals in red
    p.show()
    
    return

def plot_gt_vs_reconstructed(mesh_gt, mesh_pred, patient_name, opacity = 0.8, link_views = True):

    cam = dict(
        position=(300, 300, 300),
        focal_point=(0.0, 0, 0.0),
        viewup=(0.0, 0.0, 0.1),
    )

    # Create a PyVista plotter with 2 subplots (side-by-side) ----- toggle off_screen
    plotter = pv.Plotter(shape=(1, 2), window_size=[1280, 720])

    plotter.subplot(0, 0)
    plotter.add_text(f"Original mesh: patient {patient_name}", font_size=12)
    plotter.add_mesh(mesh_gt, color="pink", opacity=opacity)
    plotter.camera_position = [cam["position"], cam["focal_point"], cam["viewup"]]

    plotter.subplot(0, 1)
    plotter.add_text(f"Reconstructed mesh: patient {patient_name}", font_size=12)
    plotter.add_mesh(mesh_pred, color="pink", opacity=opacity)
    plotter.camera_position = [cam["position"], cam["focal_point"], cam["viewup"]]

    if link_views:
        plotter.link_views()

    return plotter

def plot_gt_vs_reconstructed_with_error(
        mesh_gt: pv.PolyData | pv.UnstructuredGrid,
        mesh_pred: pv.PolyData | pv.UnstructuredGrid,
        patient_name,
        signed_distances_pred_from_gt = None,
        off_screen = False,
        scale = 100
    ):
    """
        .
    """

    if signed_distances_pred_from_gt is None:
        # compute (signed!) distances of ground truth points (on the true surface) from the nearest spot on the predicted mesh surface
        implicit_distance = vtkImplicitPolyDataDistance()
        implicit_distance.SetInput(mesh_gt)
        points_pred = mesh_pred.points
        signed_distances_pred_from_gt = np.array([implicit_distance.EvaluateFunction(p) for p in points_pred])
        # errors = np.abs(signed_distances)

    # this modifies it in place actually !!! the original then has this field, if not wanted, should use deep copy of the mesh
    mesh_pred.point_data['error'] = signed_distances_pred_from_gt 

    plotter = pv.Plotter(shape=(1, 3), window_size=[1920, 720], off_screen=off_screen)

    plotter.subplot(0, 0)
    plotter.add_text(f"Original mesh: patient {patient_name}", font_size=12)
    plotter.add_mesh(mesh_gt, color="lightgray", opacity=1.0)

    plotter.subplot(0, 1)
    plotter.add_text(f"Reconstructed mesh: patient {patient_name}", font_size=12)
    plotter.add_mesh(mesh_pred, color="lightgray", opacity=1.0)

    plotter.subplot(0, 2)
    plotter.add_text(f"Reconstructed mesh: error", font_size=12)

    plotter.add_mesh(mesh_pred, scalars="error", cmap="jet_r", show_scalar_bar=True,
        scalar_bar_args=dict(
            title="",
            vertical=True,                 
            title_font_size=16,
            label_font_size=16,
            n_labels=5,
            fmt="%.2f",
            position_x=0.85,               # INSIDE the subplot
            position_y=0.1,
            width=0.05,
            height=0.7,
        ),
    )

    # ------------------------------------------------------------------
    # Fixed screen-space scale bar (map-style)
    # ------------------------------------------------------------------
    plotter.subplot(0, 2)
    add_static_scale_bar(
        plotter,
        length=10000.0,        # world units
        label="",
        pos=(60, 60),
    )

    # ------------------------------------------------------------------

    plotter.link_views()

    return plotter

def visually_check_sdfs_distribution(source_dir):

    files = list( Path(source_dir).iterdir() )

    n_rows, n_cols = 4, 6
    batch_size = n_rows * n_cols

    for batch_start in range(0, len(files), batch_size):
        batch_files = files[batch_start : batch_start + batch_size]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16,8))
        axes = axes.flatten()

        for i, file in enumerate(batch_files):
            patient_name = str(file.name).split("_")
            patient_name = patient_name[0] + "-" + patient_name[1]
            data = np.load(file)
            coords = data[:,:3]
            sdf = data[:,3:]
            sdf_epi = sdf[:,0]
            sdf_la  = sdf[:,1]
            sdf_ra  = sdf[:,2]

            axes[i].hist(sdf_epi, bins=100, color='yellow', alpha=0.5, label='epi')
            axes[i].hist(sdf_la, bins=100, color='red', alpha=0.5, label='la')
            axes[i].hist(sdf_ra, bins=100, color='skyblue', alpha=0.5, label='ra')
            axes[i].set_title(patient_name)

        # Hide any unused subplots in last batch
        for j in range(len(batch_files), batch_size):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()  # shows 16 plots at a time

def visually_check_all_surfaces(source_dir):

    patients_dirs =  list( Path(source_dir).iterdir() )

    n_rows, n_cols = 4,5
    batch_size = n_rows * n_cols

    for batch_start in range(0, len(patients_dirs), batch_size):

        batch_dirs = patients_dirs[batch_start : batch_start + batch_size]

        plotter = pv.Plotter(shape=(n_rows, n_cols), window_size=[1920, 1600])

        for i, patient_dir in enumerate(batch_dirs):

            patient = patient_dir.name
            epi = pv.read( patient_dir / "epicardium-processed.vtp")
            la = pv.read( patient_dir / "la_endo-processed.vtp")
            ra = pv.read( patient_dir / "ra_endo-processed.vtp")
            
            row = i // n_cols
            col = i % n_cols
            
            plotter.subplot(row,col)
            plotter.add_mesh(epi, color="white", opacity = 0.5)
            plotter.add_mesh(la, color="red", opacity= 1.0)
            plotter.add_mesh(ra, color="skyblue", opacity= 1.0)
            plotter.add_title(patient)
            plotter.show_grid()

        plotter.link_views()
        plotter.show()

        plotter.close()

    return

# ================================================================ #
# region latent space visualization
# ================================================================ #

def map_categories(patient_names, categories = ["AF", "LEU_NORM"]):
    # I just love python
    return [ categories["AF" not in name] for name in patient_names ] 

def plot_pca(latents, patients_names, save_fname):
    """
        Args:
        latents: (N, latent_size)
        patient_names: (N)

        Patient_names is assumed to indicize into latents row by row.
        So each latent is assumed it represents the corresponding patient.
    """
    
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents)
    
    df = pd.DataFrame({
        'PC1': latents_pca[:,0],
        'PC2': latents_pca[:,1],
        # 'PC3': latents_pca[:,2],
        'Category': map_categories(patients_names)
    })

    palette = {
        "AF": COLORS_PALETTE["neon_red"],  
        "LEU_NORM": COLORS_PALETTE["neon_green"], 
    }

    plt.figure(figsize=(7, 6))

    # --- Set background colors ---
    bckg_col = "#1D1D24DF"
    ax = plt.gca()
    ax.set_facecolor(bckg_col)       # plot area background
    plt.gcf().patch.set_facecolor(bckg_col)  # figure (outer) background

    sns.scatterplot(
        data=df,
        x='PC1', y='PC2',
        hue='Category',
        palette=palette,
        s=50,
    )

    # Optional: style tweaks for neon look
    ax.tick_params(colors='white')      # white ticks
    ax.spines[:].set_color('white')     # white border lines
    ax.xaxis.label.set_color('white')   # white axis labels
    ax.yaxis.label.set_color('white')
    ax.legend(facecolor=bckg_col, edgecolor=bckg_col, labelcolor='white')

    plt.savefig(save_fname, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
    plt.close()



if __name__ == "__main__":

    from pathlib import Path

    # PATIENTS_COORDS_AND_SDFS_DIR = Path("/home/navarri/AtriaProject/DATASETS/AtriaPointsAndSDF")

    # PATIENTS_NPY_DATA_DIR =  PATIENTS_COORDS_AND_SDFS_DIR / "single_patients_100000pts_npy"

    # visually_check_sdfs_distribution(PATIENTS_NPY_DATA_DIR)
    
    PATIENT_MESHES_DIR = Path("/home/navarri/AtriaProject/DATASETS/AtrialGeometries")

    visually_check_all_surfaces(PATIENT_MESHES_DIR)
    
    pass