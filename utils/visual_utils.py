import pyvista as pv
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.cm as mplcm
from matplotlib.lines import Line2D
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from tensorboard.backend.event_processing import event_accumulator # to read from events.out files created during training by Tensorboard logger
from pathlib import Path

COLORS_PALETTE = {
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


}


# ================================================================ #
# region general visualization
# ================================================================ #
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
# region training visualization
# ================================================================ #
def get_legend_label(log_dir):
    " give meaningful name to legend. This will be a pain to automate"
    
    exp_name = str(log_dir).split("/")[-2]
    version = Path(log_dir).name

    specs = json.load( open( next( Path(log_dir).glob("hparams.json"), None) ) )

    if exp_name in {
        "LipAlphaAndCodeReg",
        "LipAlphaCodeRegAndLatentSize16",
        "LipAlphaCodeRegAndLatentSize32",
        "LipAlphaCodeRegAndLatentSize64"
    }:

        alpha = specs["lipschitz_alpha"]
        lamb = specs["code_reg_lambda"]
        #label = r"$\lambda_{lip}$" + f" = {alpha:.0e},  " + r"$\lambda_{prior}$" + f" = {lamb:.0e}"  # .0e} → scientific notation with no decimal places
        label = (
            rf"$\lambda_{{lip}} = {alpha:.0e}$" "\n"
            rf"$\lambda_{{prior}} = {lamb:.0e}$"
        )
    elif exp_name == "LipLayersAndCodeReg":
        lip = specs["Network_specs"]["lipschitz_layers"]
        lamb = specs["code_reg_lambda"]
        label = f"Spectral = {lip} " + rf"$\lambda = {lamb:.0e}$" 

    elif exp_name == "SpectralLaysAndAct":
        lip = specs["Network_specs"]["lipschitz_layers"]
        act = specs["Network_specs"]["activation"]
        label = f"Spectral = {lip}, {act}" 

    elif exp_name == "LatentSizeAndCodeReg":
        latent = specs["Network_specs"]["latent_size"]
        lamb = specs["code_reg_lambda"]
        label = f"latent = {latent}" + rf"$\lambda = {lamb:.0e}$" 

    else:
        label = version

    return label  

def plot_experiment_runs_events(
    log_dir: str | Path,
    plot_scalars: list = ["latents_mean_L2_squared", "regression_loss"],
    linewidth: float = 1.0,
    cmap_name = "gist_ncar",
    alpha = 0.6,
    grid: bool = True,
    fontsize: int = 14,
    title_fontsize=18,              # NEW
    title_mapping: dict | None = None,  # NEW
    save_fname=None
):
    """
    Args:
        `log_dir` : Path to the experiment folder containing all version folders (where events.out.tfevents.* files are)
        `plot_scalars` : list of scalar names to plot
        `linewidth` : line width for plot lines
        `grid` : whether to show grid (major ticks only)
        `save_fname` : if provided, saves the figure to this file path
    """
    #TODO: make legend labels more informative instead of version_x

    log_dir = Path(log_dir)
    version_dirs = list(log_dir.glob("version_*"))
    version_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))  # numeric sort

    num_scalars = len(plot_scalars)

    # Layout selection
    if num_scalars == 1:
        nrows, ncols = 1
        figsize = (10,8)
        raise ValueError("Currently not available for single scalar plot")
    elif num_scalars == 2:
        nrows, ncols = 1, 2
        figsize = (20, 8)
    elif num_scalars == 3:
        nrows, ncols = 1, 3
        figsize = (30, 8)
    elif num_scalars == 4:
        nrows, ncols = 2, 2
        figsize = (20, 16)
    else:
        raise ValueError("Too many scalars to plot requested.")

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()[:num_scalars]

    cmap = plt.get_cmap(cmap_name)

    colors = [cmap(i / len(version_dirs)) for i in range(len(version_dirs))]

    # Keep track of handles for shared legend
    legend_handles, legend_labels = [], []

    for idx, logdir in enumerate(version_dirs):
        ea = event_accumulator.EventAccumulator(str(logdir), size_guidance={'scalars': 0})
        ea.Reload()

        scalars = set(ea.Tags()['scalars'])
        missing = set(plot_scalars) - scalars
        if missing:
            raise ValueError(f"Some requested scalar metrics not found in event file: {missing}. Available are {scalars}")

        # brief check to only plot meaningful versions
        skip_version = False
        for i, tag in enumerate(plot_scalars):
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            if min(values) <= 1e-10:
                skip_version = True

        if not skip_version:
            for i, tag in enumerate(plot_scalars):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                line, = axs[i].plot(steps, values, c = colors[idx], alpha = alpha, linewidth=linewidth, label=logdir.name)

                # Collect handles/labels from first subplot only
                if i == 0:
                    # Create thicker Line2D for the legend
                    legend_line = Line2D([0], [0], color=colors[idx], lw=linewidth*3)  # more thickness in legend
                    legend_handles.append(legend_line)
                    legend_labels.append( get_legend_label(logdir))
                    # legend_handles.append(line) # use the plotted line
                    # legend_labels.append(logdir.name)
                try:
                    axs[i].set_yscale("log")
                except:
                    pass

                # axs[i].set_xlabel("Global training step", fontsize=fontsize)
                # axs[i].set_ylabel("Value", fontsize=fontsize)
                # Use custom title if provided
                if title_mapping and tag in title_mapping:
                    title_text = title_mapping[tag]
                else:
                    title_text = tag

                axs[i].set_title(title_text, fontsize=title_fontsize)
                if grid:
                    axs[i].grid(True, which='major', linestyle='-', alpha=0.5)


    fig.subplots_adjust(right=0.78)  # lets you shrink or shift the subplots inside the figure by specifying fractions of the figure

    # Shared legend outside rightmost subplot
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center left',         # the legend's left edge is at bbox_to_anchor x
        bbox_to_anchor=(0.8, 0.5),  # bbox_to_anchor coordinates are in figure fraction units (0–1).
        fontsize=fontsize,
        #title="Versions",
        borderaxespad=0
    )

    if save_fname:
        save_fname = Path(save_fname)
        save_fname.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fname, dpi=300, transparent=True)
        plt.close()
        return

    plt.show()



# ================================================================ #
# region meshes visualization
# ================================================================ #
def plot_gt_vs_reconstructed_with_error(
        mesh_gt: pv.PolyData | pv.UnstructuredGrid,
        mesh_pred: pv.PolyData | pv.UnstructuredGrid,
        patient_name,
        signed_distances_pred_from_gt = None,
        off_screen = False
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

    plotter.link_views()

    return plotter

def render_mesh_offscreen(
    mesh,
    scalars=None,
    clim=None,
    cmap="jet_r",
    camera_position="xy",
    image_size=(1200, 1200),
):
    """
    Render a mesh off-screen and return RGB image array.
    """
    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    plotter.set_background("white")

    if scalars is not None:
        plotter.add_mesh(
            mesh,
            scalars=scalars,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=False,
        )
    else:
        plotter.add_mesh(
            mesh,
            color="lightgray"
        )  

    plotter.camera_position = camera_position
    img = plotter.screenshot(return_img=True)
    plotter.close()

    return img

def plot_two_gt_vs_reconstructed_publication_style(
    mesh_gt_1, mesh_pred_1, patient_name_1,
    mesh_gt_2, mesh_pred_2, patient_name_2,
    signed_distances_1=None,
    signed_distances_2=None,
    cam_pos_1="xy", cam_pos_2="xy",
    output_path="figure.png",
    dpi=600,
):
    """
    Render publication-ready 2x2 figure with shared colorbar.
    """

    # ---------------- Compute signed distances ----------------
    def compute_signed_distances(mesh_gt, mesh_pred, signed_distances):
        if signed_distances is None:
            implicit_distance = vtkImplicitPolyDataDistance()
            implicit_distance.SetInput(mesh_gt)
            signed_distances = np.array(
                [implicit_distance.EvaluateFunction(p) for p in mesh_pred.points]
            )
        return signed_distances

    signed_distances_1 = compute_signed_distances(
        mesh_gt_1, mesh_pred_1, signed_distances_1
    )
    signed_distances_2 = compute_signed_distances(
        mesh_gt_2, mesh_pred_2, signed_distances_2
    )

    # ---------------- Global color limits ----------------
    global_min = min(signed_distances_1.min(), signed_distances_2.min())
    global_max = max(signed_distances_1.max(), signed_distances_2.max())
    global_clim = [global_min, global_max]

    # Copy meshes
    mesh_pred_1 = mesh_pred_1.copy()
    mesh_pred_2 = mesh_pred_2.copy()
    mesh_pred_1["error"] = signed_distances_1
    mesh_pred_2["error"] = signed_distances_2

    # ---------------- Render images ----------------
    print("Rendering meshes off-screen...")

    img_gt_1 = render_mesh_offscreen(mesh_gt_1, camera_position=cam_pos_1)
    img_err_1 = render_mesh_offscreen(
        mesh_pred_1, scalars="error", clim=global_clim, camera_position=cam_pos_1
    )

    img_gt_2 = render_mesh_offscreen(mesh_gt_2, camera_position=cam_pos_2)
    img_err_2 = render_mesh_offscreen(
        mesh_pred_2, scalars="error", clim=global_clim, camera_position=cam_pos_2
    )

    # ---------------- Matplotlib Layout ----------------
    plt.rcParams.update({
        "font.family": "serif", # nice style !!
        "font.size": 12,
    })

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        2, 3, 
        width_ratios=[1, 1, 0.06], # the last colum is for legend !!
        wspace=0.02,
        hspace=0.02
    )

    # Row 1
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_gt_1)
    ax.set_title(f"{patient_name_1} – Ground Truth", fontsize=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img_err_1)
    ax.set_title(f"{patient_name_1} – Reconstructed", fontsize=12)
    ax.axis("off")

    # Row 2
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(img_gt_2)
    ax.set_title(f"{patient_name_2} – Ground Truth", fontsize=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(img_err_2)
    ax.set_title(f"{patient_name_2} – Reconstructed", fontsize=12)
    ax.axis("off")

    # ---------------- Shared Colorbar ----------------
    norm = mplcolors.Normalize(vmin=global_clim[0], vmax=global_clim[1])
    sm = mplcm.ScalarMappable(norm=norm, cmap="jet_r")

    cax = fig.add_subplot(gs[:, 2])

    # # automatic color bar 
    # cbar = fig.colorbar(sm, cax=cax)
    # set number and spacing of ticks instead
    ticks = np.linspace(global_clim[0], global_clim[1], 7)
    cbar = fig.colorbar(sm, cax=cax, ticks=ticks)

    cbar.set_label("Signed distance (mm)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # consistent numeric formatting
    cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks])


    # ---------------- Save ----------------
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved publication figure to {output_path}")

