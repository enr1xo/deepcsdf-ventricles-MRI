from pathlib import Path
import numpy as np
import pyvista as pv


# ============================================================
# PARAMETERS
# ============================================================

PATIENT = "S62"

ALL_PROCESSED_DIR = Path(
    r"/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

SAMPLES_DIR = Path(
    r"/home/rizzardi/Schreibtisch/graz_June/square_samples_min_dist_1mm"
)

SAMPLES_PATH = SAMPLES_DIR / f"{PATIENT}_square_samples.npy"

# "epi", "lv", "rv", oppure None per punti rossi
SDF_TO_PLOT = "rv"

SURFACE_OPACITY = 0.15
GLYPH_RADIUS = 0.010

PLOT_SURFACES = True
PLOT_POINTS = True


# ============================================================
# LOAD DATA
# ============================================================

samples = np.load(SAMPLES_PATH)

points = samples[:, :3]

sdf_cols = {
    "epi": 3,
    "lv": 4,
    "rv": 5,
}

if SDF_TO_PLOT is not None:
    sdf = samples[:, sdf_cols[SDF_TO_PLOT]]
    valid = np.isfinite(sdf)
    points = points[valid]
    sdf = sdf[valid]
else:
    sdf = None

print("samples shape:", samples.shape)
print("points shape:", points.shape)
print("points min:", points.min(axis=0))
print("points max:", points.max(axis=0))

patient_dir = ALL_PROCESSED_DIR / PATIENT

epi_path = patient_dir / "epicardium-processed.vtp"
lv_path = patient_dir / "lv_endo-processed.vtp"
rv_path = patient_dir / "rv_endo-processed.vtp"

print("epi exists:", epi_path.exists(), epi_path)
print("lv exists:", lv_path.exists(), lv_path)
print("rv exists:", rv_path.exists(), rv_path)

epi = pv.read(epi_path)
lv = pv.read(lv_path)
rv = pv.read(rv_path)

print("epi bounds:", epi.bounds)
print("lv bounds:", lv.bounds)
print("rv bounds:", rv.bounds)


# ============================================================
# PLOT
# ============================================================

pv.OFF_SCREEN = False
pv.set_plot_theme("document")

plotter = pv.Plotter(
    window_size=(1600, 1200),
    notebook=False,
    off_screen=False,
)

# ------------------------------------------------------------
# Surfaces
# ------------------------------------------------------------

if PLOT_SURFACES:
    plotter.add_mesh(
        epi,
        color="lightgray",
        opacity=SURFACE_OPACITY,
        show_edges=False,
    )

    plotter.add_mesh(
        lv,
        color="blue",
        opacity=SURFACE_OPACITY,
        show_edges=False,
    )

    plotter.add_mesh(
        rv,
        color="green",
        opacity=SURFACE_OPACITY,
        show_edges=False,
    )

# ------------------------------------------------------------
# Points as glyphs
# ------------------------------------------------------------

if PLOT_POINTS:
    cloud = pv.PolyData(points)

    if sdf is not None:
        cloud[f"sdf_{SDF_TO_PLOT}"] = sdf

    sphere = pv.Sphere(
        radius=GLYPH_RADIUS,
        theta_resolution=10,
        phi_resolution=10,
    )

    glyphs = cloud.glyph(
        geom=sphere,
        scale=False,
        orient=False,
    )

    print("glyphs points:", glyphs.n_points)
    print("glyphs cells:", glyphs.n_cells)

    if sdf is not None:
        abs_max = np.nanmax(np.abs(sdf))

        plotter.add_mesh(
            glyphs,
            scalars=f"sdf_{SDF_TO_PLOT}",
            cmap="bwr",
            clim=[-abs_max, abs_max],
            show_scalar_bar=True,
        )
    else:
        plotter.add_mesh(
            glyphs,
            color="red",
        )

# ------------------------------------------------------------
# Extra reference sphere at origin
# ------------------------------------------------------------

# plotter.add_mesh(
#     pv.Sphere(radius=0.05, center=(0, 0, 0)),
#     color="yellow",
# )

plotter.add_text(
    f"{PATIENT}\n"
    f"SDF plotted: {SDF_TO_PLOT}",
    font_size=12,
)

plotter.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)

plotter.add_axes()
plotter.reset_camera()

plotter.show(
    interactive=True,
    auto_close=False,
)