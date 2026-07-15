from pathlib import Path

import numpy as np
import pyvista as pv


# ============================================================
# FILE
# ============================================================

NPY_FILE = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/generated_npy_three_axis/AF001_three_axis_mri_samples.npy"
)


# ============================================================
# LOAD
# ============================================================

data = np.load(NPY_FILE)

points = data[:, :3]


# ============================================================
# PLOT
# ============================================================

plotter = pv.Plotter()

plotter.add_points(
    points,
    color="red",
    point_size=5,
    render_points_as_spheres=True,
)

plotter.add_axes()
plotter.show_grid()

plotter.show()