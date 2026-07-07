from pathlib import Path

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# =========================================================
# INPUT
# =========================================================

patient_name = "AF001"

# npy_file = Path(
#     "/home/rizzardi/Schreibtisch/sampling_noise_study_npy/5k/"
#     "S_0.025-L_0.75-R_0.5/"
#     f"{patient_name}-epi_lv_rv_5000_coords_and_sdf.npy"
# )

npy_file = Path(
    r"C:\Users\e.rizzardi\OneDrive\Desktop\graz_June\square_samples_min_dist_1mm\AF001_square_samples.npy")
    
# mesh_file = Path(
#     "/home/rizzardi/Schreibtisch/all_processed_files/"
#     f"{patient_name}/epicardium-processed.vtp"
# )


mesh_file = Path(r"C:\Users\e.rizzardi\OneDrive\Desktop\processed_patients\AF001\epicardium-processed.vtp")


# MRI-like parameters
epsilon_mm = 0.4   # thickness
delta_mm = 8.0     # spacing

# =========================================================
# LOAD DATA
# =========================================================

data = np.load(npy_file)

coords = data[:, :3]

# =========================================================
# SPLIT SURFACES
# =========================================================

surface_slices = {
    "epicardium": slice(0, 1500),
    "lv_endo": slice(1500, 3250),
    "rv_endo": slice(3250, 5000),
}

# =========================================================
# CHOOSE SURFACE TO SLICE
# =========================================================

surface_name = "rv_endo"

sl = surface_slices[surface_name]

# coords_surface = coords[sl]
coords_surface = coords

print()
print(f"Using surface: {surface_name}")
print("Surface coords shape:", coords_surface.shape)

# print("coords shape:", coords.shape)

# =========================================================
# RECOVER ORIGINAL SCALE
# =========================================================

mesh = pv.read(mesh_file)

scale_to_original = mesh.field_data["scale-tooriginalrange"][0]

print()
print("Recovered anatomical scale:")
print(scale_to_original)

# coords are normalized
# convert mm -> normalized units

# =========================================================
# convert mm -> micron
# =========================================================

epsilon_um = epsilon_mm * 1000.0
delta_um = delta_mm * 1000.0

# =========================================================
# micron -> normalized coordinates
# =========================================================

epsilon = epsilon_um / scale_to_original
delta = delta_um / scale_to_original

print()
print("Normalized parameters:")
print("epsilon =", epsilon)
print("delta   =", delta)

# =========================================================
# APEX-BASE AXIS
# =========================================================

axis = np.array([-1.0, 1.0, 0.0])
axis = axis / np.linalg.norm(axis)

# =========================================================
# PROJECT POINTS
# =========================================================

# proj = coords @ axis
proj = coords_surface @ axis

proj_min = proj.min()
proj_max = proj.max()

length = proj_max - proj_min

print()
print("Projection range:")
print(proj_min, proj_max)

# =========================================================
# GENERATE MRI-LIKE SLICES
# =========================================================

plane_centers = np.arange(proj_min, proj_max, delta)

print()
print(f"Number of slices: {len(plane_centers)}")

# =========================================================
# FILTER POINTS
# =========================================================

# global_mask = np.zeros(len(coords), dtype=bool)
global_mask = np.zeros(len(coords_surface), dtype=bool)

for c in plane_centers:

    current_mask = np.abs(proj - c) <= (epsilon / 2)

    global_mask |= current_mask

# filtered_coords = coords[global_mask]
filtered_coords = coords_surface[global_mask]

print()
print("Original points :", len(coords))
print("Filtered points :", len(filtered_coords))

# =========================================================
# PLOT
# =========================================================

fig = plt.figure(figsize=(14, 7))

ax = fig.add_subplot(111, projection="3d")

# full anatomy
ax.scatter(
    coords[:, 0],
    coords[:, 1],
    coords[:, 2],
    s=1,
    alpha=0.03,
)

# MRI-like slices
ax.scatter(
    filtered_coords[:, 0],
    filtered_coords[:, 1],
    filtered_coords[:, 2],
    s=8,
)

ax.set_title(
    f"MRI-like sampling\n"
    f"thickness={epsilon_mm} mm | spacing={delta_mm} mm"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_box_aspect([1,1,1])

plt.tight_layout()
plt.show()