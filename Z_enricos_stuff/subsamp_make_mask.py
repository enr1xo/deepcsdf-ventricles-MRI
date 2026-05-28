"""
prendiamo un file npy, definito in una dir, composto da 5mila righe e 6 colonne. delle righe:
    le prime 1500 sono punti samplati attorno all'epicardio
    i secondi 1750 sono relativi all'endocardio sinistro
    gli ultimi 1750 all'endocardio destro

le colonne invece indicano il valore della sdf nell'ordine rispetto a epicardio endo sinistro ed endo destro.

vogliamo vedere cosa succede se subsampliamo i punti, sia per superficie che per sdf.
vogliamo verificare che usando una mschera si ottiene qualcosa di sensato.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

patients_npy_filepath = Path("/home/rizzardi/Schreibtisch/sampling_noise_study_npy/5k/S_0.025-L_0.75-R_0.5")

patient_name = "AF006"

if patients_npy_filepath.is_dir():

    matching_files = list(
        patients_npy_filepath.glob(f"*{patient_name}*.npy")
    )

    if len(matching_files) == 0:
        raise FileNotFoundError(
            f"Nessun file trovato per paziente '{patient_name}'"
        )

    if len(matching_files) > 1:
        print("\nTrovati più file:")
        for f in matching_files:
            print("  ", f.name)

        raise ValueError(
            f"Più file matchano il paziente '{patient_name}'"
        )

    npy_file = matching_files[0]

else:
    npy_file = patients_npy_filepath

print(f"Loading: {npy_file}")

data = np.load(npy_file)

print("Data shape:", data.shape)

if data.shape[1] != 6:
    raise ValueError(f"Mi aspettavo 6 colonne, trovate {data.shape[1]}")

coords = data[:, :3]
sdfs = data[:, 3:]

# -----------------------------
# Split per superficie
# -----------------------------
epi_start, epi_end = 0, 1500
lv_start, lv_end = 1500, 1500 + 1750
rv_start, rv_end = 1500 + 1750, 5000

surface_slices = {
    "epicardium": slice(epi_start, epi_end),
    "lv_endo": slice(lv_start, lv_end),
    "rv_endo": slice(rv_start, rv_end),
}

sdf_columns = {
    "epicardium": 0,
    "lv_endo": 1,
    "rv_endo": 2,
}

for surface_name, sl in surface_slices.items():
    print(f"{surface_name}: rows {sl.start}:{sl.stop}, count = {sl.stop - sl.start}")

# -----------------------------
# Subsampling
# -----------------------------
rng = np.random.default_rng(seed=42)

subsample_fraction = 0.25

subsampled_indices = {}

for surface_name, sl in surface_slices.items():
    indices = np.arange(sl.start, sl.stop)

    n_keep = int(len(indices) * subsample_fraction)

    chosen = rng.choice(indices, size=n_keep, replace=False)
    chosen = np.sort(chosen)

    subsampled_indices[surface_name] = chosen

    print(f"{surface_name}: kept {len(chosen)} / {len(indices)} points")

# Unione di tutti gli indici subsamplati
all_subsampled_indices = np.concatenate(list(subsampled_indices.values()))
all_subsampled_indices = np.sort(all_subsampled_indices)

coords_sub = coords[all_subsampled_indices]
sdfs_sub = sdfs[all_subsampled_indices]

print("Subsampled coords shape:", coords_sub.shape)
print("Subsampled sdfs shape:", sdfs_sub.shape)

# -----------------------------
# Verifica maschere SDF vicine a zero
# -----------------------------
threshold = 0.1


for surface_name, sl in surface_slices.items():
    sdf_col = sdf_columns[surface_name]

    coords_surface = coords[sl]
    sdfs_surface = sdfs[sl]

    mask_near_surface = np.abs(sdfs_surface[:, sdf_col]) <= threshold

    selected_coords = coords_surface[mask_near_surface]
    selected_sdf = sdfs_surface[mask_near_surface, sdf_col]

    # print()
    # print(f"{surface_name}")
    # print(f"  punti totali nella sezione: {coords_surface.shape[0]}")
    # print(f"  punti con |SDF| <= {threshold}: {selected_coords.shape[0]}")
    # print(f"  SDF min/max selezionata: {selected_sdf.min():.6f}, {selected_sdf.max():.6f}")


#---------------------------------
# blend mask + random subsampling

# 1) SDF filter
# 2) percentage subsampling
# -----------------------------
rng = np.random.default_rng(seed=42)

threshold = 0.05
subsample_fraction = 0.25

combined_indices = {}

for surface_name, sl in surface_slices.items():
    sdf_col = sdf_columns[surface_name]

    indices = np.arange(sl.start, sl.stop)

    sdfs_surface = sdfs[indices]

    # 1) filtro SDF
    mask_near_surface = np.abs(sdfs_surface[:, sdf_col]) <= threshold
    filtered_indices = indices[mask_near_surface]

    # 2) subsampling percentuale sui punti filtrati
    n_keep = int(len(filtered_indices) * subsample_fraction)

    if n_keep > 0:
        chosen = rng.choice(filtered_indices, size=n_keep, replace=False)
        chosen = np.sort(chosen)
    else:
        chosen = np.array([], dtype=int)

    combined_indices[surface_name] = chosen

    print()
    print(f"{surface_name}")
    print(f"  punti totali nella sezione: {len(indices)}")
    print(f"  punti dopo filtro SDF |SDF| <= {threshold}: {len(filtered_indices)}")
    print(f"  punti tenuti dopo subsampling {subsample_fraction*100:.1f}%: {len(chosen)}")

# Unione degli indici finali
all_combined_indices = np.concatenate(list(combined_indices.values()))
all_combined_indices = np.sort(all_combined_indices)

coords_sub = coords[all_combined_indices]
sdfs_sub = sdfs[all_combined_indices]

print()
print("Combined coords shape:", coords_sub.shape)
print("Combined sdfs shape:", sdfs_sub.shape)




# -----------------------------
# Plot distribuzione SDF
# -----------------------------
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# for ax, (surface_name, sl) in zip(axes, surface_slices.items()):
#     sdf_col = sdf_columns[surface_name]

#     sdf_values = sdfs[sl, sdf_col]

#     ax.hist(sdf_values, bins=50)
#     ax.axvline(-threshold, linestyle="--")
#     ax.axvline(threshold, linestyle="--")
#     ax.set_title(surface_name)
#     ax.set_xlabel("SDF")
#     ax.set_ylabel("count")

# plt.tight_layout()
# plt.show()

# -----------------------------
# Plot 3D: punti originali vs subsamplati
# -----------------------------
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2)
ax1.set_title("Original points")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(coords_sub[:, 0], coords_sub[:, 1], coords_sub[:, 2], s=4)
ax2.set_title(f"Subsampled points ({subsample_fraction:.0%})")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

plt.tight_layout()
plt.show()

# -----------------------------
# Plot per superficie dopo maschera
# -----------------------------
fig = plt.figure(figsize=(15, 5))

for i, (surface_name, sl) in enumerate(surface_slices.items(), start=1):
    sdf_col = sdf_columns[surface_name]

    coords_surface = coords[sl]
    sdfs_surface = sdfs[sl]

    mask = np.abs(sdfs_surface[:, sdf_col]) <= threshold
    selected_coords = coords_surface[mask]

    ax = fig.add_subplot(1, 3, i, projection="3d")
    ax.scatter(
        selected_coords[:, 0],
        selected_coords[:, 1],
        selected_coords[:, 2],
        s=4,
    )
    ax.set_title(f"{surface_name} |SDF| <= {threshold}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

plt.tight_layout()
plt.show()


# -----------------------------
# Plot per superficie dopo maschera
# + subplot combinato
# -----------------------------
fig = plt.figure(figsize=(20, 5))

all_selected_coords = []

for i, (surface_name, sl) in enumerate(surface_slices.items(), start=1):

    sdf_col = sdf_columns[surface_name]

    coords_surface = coords[sl]
    sdfs_surface = sdfs[sl]

    mask = np.abs(sdfs_surface[:, sdf_col]) <= threshold

    selected_coords = coords_surface[mask]

    # salviamo per il plot combinato
    all_selected_coords.append(selected_coords)

    ax = fig.add_subplot(1, 4, i, projection="3d")

    ax.scatter(
        selected_coords[:, 0],
        selected_coords[:, 1],
        selected_coords[:, 2],
        s=4,
    )

    ax.set_title(f"{surface_name}\n|SDF| <= {threshold}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

# -----------------------------
# Plot combinato
# -----------------------------
all_selected_coords = np.concatenate(all_selected_coords, axis=0)

ax_combined = fig.add_subplot(1, 4, 4, projection="3d")

ax_combined.scatter(
    all_selected_coords[:, 0],
    all_selected_coords[:, 1],
    all_selected_coords[:, 2],
    s=4,
)

ax_combined.set_title(
    f"Combined\nTotal points = {len(all_selected_coords)}"
)

ax_combined.set_xlabel("x")
ax_combined.set_ylabel("y")
ax_combined.set_zlabel("z")

plt.tight_layout()
plt.show()


# -----------------------------
# Plot combined:
# SDF filter + random subsampling
# -----------------------------
fig = plt.figure(figsize=(20, 5))

all_selected_coords = []

for i, surface_name in enumerate(surface_slices.keys(), start=1):

    chosen = combined_indices[surface_name]

    selected_coords = coords[chosen]

    all_selected_coords.append(selected_coords)

    ax = fig.add_subplot(1, 4, i, projection="3d")

    ax.scatter(
        selected_coords[:, 0],
        selected_coords[:, 1],
        selected_coords[:, 2],
        s=4,
    )

    ax.set_title(
        f"{surface_name}\ncombined filter"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

# -----------------------------
# Plot combinato finale
# -----------------------------
all_selected_coords = np.concatenate(all_selected_coords, axis=0)

ax_combined = fig.add_subplot(1, 4, 4, projection="3d")

ax_combined.scatter(
    all_selected_coords[:, 0],
    all_selected_coords[:, 1],
    all_selected_coords[:, 2],
    s=4,
)

ax_combined.set_title(
    f"Combined\nTotal points = {len(all_selected_coords)}"
)

ax_combined.set_xlabel("x")
ax_combined.set_ylabel("y")
ax_combined.set_zlabel("z")

plt.tight_layout()
plt.show()