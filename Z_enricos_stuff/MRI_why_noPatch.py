import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pyvista as pv

from ventricles_data_preparing import (
    extract_raw_ventricle_surfaces,
    make_surface_watertight,
)

# ============================================================
# PARAMETRI
# ============================================================

patient = "LEU_NORM_0093"

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/all_processed_files"
)

# ============================================================
# LOAD VTU
# ============================================================

patient_dir = ALL_PROCESSED_DIR / patient

vtu_path = (
    patient_dir /
    f"{patient}.vtu"
)

print("\nLOADING VTU")
print(vtu_path)

mesh = pv.read(vtu_path)

print("\nVTU LOADED")
print(mesh)

# ============================================================
# EXTRACT RAW SURFACES
# ============================================================

epi_raw, rv_raw, lv_raw = (
    extract_raw_ventricle_surfaces(
        mesh
    )
)

print("\nRAW LV")
print(lv_raw)

# ============================================================
# RAW BOUNDARY EDGES
# ============================================================

boundary_raw = lv_raw.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False,
)

print(
    f"\nRAW BOUNDARY EDGES: "
    f"{boundary_raw.n_cells}"
)

# ============================================================
# FIRST REPAIR
# ============================================================

print("\n================================================")
print("FIRST REPAIR")
print("================================================")

lv_first = make_surface_watertight(
    lv_raw
)

print("\nFIRST REPAIR")
print(lv_first)

# ------------------------------------------------------------
# CHECK ISHOLEPATCH
# ------------------------------------------------------------

print("\nCELL DATA KEYS")

print(
    list(lv_first.cell_data.keys())
)

if "isholepatch" in lv_first.cell_data:

    unique_values = np.unique(
        lv_first.cell_data["isholepatch"]
    )

    print("\nISHOLEPATCH UNIQUE VALUES")

    print(unique_values)

    patch_mask = (
        lv_first.cell_data["isholepatch"] == 1
    )

    n_patch_cells = np.sum(
        patch_mask
    )

    print(
        f"\nFIRST REPAIR PATCH CELLS: "
        f"{n_patch_cells}"
    )

    patches_first = lv_first.extract_cells(
        patch_mask
    )

    print("\nFIRST PATCHES")
    print(patches_first)

    if patches_first.n_cells > 0:

        conn_first = patches_first.connectivity()

        region_ids = np.unique(
            conn_first.cell_data["RegionId"]
        )

        print(
            f"\nFIRST REPAIR REGIONS: "
            f"{region_ids}"
        )

else:

    print(
        "\nNO ISHOLEPATCH AFTER FIRST REPAIR"
    )

# ------------------------------------------------------------
# FIRST REPAIR BOUNDARY EDGES
# ------------------------------------------------------------

boundary_first = lv_first.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False,
)

print(
    f"\nFIRST REPAIR BOUNDARY EDGES: "
    f"{boundary_first.n_cells}"
)

# ============================================================
# SECOND REPAIR
# ============================================================

print("\n================================================")
print("SECOND REPAIR")
print("================================================")

lv_second = make_surface_watertight(
    lv_first
)

print("\nSECOND REPAIR")
print(lv_second)

# ------------------------------------------------------------
# CHECK ISHOLEPATCH
# ------------------------------------------------------------

print("\nCELL DATA KEYS")

print(
    list(lv_second.cell_data.keys())
)

if "isholepatch" in lv_second.cell_data:

    unique_values = np.unique(
        lv_second.cell_data["isholepatch"]
    )

    print("\nISHOLEPATCH UNIQUE VALUES")

    print(unique_values)

    patch_mask = (
        lv_second.cell_data["isholepatch"] == 1
    )

    n_patch_cells = np.sum(
        patch_mask
    )

    print(
        f"\nSECOND REPAIR PATCH CELLS: "
        f"{n_patch_cells}"
    )

    patches_second = lv_second.extract_cells(
        patch_mask
    )

    print("\nSECOND PATCHES")
    print(patches_second)

    if patches_second.n_cells > 0:

        conn_second = patches_second.connectivity()

        region_ids = np.unique(
            conn_second.cell_data["RegionId"]
        )

        print(
            f"\nSECOND REPAIR REGIONS: "
            f"{region_ids}"
        )

else:

    print(
        "\nNO ISHOLEPATCH AFTER SECOND REPAIR"
    )

# ------------------------------------------------------------
# SECOND REPAIR BOUNDARY EDGES
# ------------------------------------------------------------

boundary_second = lv_second.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False,
)

print(
    f"\nSECOND REPAIR BOUNDARY EDGES: "
    f"{boundary_second.n_cells}"
)

# ============================================================
# VISUALIZATION
# ============================================================

plotter = pv.Plotter(
    shape=(1, 3),
    window_size=(2400, 800)
)

# ============================================================
# RAW
# ============================================================

plotter.subplot(0, 0)

plotter.add_text(
    "RAW",
    font_size=14
)

plotter.add_mesh(
    lv_raw,
    color="lightgray",
    opacity=0.4,
)

if boundary_raw.n_cells > 0:

    plotter.add_mesh(
        boundary_raw,
        color="red",
        line_width=5,
    )

# ============================================================
# FIRST REPAIR
# ============================================================

plotter.subplot(0, 1)

plotter.add_text(
    "FIRST REPAIR",
    font_size=14
)

plotter.add_mesh(
    lv_first,
    color="lightgray",
    opacity=0.25,
)

if "isholepatch" in lv_first.cell_data:

    patches_first = lv_first.extract_cells(
        lv_first.cell_data["isholepatch"] == 1
    )

    if patches_first.n_cells > 0:

        plotter.add_mesh(
            patches_first,
            color="red",
            show_edges=False,
        )

# ============================================================
# SECOND REPAIR
# ============================================================

plotter.subplot(0, 2)

plotter.add_text(
    "SECOND REPAIR",
    font_size=14
)

plotter.add_mesh(
    lv_second,
    color="lightgray",
    opacity=0.25,
)

if "isholepatch" in lv_second.cell_data:

    patches_second = lv_second.extract_cells(
        lv_second.cell_data["isholepatch"] == 1
    )

    if patches_second.n_cells > 0:

        plotter.add_mesh(
            patches_second,
            color="red",
            show_edges=False,
        )

plotter.link_views()

plotter.show()