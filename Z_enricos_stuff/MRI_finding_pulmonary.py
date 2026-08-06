import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pyvista as pv


# ============================================================
# PARAMETRI
# ============================================================

# patient = "AF001"
# patient = "LEU_BBB_21359"
# patient = "LEU_BBB_21471"
# patient = "yrm3230_v1"
patient = "LEU_NORM_1315"

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

patient_dir = ALL_PROCESSED_DIR / patient

rv_path = patient_dir / "rv_endo-processed.vtp"

# asse apex -> base
APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)


# ============================================================
# PESI LIKELIHOOD POLMONARE
# ============================================================

W_PROJ = 0.0   # basalità lungo asse apex-base
W_AREA = 0.1    # la polomonare è spesso una patch grande
W_GEOM = 0.9     # prior geometrico da regolare


# ============================================================
# LOAD RV
# ============================================================

rv = pv.read(rv_path)

print("\nRV loaded")
print(rv)

if "isholepatch" not in rv.cell_data:
    raise RuntimeError("Campo 'isholepatch' non trovato in rv.cell_data")


# ============================================================
# ESTRAI PATCH
# ============================================================

patches = rv.extract_cells(
    rv.cell_data["isholepatch"] == 1
)

print("\nPatch cells:", patches.n_cells)

if patches.n_cells == 0:
    raise RuntimeError("Nessuna hole patch trovata nel RV")


# ============================================================
# CONNECTIVITY
# ============================================================

patches_conn = patches.connectivity()

region_ids = np.unique(
    patches_conn.cell_data["RegionId"]
)

print("\nFound regions:", region_ids)


# ============================================================
# FUNZIONE CENTROIDE AREA-WEIGHTED
# ============================================================

def area_weighted_centroid(mesh):
    """
    Calcola il centroide pesato per area delle celle triangolari.
    Se qualcosa va storto, usa la media dei punti.
    """

    centers = mesh.cell_centers().points

    try:
        areas = mesh.compute_cell_sizes(
            length=False,
            area=True,
            volume=False
        ).cell_data["Area"]

        if np.sum(areas) <= 0:
            return mesh.points.mean(axis=0)

        return np.average(
            centers,
            axis=0,
            weights=areas
        )

    except Exception:
        return mesh.points.mean(axis=0)


# ============================================================
# ANALISI PATCH
# ============================================================

patch_infos = []

for rid in region_ids:

    patch = patches_conn.threshold(
        [rid - 0.5, rid + 0.5],
        scalars="RegionId"
    )

    centroid = area_weighted_centroid(patch)

    x, y, z = centroid

    # --------------------------------------------------------
    # PROIEZIONE APEX-BASE
    # --------------------------------------------------------

    projection = np.dot(
        centroid,
        APEX_BASE_AXIS
    )

    # --------------------------------------------------------
    # AREA PATCH
    # --------------------------------------------------------

    area = patch.area

    # --------------------------------------------------------
    # PRIOR GEOMETRICO
    # --------------------------------------------------------
    # Questo va eventualmente regolato guardando il plot.
    # Mantengo una forma simile al tuo codice.
    #
    # Possibili alternative:
    # geom_value = y + x
    # geom_value = y - x
    # geom_value = -x
    # geom_value = x

    geom_value = z

    patch_infos.append({
        "rid": rid,
        "projection": projection,
        "area": area,
        "geom_value": geom_value,
        "centroid": centroid,
        "mesh": patch,
    })

    print(
        f"Region {rid} | "
        f"proj={projection:.4f} | "
        f"area={area:.4f} | "
        f"geom={geom_value:.4f} | "
        f"centroid={centroid}"
    )


# ============================================================
# NORMALIZZAZIONE SCORE
# ============================================================

projections = np.array([p["projection"] for p in patch_infos])
areas = np.array([p["area"] for p in patch_infos])
geom_values = np.array([p["geom_value"] for p in patch_infos])


def normalize(values):
    values = np.asarray(values)

    vmin = values.min()
    vmax = values.max()

    if np.isclose(vmax, vmin):
        return np.ones_like(values)

    return (values - vmin) / (vmax - vmin)


projection_scores = normalize(projections)
area_scores = normalize(areas)
geom_scores = normalize(geom_values)


# ============================================================
# LIKELIHOOD
# ============================================================

print("\nPatch likelihoods:\n")

for i, p in enumerate(patch_infos):

    projection_score = projection_scores[i]
    area_score = area_scores[i]
    geom_score = geom_scores[i]

    likelihood = (
        W_PROJ * projection_score +
        W_AREA * area_score +
        W_GEOM * geom_score
    )

    p["projection_score"] = projection_score
    p["area_score"] = area_score
    p["geom_score"] = geom_score
    p["likelihood"] = likelihood

    print(
        f"Region {p['rid']} | "
        f"proj={projection_score:.3f} | "
        f"area={area_score:.3f} | "
        f"geom={geom_score:.3f} | "
        f"LIK={likelihood:.3f}"
    )


# ============================================================
# SORT LIKELIHOODS
# ============================================================

patch_infos = sorted(
    patch_infos,
    key=lambda x: x["likelihood"],
    reverse=True
)

print("\nLIKELIHOOD RANKING\n")

for p in patch_infos:

    print(
        f"rid={p['rid']} | "
        f"LIK={p['likelihood']:.3f} | "
        f"area={p['area']:.4f} | "
        f"proj={p['projection']:.4f} | "
        f"geom={p['geom_value']:.4f}"
    )


# ============================================================
# CHOOSE Pulmunary PATCH
# ============================================================

pulmunary_patch_info = patch_infos[0]

pulmunary_region = pulmunary_patch_info["rid"]

print(
    f"\nChosen pulmunary region: "
    f"{pulmunary_region}"
)

pulmunary_centroid = pulmunary_patch_info["centroid"]

print("Pulmunary centroid:", pulmunary_centroid)


# ============================================================
# CREA LABEL
# ============================================================

labels = np.zeros(
    rv.n_cells,
    dtype=np.int8
)

patch_region_ids = patches_conn.cell_data["RegionId"]

patch_cell_ids = np.where(
    rv.cell_data["isholepatch"] == 1
)[0]

pulmunary_mask = (
    patch_region_ids == pulmunary_region
)

labels[
    patch_cell_ids[pulmunary_mask]
] = 1

rv.cell_data["pulmunary_patch"] = labels


# ============================================================
# ESTRAI PATCH PULUMUNARY
# ============================================================

pulmunary_cells = rv.extract_cells(
    rv.cell_data["pulmunary_patch"] == 1
)

other_patch_mask = (
    (rv.cell_data["isholepatch"] == 1) &
    (rv.cell_data["pulmunary_patch"] == 0)
)

other_cells = rv.extract_cells(
    other_patch_mask
)


# ============================================================
# PLOT
# ============================================================

plotter = pv.Plotter()

# RV completa
plotter.add_mesh(
    rv,
    color="lightgray",
    opacity=0.25,
)

# patch polmonare
if pulmunary_cells.n_points > 0:
    plotter.add_mesh(
        pulmunary_cells,
        color="red",
        show_edges=True,
        line_width=2,
        label="Pulmunary patch"
    )

# altre patch
# plotter.add_mesh(
#     other_cells,
#     color="lightblue",
#     opacity=0.9,
#     show_edges=True,
#     label="Other hole patches"
# )
if other_cells.n_points > 0 and other_cells.n_cells > 0:
    plotter.add_mesh(
        other_cells,
        color="lightblue",
        opacity=0.9,
        show_edges=True,
        label="Other hole patches",
    )
else:
    print("No other hole patches to plot.")

# centroide tricuspide
centroid_sphere = pv.Sphere(
    radius=0.03,
    center=pulmunary_centroid
)

plotter.add_mesh(
    centroid_sphere,
    color="magenta",
    label="Pulmunary centroid"
)


# ============================================================
# PUNTI RAPPRESENTATIVI
# ============================================================

for p in patch_infos:

    sphere = pv.Sphere(
        radius=0.02,
        center=p["centroid"]
    )

    if p["rid"] == pulmunary_region:
        color = "yellow"
    else:
        color = "blue"

    plotter.add_mesh(
        sphere,
        color=color
    )

    plotter.add_point_labels(
        [p["centroid"]],
        [f"rid {p['rid']} | {p['likelihood']:.2f}"],
        font_size=10,
        point_size=5,
    )


# ============================================================
# ORIGINE
# ============================================================

origin = pv.Sphere(
    radius=0.02,
    center=(0, 0, 0)
)

plotter.add_mesh(
    origin,
    color="black"
)


# ============================================================
# ASSE APEX-BASE
# ============================================================

axis_length = 2.0

p0 = -axis_length * APEX_BASE_AXIS
p1 = axis_length * APEX_BASE_AXIS

axis_line = pv.Line(p0, p1)

plotter.add_mesh(
    axis_line,
    color="green",
    line_width=5,
)

arrow = pv.Arrow(
    start=(0, 0, 0),
    direction=APEX_BASE_AXIS,
    scale=0.3
)

plotter.add_mesh(
    arrow,
    color="yellow"
)


# ============================================================
# ASSI XYZ + COORDINATE
# ============================================================

plotter.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)

plotter.add_axes()
plotter.add_legend()

plotter.show()