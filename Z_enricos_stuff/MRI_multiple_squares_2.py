"""
Costruisce e visualizza quadrati MRI-like lungo l'asse C_area -> A_maxD.

- Legge C_area e A_maxD da CSV
- Carica LV, RV ed epicardio
- Costruisce quadrati ortogonali all'asse apex-base
- Spaziatura controllata da SQUARE_SPACING_MM e scale-tooriginalrange
- Stop: al massimo una slice oltre l'apex
- Due slice prima del centroide
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv


# ============================================================
# PYVISTA SETTINGS
# ============================================================

pv.OFF_SCREEN = False
pv.set_plot_theme("document")
pv.global_theme.jupyter_backend = "none"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

ALL_PROCESSED_DIR = Path(
    r"/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

CSV_PATH = Path(
    r"/home/rizzardi/Schreibtisch/MRI_model/mitral_Carea_apex_MaxD.csv"
)

patient = "LEU_BBB_21036"

patient_dir = ALL_PROCESSED_DIR / patient

lv_path = patient_dir / "lv_endo-processed.vtp"
rv_path = patient_dir / "rv_endo-processed.vtp"
epi_path = patient_dir / "epicardium-processed.vtp"


# ============================================================
# PARAMETERS
# ============================================================

SQUARE_SIDE_LENGTH = 2.0
SQUARE_FRACTION = 0.4

SQUARE_SPACING_MM = 6.0
MM_TO_UM = 1000.0

SHOW_LV = True
SHOW_RV = True
SHOW_EPI = True


# ============================================================
# FUNCTIONS
# ============================================================

def find_patient_column(df):
    possible_names = [
        "patient",
        "Patient",
        "PATIENT",
        "patient_id",
        "PatientID",
        "id",
        "ID",
    ]

    for name in possible_names:
        if name in df.columns:
            return name

    raise ValueError(
        "Could not find patient column in CSV. "
        f"Available columns are: {list(df.columns)}"
    )


def read_point_from_row(row, possible_column_sets, point_name):
    for cols in possible_column_sets:
        if all(c in row.index for c in cols):
            return np.array(
                [row[cols[0]], row[cols[1]], row[cols[2]]],
                dtype=float,
            )

    raise ValueError(
        f"Could not find columns for {point_name}. "
        f"Available columns are: {list(row.index)}"
    )


def make_oriented_square_with_point_on_diagonal(
    point,
    normal,
    side_length,
    fraction=0.5,
):
    """
    Crea un quadrato nel piano ortogonale a `normal`.

    `point` non è necessariamente il centro del quadrato:
    viene imposto che `point` stia lungo la diagonale corner0 -> corner2.

    fraction = 0.5  -> point al centro del quadrato
    fraction = 0.4  -> quadrato traslato lungo diagonale
    """

    point = np.asarray(point, dtype=float)
    normal = np.asarray(normal, dtype=float)

    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("Normal vector has zero norm.")

    normal = normal / normal_norm

    tmp = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    h = side_length / 2.0

    rel_corners = np.array([
        -h * u - h * v,
         h * u - h * v,
         h * u + h * v,
        -h * u + h * v,
    ])

    rel_point_on_diag = (
        (1.0 - fraction) * rel_corners[0]
        + fraction * rel_corners[2]
    )

    square_center = point - rel_point_on_diag

    corners = square_center + rel_corners
    faces = np.hstack([[4, 0, 1, 2, 3]])

    square = pv.PolyData(corners, faces)

    return square, square_center


def make_scaled_squares_until_apex(
    start_point,
    apex_point,
    axis,
    side_length,
    spacing,
    fraction=0.5,
    allow_one_beyond_apex=True,
    n_before_start=2,
):
    """
    Genera quadrati lungo l'asse apex-base.

    I quadrati partono anche prima di start_point, nella direzione opposta
    all'apex, e poi avanzano verso apex_point.

    n_before_start:
        numero di quadrati prima del centroide mitralico.
    """

    start_point = np.asarray(start_point, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    axis = np.asarray(axis, dtype=float)

    axis_norm = np.linalg.norm(axis)

    if axis_norm == 0:
        raise ValueError("Axis has zero norm.")

    axis = axis / axis_norm

    axis_length = np.linalg.norm(apex_point - start_point)

    if spacing <= 0:
        raise ValueError("Spacing must be positive.")

    n_full_steps = int(np.floor(axis_length / spacing))

    axis_points = []

    # punti prima del centroide, direzione opposta all'apex
    for i in range(n_before_start, 0, -1):
        axis_point = start_point - i * spacing * axis
        axis_points.append(axis_point)

    # punti dal centroide verso l'apex
    for i in range(n_full_steps + 1):
        axis_point = start_point + i * spacing * axis
        axis_points.append(axis_point)

    last_distance = n_full_steps * spacing

    if allow_one_beyond_apex:
        next_distance = last_distance + spacing

        if next_distance > axis_length:
            axis_point = start_point + next_distance * axis
            axis_points.append(axis_point)

    squares = []
    square_centers = []

    for axis_point in axis_points:
        square, square_center = make_oriented_square_with_point_on_diagonal(
            point=axis_point,
            normal=axis,
            side_length=side_length,
            fraction=fraction,
        )

        squares.append(square)
        square_centers.append(square_center)

    return squares, np.asarray(axis_points), np.asarray(square_centers)

# ============================================================
# LOAD MESHES
# ============================================================

lv = None
rv = None
epi = None

if SHOW_LV:
    lv = pv.read(lv_path)
    print("\nLV loaded")
    print(lv)

if SHOW_RV and rv_path.exists():
    rv = pv.read(rv_path)
    print("\nRV loaded")
    print(rv)

if SHOW_EPI and epi_path.exists():
    epi = pv.read(epi_path)
    print("\nEpi loaded")
    print(epi)

if epi is None:
    raise ValueError("Epicardium mesh is required to read scale-tooriginalrange.")


# ============================================================
# READ C_AREA AND A_MAXD FROM CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

patient_col = find_patient_column(df)

row_df = df[df[patient_col].astype(str) == str(patient)]

if row_df.empty:
    raise ValueError(
        f"Patient '{patient}' not found in CSV.\n"
        f"CSV path: {CSV_PATH}\n"
        f"Patient column: {patient_col}\n"
        f"Available patients example:\n{df[patient_col].head(20).to_list()}"
    )

row = row_df.iloc[0]

mitral_centroid = read_point_from_row(
    row,
    possible_column_sets=[
        ["C_area_x", "C_area_y", "C_area_z"],
        ["Carea_x", "Carea_y", "Carea_z"],
        ["mitral_centroid_x", "mitral_centroid_y", "mitral_centroid_z"],
        ["centroid_x", "centroid_y", "centroid_z"],
        ["C_x", "C_y", "C_z"],
    ],
    point_name="mitral centroid",
)

apex_point = read_point_from_row(
    row,
    possible_column_sets=[
        ["A_maxD_x", "A_maxD_y", "A_maxD_z"],
        ["apex_MaxD_x", "apex_MaxD_y", "apex_MaxD_z"],
        ["apex_maxD_x", "apex_maxD_y", "apex_maxD_z"],
        ["apex_x", "apex_y", "apex_z"],
        ["A_x", "A_y", "A_z"],
    ],
    point_name="apex MaxD",
)

axis = apex_point - mitral_centroid

axis_norm = np.linalg.norm(axis)
if axis_norm == 0:
    raise ValueError("Apex and mitral centroid are identical. Cannot define axis.")

axis = axis / axis_norm

print("\nRead from CSV")
print("Patient:", patient)
print("Mitral centroid C_area:", mitral_centroid)
print("Apex A_maxD:", apex_point)
print("Axis C_area -> A_maxD:", axis)
print("Distance C_area-apex normalized:", np.linalg.norm(apex_point - mitral_centroid))


# ============================================================
# CONVERT SQUARE SPACING FROM MM TO NORMALIZED COORDINATES
# ============================================================

scale_to_original_um = epi.field_data["scale-tooriginalrange"][0]
scale_to_original_mm = scale_to_original_um / 1000.0

SQUARE_SPACING = SQUARE_SPACING_MM / scale_to_original_mm

print("\nScale")
print("scale_to_original:", scale_to_original_um, "um")
print("scale_to_original:", scale_to_original_mm, "mm")
print("square spacing:", SQUARE_SPACING_MM, "mm")
print("square spacing normalized:", SQUARE_SPACING)


# ============================================================
# CREATE SQUARES UNTIL APEX
# ============================================================

squares, axis_points, square_centers = make_scaled_squares_until_apex(
    start_point=mitral_centroid,
    apex_point=apex_point,
    axis=axis,
    side_length=SQUARE_SIDE_LENGTH,
    spacing=SQUARE_SPACING,
    fraction=SQUARE_FRACTION,
    allow_one_beyond_apex=True,
    n_before_start=2,
)

print("\nSquares")
print("Number of squares:", len(squares))
print("Number of axis points:", len(axis_points))
print("Number of square centers:", len(square_centers))

axis_distances_norm = np.linalg.norm(axis_points - mitral_centroid, axis=1)
axis_distances_mm = axis_distances_norm * scale_to_original_mm

# print("\nSquare positions along axis [mm]:")
# for i, d in enumerate(axis_distances_mm):
#     flag = " beyond apex" if d > np.linalg.norm(apex_point - mitral_centroid) * scale_to_original_mm else ""
#     print(f"Square {i:02d}: {d:.3f} mm{flag}")


# ============================================================
# PLOT
# ============================================================

plotter = pv.Plotter(
    off_screen=False,
    notebook=False,
)

if lv is not None:
    plotter.add_mesh(
        lv,
        color="lightgray",
        opacity=0.48,
    )

if rv is not None:
    plotter.add_mesh(
        rv,
        color="lightblue",
        opacity=0.48,
    )

if epi is not None:
    plotter.add_mesh(
        epi,
        color="salmon",
        opacity=0.28,
    )

plotter.add_mesh(
    pv.Sphere(
        radius=0.02,
        center=mitral_centroid,
    ),
    color="magenta",
)

plotter.add_mesh(
    pv.Sphere(
        radius=0.015,
        center=apex_point,
    ),
    color="yellow",
)

plotter.add_mesh(
    pv.Line(
        mitral_centroid,
        apex_point,
    ),
    color="yellow",
    line_width=7,
)

plotter.add_mesh(
    pv.PolyData(axis_points),
    color="black",
    point_size=10,
    render_points_as_spheres=True,
)

plotter.add_mesh(
    pv.PolyData(square_centers),
    color="magenta",
    point_size=8,
    render_points_as_spheres=True,
)

for square in squares:
    plotter.add_mesh(
        square,
        color="green",
        opacity=0.18,
        show_edges=True,
    )

plotter.add_point_labels(
    [mitral_centroid],
    ["C_area"],
    font_size=18,
)

plotter.add_point_labels(
    [apex_point],
    ["A_maxD"],
    font_size=18,
)

plotter.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)

plotter.add_axes()

plotter.show(
    interactive=True,
    auto_close=False,
)