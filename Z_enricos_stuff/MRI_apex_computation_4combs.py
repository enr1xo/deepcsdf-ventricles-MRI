"""
in questo codice proviamoa  leggere il csv e in base a quel combinazione vogliamo fare (centroid-apex) generiamo l'asse e il seguente piano ortogonale.
"""

"""
Legge il CSV con:
- C_area
- A_maxD
- A_PCA

In base alla combinazione scelta genera:
- asse apex-base: C_area -> apex
- piano ortogonale all'asse e passante per C_area
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv


# ============================================================
# PATHS
# ============================================================

CSV_PATH = Path(
    r"C:\Users\e.rizzardi\OneDrive\Desktop\mitral_Carea_and_apexes.csv"
)

ALL_PROCESSED_DIR = Path(
    r"C:\Users\e.rizzardi\OneDrive\Desktop\processed_patients"
)

patient = "LEU_NORM_0016"
patient = "AF001"
patient = "LEU_BBB_21350"


# ============================================================
# PARAMETERS
# ============================================================

APEX_METHOD = "maxD"
# APEX_METHOD = "PCA"

PLANE_SIZE = 2
frac = 0.4

# ============================================================
# FUNCTIONS
# ============================================================

def get_point_from_row(row, prefix):
    return np.array([
        row[f"{prefix}_x"],
        row[f"{prefix}_y"],
        row[f"{prefix}_z"],
    ], dtype=float)


def make_oriented_square(center, normal, side_length):
    """
    Creates a square plane centered at `center`,
    lying in the plane normal to `normal`.
    """

    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    h = side_length / 2.0

    corners = np.array([
        center - h * u - h * v,
        center + h * u - h * v,
        center + h * u + h * v,
        center - h * u + h * v,
    ])

    faces = np.hstack([[4, 0, 1, 2, 3]])

    return pv.PolyData(corners, faces)

def make_oriented_square_with_point_on_diagonal(point, normal, side_length, fraction=0.5):
    """
    Creates a square lying in the plane normal to `normal`.

    `point` lies on one diagonal of the square.
    fraction=0.75 means that `point` is at 3/4 of the diagonal,
    going from corner 0 to corner 2.
    """

    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    h = side_length / 2.0

    # corners relative to the square center
    rel_corners = np.array([
        -h * u - h * v,
         h * u - h * v,
         h * u + h * v,
        -h * u + h * v,
    ])

    # point is at fraction along diagonal corner 0 -> corner 2
    rel_point_on_diag = (1.0 - fraction) * rel_corners[0] + fraction * rel_corners[2]

    # shift square center so that `point` is exactly there
    square_center = point - rel_point_on_diag

    corners = square_center + rel_corners

    faces = np.hstack([[4, 0, 1, 2, 3]])

    return pv.PolyData(corners, faces)


def get_axis_from_csv(csv_path, patient, apex_method):
    """
    Reads C_area and selected apex from CSV.
    apex_method can be:
    - "maxD"
    - "PCA"
    """

    df = pd.read_csv(csv_path)

    row_df = df[df["patient"] == patient]

    if len(row_df) == 0:
        raise ValueError(f"Patient {patient} not found in CSV")

    if len(row_df) > 1:
        raise ValueError(f"Patient {patient} appears multiple times in CSV")

    row = row_df.iloc[0]

    C_area = get_point_from_row(row, "C_area")

    if apex_method == "maxD":
        apex = get_point_from_row(row, "A_maxD")
    elif apex_method == "PCA":
        apex = get_point_from_row(row, "A_PCA")
    else:
        raise ValueError(
            "apex_method must be 'maxD' or 'PCA'"
        )

    axis = apex - C_area
    axis /= np.linalg.norm(axis)

    return C_area, apex, axis


# ============================================================
# LOAD AXIS FROM CSV
# ============================================================

C_area, apex, axis = get_axis_from_csv(
    CSV_PATH,
    patient,
    APEX_METHOD
)

print("\nPatient:", patient)
print("Apex method:", APEX_METHOD)
print("C_area:", C_area)
print("Apex:", apex)
print("Axis C_area -> Apex:", axis)


# ============================================================
# BUILD ORTHOGONAL PLANE
# ============================================================

plane = make_oriented_square(
    center=C_area,
    normal=axis,
    side_length=PLANE_SIZE
)

plane = make_oriented_square_with_point_on_diagonal(
    point=C_area,
    normal=axis,
    side_length=PLANE_SIZE,
    fraction=frac 
)

# ============================================================
# OPTIONAL: LOAD MESH FOR VISUALIZATION
# ============================================================

lv_path = ALL_PROCESSED_DIR / patient / "lv_endo-processed.vtp"
rv_path = ALL_PROCESSED_DIR / patient / "rv_endo-processed.vtp"
epi_path = ALL_PROCESSED_DIR / patient / "epicardium-processed.vtp"

lv = pv.read(lv_path)
rv = pv.read(rv_path) if rv_path.exists() else None
epi = pv.read(epi_path) if epi_path.exists() else None

# ============================================================
# PLOT
# ============================================================

plotter = pv.Plotter()

plotter.add_mesh(
    lv,
    color="lightgray",
    opacity=0.45,
)

if rv is not None:
    plotter.add_mesh(
        rv,
        color="lightblue",
        opacity=0.45,
    )

if epi is not None:
    plotter.add_mesh(
        epi,
        color="salmon",
        opacity=0.25,
    )

plotter.add_mesh(
    pv.Sphere(radius=0.02, center=C_area),
    color="magenta"
)

plotter.add_mesh(
    pv.Sphere(radius=0.017, center=apex),
    color="yellow" if APEX_METHOD == "maxD" else "lime"
)

plotter.add_mesh(
    pv.Line(C_area, apex),
    color="yellow" if APEX_METHOD == "maxD" else "lime",
    line_width=6
)

plotter.add_mesh(
    plane,
    color="orange",
    opacity=0.35,
    show_edges=True
)

plotter.add_point_labels(
    [C_area],
    ["C_area"],
    font_size=18
)

plotter.add_point_labels(
    [apex],
    [f"Apex {APEX_METHOD}"],
    font_size=18
)

plotter.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)

plotter.add_axes()
plotter.show()