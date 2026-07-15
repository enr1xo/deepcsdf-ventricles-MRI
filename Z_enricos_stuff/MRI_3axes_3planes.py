"""
in questo codice leggiamo il csv in cui sono riportate le posizione di:
    centroide della mitrale
    apex
    centroide della tricuspide

a partire da queste tre posizioni, costruiamo i 3assi e i 3 piani di acquisizione.
Gli assi sono tutti originati nel centroide della mitrale.
    1. apex-base axis: asse che va dalla mitrale all'apex
    2. centroid-centroid axis: asse che va dal centroide della mitrale a quello della tricuspide
    3. third axis: il terzo asse normale aglia ltri due per avere un sistema ortonormale destrorso

Definiti gli assi, possiamo definire i piani:
    A. (short axis) piano normale all'asse apex-base contentente il centroide della mitrale
    B. (vertical long axis) piano normale all'asse centroid-centroid contente il centoride della mitrale (quello che ci da la 2 chambers view))
    C. (horizontal long axis) piano normale agli altri due (quello che ci da la 4 chambers view)

Per ogni piano costuriamo un quadrato di lato 2 centrato nella centoide della mitrale.
Plottiamo i 3 piani e i 3 assi di colore diverso.
"""


from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv


# ============================================================
# PATHS
# ============================================================

CSV_PATH = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/mitral_apex_tricuspid_locations.csv"
)

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

patient = "AF001"


# ============================================================
# PARAMETRI
# ============================================================

SQUARE_SIDE = 3.0
AXIS_LENGTH = 2.0


SLICE_THICKNESS_MM = 0.75


# ============================================================
# FUNZIONI
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError("Cannot normalize zero vector")

    return v / norm


def make_square(center, u, v, side):
    """
    Crea un quadrato centrato in center, con lati lungo u e v.
    """

    h = side / 2.0

    p0 = center - h * u - h * v
    p1 = center + h * u - h * v
    p2 = center + h * u + h * v
    p3 = center - h * u + h * v

    points = np.array([
        p0,
        p1,
        p2,
        p3,
    ])

    faces = np.hstack([
        [4, 0, 1, 2, 3]
    ])

    return pv.PolyData(points, faces)


def make_axis_line(center, direction, length):
    p0 = center - length * direction
    p1 = center + length * direction
    return pv.Line(p0, p1)


# ============================================================
# READ CSV
# ============================================================

df = pd.read_csv(
    CSV_PATH,
    sep=";"
)

row = df[df["patient_id"] == patient]

if len(row) == 0:
    raise ValueError(f"Patient {patient} not found in CSV")

row = row.iloc[0]


# ============================================================
# READ POINTS
# ============================================================

C_area = np.array([
    row["C_area_x"],
    row["C_area_y"],
    row["C_area_z"],
], dtype=float)

A_maxD = np.array([
    row["A_maxD_x"],
    row["A_maxD_y"],
    row["A_maxD_z"],
], dtype=float)

T_area = np.array([
    row["T_area_x"],
    row["T_area_y"],
    row["T_area_z"],
], dtype=float)


# ============================================================
# COSTRUISCI ASSI
# ============================================================

# 1. asse apex-base: mitrale -> apex
e1 = normalize(
    A_maxD - C_area
)

# 2. direzione mitrale -> tricuspide
raw_e2 = normalize(
    T_area - C_area
)

# Rimuoviamo da raw_e2 la componente parallela a e1.
# In questo modo e2 è davvero ortogonale a e1.
e2 = raw_e2 - np.dot(raw_e2, e1) * e1
e2 = normalize(e2)

# 3. terzo asse destrorso
e3 = np.cross(e1, e2)
e3 = normalize(e3)

# Ricostruiamo e2 per sicurezza così la base è perfettamente destrorsa
e2 = np.cross(e3, e1)
e2 = normalize(e2)

print("\nPoints")
print("C_area:", C_area)
print("A_maxD:", A_maxD)
print("T_area:", T_area)

print("\nAxes")
print("e1 apex-base:", e1)
print("e2 mitral-tricuspid orthogonalized:", e2)
print("e3 third axis:", e3)

print("\nDot products")
print("e1 · e2:", np.dot(e1, e2))
print("e1 · e3:", np.dot(e1, e3))
print("e2 · e3:", np.dot(e2, e3))


# ============================================================
# COSTRUISCI QUADRATI / PIANI
# ============================================================

# A. Short-axis: normale e1, quindi piano generato da e2, e3
square_short_axis = make_square(
    C_area,
    e2,
    e3,
    SQUARE_SIDE
)

# B. Piano normale a e2, generato da e1, e3
square_normal_e2 = make_square(
    C_area,
    e1,
    e3,
    SQUARE_SIDE
)

# C. Piano normale a e3, generato da e1, e2
square_normal_e3 = make_square(
    C_area,
    e1,
    e2,
    SQUARE_SIDE
)


# ============================================================
# LOAD MESHES OPTIONAL
# ============================================================

patient_dir = ALL_PROCESSED_DIR / patient

lv_path = patient_dir / "lv_endo-processed.vtp"
rv_path = patient_dir / "rv_endo-processed.vtp"
epi_path = patient_dir / "epicardium-processed.vtp"

lv = pv.read(lv_path) if lv_path.exists() else None
rv = pv.read(rv_path) if rv_path.exists() else None
epi = pv.read(epi_path) if epi_path.exists() else None

if epi is None:
    raise RuntimeError("Epicardium mesh not found")

scale_to_original_range = epi.field_data["scale-tooriginalrange"][0]
scale_to_original_mm = scale_to_original_range / 1000.0

slice_thickness_norm = SLICE_THICKNESS_MM / scale_to_original_mm
slice_half_thickness_norm = slice_thickness_norm / 2.0

print("\nScale")
print("scale_to_original_range:", scale_to_original_range)
print("scale_to_original_mm:", scale_to_original_mm)
print("slice_thickness_mm:", SLICE_THICKNESS_MM)
print("slice_thickness_norm:", slice_thickness_norm)
print("slice_half_thickness_norm:", slice_half_thickness_norm)

epi_points = epi.points

# distanza signed dai tre piani
# ogni piano passa per C_area
# normale piano short-axis = e1
# normale piano normal-e2 = e2
# normale piano normal-e3 = e3

d_short = np.dot(
    epi_points - C_area,
    e1
)

d_e2 = np.dot(
    epi_points - C_area,
    e2
)

d_e3 = np.dot(
    epi_points - C_area,
    e3
)

mask_short = (
    np.abs(d_short) <= slice_half_thickness_norm
)

mask_e2 = (
    np.abs(d_e2) <= slice_half_thickness_norm
)

mask_e3 = (
    np.abs(d_e3) <= slice_half_thickness_norm
)

epi_points_short = epi_points[mask_short]
epi_points_e2 = epi_points[mask_e2]
epi_points_e3 = epi_points[mask_e3]

print("\nEpicardium points inside slice")
print("short-axis slice:", epi_points_short.shape[0])
print("normal-e2 slice:", epi_points_e2.shape[0])
print("normal-e3 slice:", epi_points_e3.shape[0])

# ============================================================
# PLOT
# ============================================================

plotter = pv.Plotter()

if epi is not None:
    plotter.add_mesh(
        epi,
        color="lightgray",
        opacity=0.15,
    )

if lv is not None:
    plotter.add_mesh(
        lv,
        color="red",
        opacity=0.20,
    )

if rv is not None:
    plotter.add_mesh(
        rv,
        color="blue",
        opacity=0.20,
    )

# punti
plotter.add_mesh(
    pv.Sphere(radius=0.035, center=C_area),
    color="magenta",
    label="Mitral centroid C_area"
)

plotter.add_mesh(
    pv.Sphere(radius=0.035, center=A_maxD),
    color="black",
    label="Apex A_maxD"
)

plotter.add_mesh(
    pv.Sphere(radius=0.035, center=T_area),
    color="cyan",
    label="Tricuspid centroid T_area"
)

# assi
axis_e1 = make_axis_line(C_area, e1, AXIS_LENGTH)
axis_e2 = make_axis_line(C_area, e2, AXIS_LENGTH)
axis_e3 = make_axis_line(C_area, e3, AXIS_LENGTH)

plotter.add_mesh(
    axis_e1,
    color="green",
    line_width=5,
    label="e1 apex-base"
)

plotter.add_mesh(
    axis_e2,
    color="orange",
    line_width=5,
    label="e2 mitral-tricuspid"
)

plotter.add_mesh(
    axis_e3,
    color="purple",
    line_width=5,
    label="e3 third axis"
)

# frecce
plotter.add_mesh(
    pv.Arrow(start=C_area, direction=e1, scale=0.35),
    color="green"
)

plotter.add_mesh(
    pv.Arrow(start=C_area, direction=e2, scale=0.35),
    color="orange"
)

plotter.add_mesh(
    pv.Arrow(start=C_area, direction=e3, scale=0.35),
    color="purple"
)

# piani
plotter.add_mesh(
    square_short_axis,
    color="green",
    opacity=0.30,
    show_edges=True,
    label="Short-axis plane normal e1"
)

plotter.add_mesh(
    square_normal_e2,
    color="orange",
    opacity=0.30,
    show_edges=True,
    label="Vertical Long-axis Plane normal e2"
)

plotter.add_mesh(
    square_normal_e3,
    color="purple",
    opacity=0.30,
    show_edges=True,
    label="Horizonral Long-axis Plane normal e3"
)

plotter.add_points(
    epi_points_short,
    color="green",
    point_size=8,
    render_points_as_spheres=True,
    label="Epi points short-axis slice"
)

plotter.add_points(
    epi_points_e2,
    color="orange",
    point_size=8,
    render_points_as_spheres=True,
    label="Epi points normal-e2 slice"
)

plotter.add_points(
    epi_points_e3,
    color="purple",
    point_size=8,
    render_points_as_spheres=True,
    label="Epi points normal-e3 slice"
)

plotter.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)

plotter.add_axes()
plotter.add_legend()

plotter.show()