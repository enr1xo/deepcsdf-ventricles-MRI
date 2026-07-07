"""
Generate MRI-like square samples for DeepSDF training.

For each patient:
- read mitral centroid C_area and apex A_maxD from CSV
- build apex-base axis
- generate oriented squares normal to the axis
- sample points inside each square with minimum distance 1 mm
- compute L2 distance in local slab to each surface
- compute sign using compute_signed_distance_libigl
- save x,y,z,sdf_epi,sdf_lv,sdf_rv

Also plots one example patient.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.surface_utils import compute_signed_distance_libigl

ALL_PROCESSED_DIR = Path(
    r"C:\Users\e.rizzardi\OneDrive\Desktop\processed_patients"
)

CSV_PATH = Path(
    r"C:\Users\e.rizzardi\OneDrive\Desktop\graz_June\mitral_Carea_and_apex_MaxD.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\e.rizzardi\OneDrive\Desktop\graz_June\square_samples_min_dist_1mm"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLE_PATIENT = "AF001"
PLOT_EXAMPLE = True

# ============================================================
# PARAMETERS
# ============================================================

SQUARE_SIDE_LENGTH = 2.0          # normalized units
SQUARE_DIAGONAL_FRACTION = 0.4   # mitral point position on first square diagonal

MIN_DIST_MM = 1.0                # minimum distance between samples in each square

SQUARE_SPACING_MM = 5.0          # distance between square centers along apex-base axis
SLAB_WIDTH_MM = 1.0              # slab thickness used for L2 distance search

ALLOW_ONE_SQUARE_BEYOND_APEX = True

SAVE_AS_NPY = True
SAVE_AS_CSV = False

N_POINTS_PER_SQUARE = 100

# ============================================================
# CSV HELPERS
# ============================================================

def find_patient_column(df):
    possible_names = [
        "patient", "Patient", "PATIENT",
        "patient_id", "PatientID",
        "id", "ID",
    ]

    for name in possible_names:
        if name in df.columns:
            return name

    raise ValueError(
        "Could not find patient column. "
        f"Available columns: {list(df.columns)}"
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
        f"Available columns: {list(row.index)}"
    )


def read_patient_points(df, patient):
    patient_col = find_patient_column(df)

    row_df = df[df[patient_col].astype(str) == str(patient)]

    if row_df.empty:
        raise ValueError(
            f"Patient '{patient}' not found in CSV."
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
            ["x", "y", "z"],
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

    return mitral_centroid, apex_point


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def get_square_frame(normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    tmp = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    return u, v


def make_square_with_point_on_diagonal(point, normal, side_length, fraction):
    point = np.asarray(point, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    u, v = get_square_frame(normal)

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


def sample_grid_in_square(center, normal, side_length, min_dist):
    """
    Regular grid in the square.
    Adjacent points are min_dist apart.
    Therefore the minimum distance condition is guaranteed.
    """

    u, v = get_square_frame(normal)

    half = side_length / 2.0

    coords = np.arange(
        -half,
        half + 0.5 * min_dist,
        min_dist,
    )

    aa, bb = np.meshgrid(coords, coords, indexing="ij")

    points = (
        center[None, None, :]
        + aa[:, :, None] * u[None, None, :]
        + bb[:, :, None] * v[None, None, :]
    )

    return points.reshape(-1, 3)

def sample_points_in_square_min_dist_fast(
    square_center,
    normal,
    side_length,
    n_points,
    min_dist,
    seed=42,
    max_trials=500_000,
):
    rng = np.random.default_rng(seed)

    u, v = get_square_frame(normal)

    half = side_length / 2.0

    points_2d = np.empty((n_points, 2), dtype=float)
    points_3d = np.empty((n_points, 3), dtype=float)

    n_accepted = 0
    trials = 0

    min_dist2 = min_dist**2

    while n_accepted < n_points and trials < max_trials:

        a = rng.uniform(-half, half)
        b = rng.uniform(-half, half)

        if n_accepted == 0:
            accept = True
        else:
            diff = points_2d[:n_accepted] - np.array([a, b])
            dist2 = np.sum(diff**2, axis=1)

            accept = np.all(dist2 >= min_dist2)

        if accept:
            points_2d[n_accepted] = [a, b]
            points_3d[n_accepted] = square_center + a * u + b * v
            n_accepted += 1

        trials += 1

    if n_accepted < n_points:
        raise RuntimeError(
            f"Generated only {n_accepted}/{n_points} points after "
            f"{max_trials} trials. Try reducing n_points, reducing min_dist, "
            f"or increasing square side length."
        )

    return points_3d

def generate_square_centers(
    mitral_centroid,
    apex_point,
    first_square_center,
    square_spacing,
    allow_one_beyond=True,
):
    axis = apex_point - mitral_centroid
    axis_len = np.linalg.norm(axis)

    if axis_len == 0:
        raise ValueError("Apex and mitral centroid coincide.")

    axis /= axis_len

    n_steps = int(np.floor(axis_len / square_spacing)) + 1

    if allow_one_beyond:
        n_steps += 1

    centers = np.array([
        first_square_center + i * square_spacing * axis
        for i in range(n_steps)
    ])

    return centers, axis


# ============================================================
# L2 DISTANCE IN LOCAL SLAB
# ============================================================

def l2_distance_to_surface_in_slab(
    query_points,
    surface_points,
    axis_origin,
    axis,
    slab_half_width,
):
    """
    For each query point:
    - take surface vertices whose projection along axis is inside the same slab
    - compute minimum Euclidean distance to those vertices

    Returns normalized distances.
    """

    query_q = np.dot(query_points - axis_origin, axis)
    surf_q = np.dot(surface_points - axis_origin, axis)

    distances = np.full(len(query_points), np.nan, dtype=float)

    for i, p in enumerate(query_points):
        q = query_q[i]

        mask = np.abs(surf_q - q) <= slab_half_width
        slab_pts = surface_points[mask]

        if len(slab_pts) == 0:
            continue

        diff = slab_pts - p[None, :]
        d = np.linalg.norm(diff, axis=1)

        distances[i] = np.min(d)

    return distances


def signed_slab_sdf(
    query_points,
    surface,
    axis_origin,
    axis,
    slab_half_width,
):
    """
    SDF = sign from libigl signed distance * L2 distance in local slab.
    """

    unsigned_l2 = l2_distance_to_surface_in_slab(
        query_points=query_points,
        surface_points=surface.points,
        axis_origin=axis_origin,
        axis=axis,
        slab_half_width=slab_half_width,
    )

    classical_sdf = compute_signed_distance_libigl(
        surface,
        query_points,
    )

    sign = np.sign(classical_sdf)

    signed_dist = unsigned_l2 * sign

    return signed_dist


# ============================================================
# PROCESS SINGLE PATIENT
# ============================================================

def process_patient(patient, df):
    print("\n========================================")
    print("Processing patient:", patient)
    print("========================================")

    patient_dir = ALL_PROCESSED_DIR / patient

    epi_path = patient_dir / "epicardium-processed.vtp"
    lv_path = patient_dir / "lv_endo-processed.vtp"
    rv_path = patient_dir / "rv_endo-processed.vtp"

    if not epi_path.exists():
        raise FileNotFoundError(epi_path)

    if not lv_path.exists():
        raise FileNotFoundError(lv_path)

    if not rv_path.exists():
        raise FileNotFoundError(rv_path)

    epi = pv.read(epi_path)
    lv = pv.read(lv_path)
    rv = pv.read(rv_path)

    mitral_centroid, apex_point = read_patient_points(
        df,
        patient,
    )

    axis_raw = apex_point - mitral_centroid
    axis_len = np.linalg.norm(axis_raw)

    if axis_len == 0:
        raise ValueError(
            f"{patient}: apex and mitral centroid are identical."
        )

    axis = axis_raw / axis_len

    scale_to_original_um = epi.field_data["scale-tooriginalrange"][0]
    scale_to_original_mm = scale_to_original_um / 1000.0

    min_dist_norm = MIN_DIST_MM / scale_to_original_mm
    square_spacing_norm = SQUARE_SPACING_MM / scale_to_original_mm
    slab_half_width_norm = (SLAB_WIDTH_MM / 2.0) / scale_to_original_mm

    print("scale_to_original_mm:", scale_to_original_mm)
    print("min_dist_norm:", min_dist_norm)
    print("square_spacing_norm:", square_spacing_norm)
    print("slab_half_width_norm:", slab_half_width_norm)

    first_square, first_square_center = make_square_with_point_on_diagonal(
        point=mitral_centroid,
        normal=axis,
        side_length=SQUARE_SIDE_LENGTH,
        fraction=SQUARE_DIAGONAL_FRACTION,
    )

    square_centers, axis = generate_square_centers(
        mitral_centroid=mitral_centroid,
        apex_point=apex_point,
        first_square_center=first_square_center,
        square_spacing=square_spacing_norm,
        allow_one_beyond=ALLOW_ONE_SQUARE_BEYOND_APEX,
    )

    

    all_points = []

    for i, center in enumerate(square_centers):

        pts = sample_points_in_square_min_dist_fast(
            square_center=center,
            normal=axis,
            side_length=SQUARE_SIDE_LENGTH,
            n_points=N_POINTS_PER_SQUARE,
            min_dist=min_dist_norm,
            seed=42 + i,
        )

        all_points.append(pts)

    points = np.vstack(all_points)

    print("Number of squares:", len(square_centers))
    print("Samples per square:", len(all_points[0]))
    print("Total samples:", len(points))

    sdf_epi = signed_slab_sdf(
        query_points=points,
        surface=epi,
        axis_origin=mitral_centroid,
        axis=axis,
        slab_half_width=slab_half_width_norm,
    )

    sdf_lv = signed_slab_sdf(
        query_points=points,
        surface=lv,
        axis_origin=mitral_centroid,
        axis=axis,
        slab_half_width=slab_half_width_norm,
    )

    sdf_rv = signed_slab_sdf(
        query_points=points,
        surface=rv,
        axis_origin=mitral_centroid,
        axis=axis,
        slab_half_width=slab_half_width_norm,
    )

    samples = np.column_stack([
        points,
        sdf_epi,
        sdf_lv,
        sdf_rv,
    ])

    # valid_mask = np.all(np.isfinite(samples), axis=1)
    # samples = samples[valid_mask]

    print("Valid samples:", len(samples))

    out_npy = OUTPUT_DIR / f"{patient}_square_samples.npy"
    out_csv = OUTPUT_DIR / f"{patient}_square_samples.csv"

    if SAVE_AS_NPY:
        np.save(out_npy, samples)
        print("Saved:", out_npy)

    if SAVE_AS_CSV:
        out_df = pd.DataFrame(
            samples,
            columns=[
                "x", "y", "z",
                "sdf_epi", "sdf_lv", "sdf_rv",
            ],
        )
        out_df.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

    debug = {
        "patient": patient,
        "epi": epi,
        "lv": lv,
        "rv": rv,
        "mitral_centroid": mitral_centroid,
        "apex_point": apex_point,
        "axis": axis,
        "square_centers": square_centers,
        "samples": samples,
        "scale_to_original_mm": scale_to_original_mm,
    }

    return debug


# ============================================================
# PLOT EXAMPLE PATIENT
# ============================================================

def plot_patient_result(debug):
    patient = debug["patient"]
    epi = debug["epi"]
    lv = debug["lv"]
    rv = debug["rv"]
    mitral_centroid = debug["mitral_centroid"]
    apex_point = debug["apex_point"]
    axis = debug["axis"]
    square_centers = debug["square_centers"]
    samples = debug["samples"]

    points = samples[:, :3]
    sdf_epi = samples[:, 3]

    plotter = pv.Plotter(
        window_size=(1800, 1400),
        notebook=False,
        off_screen=False,
    )

    plotter.add_mesh(
        epi,
        color="lightgray",
        opacity=0.18,
    )

    plotter.add_mesh(
        lv,
        color="blue",
        opacity=0.12,
    )

    plotter.add_mesh(
        rv,
        color="green",
        opacity=0.12,
    )

    plotter.add_mesh(
        pv.PolyData(points),
        scalars=sdf_epi,
        cmap="bwr",
        point_size=5,
        render_points_as_spheres=True,
        show_scalar_bar=True,
    )

    plotter.add_mesh(
        pv.PolyData(square_centers),
        color="magenta",
        point_size=14,
        render_points_as_spheres=True,
    )

    plotter.add_mesh(
        pv.Sphere(radius=0.02, center=mitral_centroid),
        color="yellow",
    )

    plotter.add_mesh(
        pv.Sphere(radius=0.02, center=apex_point),
        color="red",
    )

    plotter.add_mesh(
        pv.Line(mitral_centroid, apex_point),
        color="black",
        line_width=6,
    )

    for c in square_centers:
        square, _ = make_square_with_point_on_diagonal(
            point=c,
            normal=axis,
            side_length=SQUARE_SIDE_LENGTH,
            fraction=0.5,
        )

        plotter.add_mesh(
            square,
            color="yellow",
            opacity=0.10,
            show_edges=True,
        )

    plotter.add_text(
        f"{patient}\n"
        f"Points colored by signed slab distance to epicardium",
        font_size=12,
    )

    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
    )

    plotter.add_axes()
    plotter.show(interactive=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # pv.OFF_SCREEN = False
    # pv.set_plot_theme("document")
    # pv.global_theme.jupyter_backend = "none"

    pv.OFF_SCREEN = False
    pv.set_plot_theme("document")

    df = pd.read_csv(CSV_PATH)

    patient_col = find_patient_column(df)

    patients = df[patient_col].astype(str).to_list()

    # --------------------------------------------------------
    # Process all patients
    # --------------------------------------------------------

    example_debug = None

    for patient in patients:
        if patient == EXAMPLE_PATIENT:
            try:
                debug = process_patient(patient, df)

                if patient == EXAMPLE_PATIENT:
                    example_debug = debug

            except Exception as e:
                print(f"\nSkipping {patient} because of error:")
                print(e)
        else:
            break
    # --------------------------------------------------------
    # Plot example patient
    # --------------------------------------------------------

    if PLOT_EXAMPLE:
        if example_debug is None:
            example_debug = process_patient(EXAMPLE_PATIENT, df)

        plot_patient_result(example_debug)