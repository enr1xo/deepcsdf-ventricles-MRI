import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyvista as pv
import igl


# ============================================================
# PARAMS
# ============================================================

@dataclass
class MRIDatasetParams:
    square_fraction: float = 0.4
    square_spacing_mm: float = 6.0
    slab_width_mm: float = 0.75
    min_dist_mm: float = 1.0
    n_points_per_square: int = 1000
    n_before_start: int = 4 
    n_after_apex: int = 3
    reference_slice_index: int = 4
    margin_factor: float = 1.25
    save_npy: bool = True
    save_csv: bool = False
    plot_debug: bool = False


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


def read_patient_points(df, patient):
    patient_col = find_patient_column(df)
    row_df = df[df[patient_col].astype(str) == str(patient)]

    if row_df.empty:
        raise ValueError(f"Patient '{patient}' not found in CSV.")

    row = row_df.iloc[0]

    c_area = read_point_from_row(
        row,
        possible_column_sets=[
            ["C_area_x", "C_area_y", "C_area_z"],
        ],
        point_name="C_area",
    )

    apex = read_point_from_row(
        row,
        possible_column_sets=[
            ["A_maxD_x", "A_maxD_y", "A_maxD_z"],
        ],
        point_name="A_maxD",
    )

    return c_area, apex


# ============================================================
# GEOMETRY
# ============================================================

def normalize(v, name="vector"):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n == 0:
        raise ValueError(f"{name} has zero norm.")

    return v / n


def get_square_frame(normal):
    normal = normalize(normal, "normal")

    tmp = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u = normalize(u, "u")

    v = np.cross(normal, u)
    v = normalize(v, "v")

    return u, v


def make_oriented_square_with_point_on_diagonal(
    point,
    normal,
    side_length,
    fraction=0.5,
):
    point = np.asarray(point, dtype=float)
    normal = normalize(normal, "normal")

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


def make_scaled_squares_until_apex(
    start_point,
    apex_point,
    axis,
    side_length,
    spacing,
    fraction=0.5,
    n_before_start=2,
    n_after_apex=2,
):
    start_point = np.asarray(start_point, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    axis = normalize(axis, "axis")

    axis_length = np.linalg.norm(apex_point - start_point)

    if spacing <= 0:
        raise ValueError("Spacing must be positive.")

    n_full_steps = int(np.floor(axis_length / spacing))

    axis_points = []

    for i in range(n_before_start, 0, -1):
        axis_points.append(start_point - i * spacing * axis)

    for i in range(n_full_steps + 1):
        axis_points.append(start_point + i * spacing * axis)

    last_distance = n_full_steps * spacing

    for i in range(1, n_after_apex + 1):
        next_distance = last_distance + i * spacing
        axis_points.append(start_point + next_distance * axis)

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


def estimate_square_side_from_epi_on_slice(
    epi,
    slice_point,
    axis,
    slab_half_width,
    margin_factor=1.25,
):
    axis = normalize(axis, "axis")
    u, v = get_square_frame(axis)

    epi_points = epi.points

    q = np.dot(epi_points - slice_point, axis)

    slab_mask = np.abs(q) <= slab_half_width
    epi_slice_points = epi_points[slab_mask]

    if len(epi_slice_points) == 0:
        raise ValueError(
            "No epicardial points found in the selected slice slab. "
            "Try increasing slab width."
        )

    local = epi_slice_points - slice_point

    coord_u = np.dot(local, u)
    coord_v = np.dot(local, v)

    range_u = coord_u.max() - coord_u.min()
    range_v = coord_v.max() - coord_v.min()

    side_length = margin_factor * max(abs(range_u), abs(range_v))

    return side_length, epi_slice_points


# ============================================================
# SAMPLING
# ============================================================

def estimate_max_points_in_square(side_length, min_dist):
    return int((2.0 / np.sqrt(3.0)) * side_length**2 / min_dist**2)


def sample_points_in_square_min_dist(
    square_center,
    normal,
    side_length,
    n_points,
    min_dist,
    seed=42,
    max_trials=1_000_000,
):
    rng = np.random.default_rng(seed)

    u, v = get_square_frame(normal)
    half = side_length / 2.0

    points_2d = []
    points_3d = []

    n_max_estimated = estimate_max_points_in_square(
        side_length=side_length,
        min_dist=min_dist,
    )

    if n_points > n_max_estimated:
        raise ValueError(
            f"Requested {n_points} points, but only about "
            f"{n_max_estimated} can fit with min_dist={min_dist:.6f}."
        )

    trials = 0

    while len(points_3d) < n_points and trials < max_trials:
        a = rng.uniform(-half, half)
        b = rng.uniform(-half, half)

        candidate_2d = np.array([a, b])

        if len(points_2d) == 0:
            accept = True
        else:
            existing_2d = np.asarray(points_2d)
            dists = np.linalg.norm(existing_2d - candidate_2d, axis=1)
            accept = np.all(dists >= min_dist)

        if accept:
            candidate_3d = square_center + a * u + b * v
            points_2d.append(candidate_2d)
            points_3d.append(candidate_3d)

        trials += 1

    if len(points_3d) < n_points:
        raise RuntimeError(
            f"Sampling failed: generated only {len(points_3d)} / {n_points} "
            f"after {max_trials} trials."
        )

    return np.asarray(points_3d)


def sample_all_squares(
    square_centers,
    axis,
    side_length,
    n_points_per_square,
    min_dist,
    seed0=42,
):
    all_points = []

    for i, square_center in enumerate(square_centers):
        pts = sample_points_in_square_min_dist(
            square_center=square_center,
            normal=axis,
            side_length=side_length,
            n_points=n_points_per_square,
            min_dist=min_dist,
            seed=seed0 + i,
        )

        all_points.append(pts)

    return np.vstack(all_points)


# ============================================================
# SDF
# ============================================================

def compute_sign_libigl(mesh, query_points):
    """
    Returns only the SDF sign:
    +1 outside
    -1 inside
    """

    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)

    w = igl.fast_winding_number(
        V=vertices,
        F=faces,
        Q=query_points.astype(np.float64),
    )

    sign = np.sign(0.5 - np.abs(w))

    bbox_max = mesh.bounds[1::2]

    outside_point = np.array([[
        bbox_max[0] + 100.0,
        bbox_max[1] + 100.0,
        bbox_max[2] + 100.0,
    ]])

    w_out = igl.fast_winding_number(
        V=vertices,
        F=faces,
        Q=outside_point.astype(np.float64),
    )[0]

    outside_sign = np.sign(0.5 - np.abs(w_out))

    if outside_sign < 0:
        sign *= -1

    return sign


def l2_distance_to_surface_in_slab(
    query_points,
    surface_points,
    axis_origin,
    axis,
    slab_half_width,
):
    axis = normalize(axis, "axis")

    query_q = np.dot(query_points - axis_origin, axis)
    surf_q = np.dot(surface_points - axis_origin, axis)

    distances = np.full(len(query_points), np.nan, dtype=float)

    for i, p in enumerate(query_points):
        q = query_q[i]

        slab_mask = np.abs(surf_q - q) <= slab_half_width
        slab_points = surface_points[slab_mask]

        if len(slab_points) == 0:
            continue

        dists = np.linalg.norm(slab_points - p[None, :], axis=1)
        distances[i] = np.min(dists)

    return distances


def signed_slab_sdf(
    query_points,
    surface,
    axis_origin,
    axis,
    slab_half_width,
):
    unsigned_distance = l2_distance_to_surface_in_slab(
        query_points=query_points,
        surface_points=surface.points,
        axis_origin=axis_origin,
        axis=axis,
        slab_half_width=slab_half_width,
    )

    sign = compute_sign_libigl(
        mesh=surface,
        query_points=query_points,
    )

    return unsigned_distance * sign


# ============================================================
# MAIN PATIENT FUNCTION
# ============================================================

def generate_patient_mri_dataset(
    patient,
    all_processed_dir,
    csv_path,
    output_dir,
    params=MRIDatasetParams(),
):
    all_processed_dir = Path(all_processed_dir)
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_dir = all_processed_dir / patient

    lv_path = patient_dir / "lv_endo-processed.vtp"
    rv_path = patient_dir / "rv_endo-processed.vtp"
    epi_path = patient_dir / "epicardium-processed.vtp"

    if not lv_path.exists():
        raise FileNotFoundError(lv_path)
    if not rv_path.exists():
        raise FileNotFoundError(rv_path)
    if not epi_path.exists():
        raise FileNotFoundError(epi_path)

    lv = pv.read(lv_path)
    rv = pv.read(rv_path)
    epi = pv.read(epi_path)

    df = pd.read_csv(csv_path)

    c_area, apex = read_patient_points(df, patient)

    axis = apex - c_area
    axis = normalize(axis, "axis")

    scale_to_original_um = epi.field_data["scale-tooriginalrange"][0]
    scale_to_original_mm = scale_to_original_um / 1000.0

    square_spacing = params.square_spacing_mm / scale_to_original_mm
    min_dist = params.min_dist_mm / scale_to_original_mm
    slab_half_width = (params.slab_width_mm / 2.0) / scale_to_original_mm

    reference_slice_point = (
        c_area
        + params.reference_slice_index * square_spacing * axis
    )

    square_side_length, epi_points_used_for_side = estimate_square_side_from_epi_on_slice(
        epi=epi,
        slice_point=reference_slice_point,
        axis=axis,
        slab_half_width=slab_half_width,
        margin_factor=params.margin_factor,
    )

    squares, axis_points, square_centers = make_scaled_squares_until_apex(
        start_point=c_area,
        apex_point=apex,
        axis=axis,
        side_length=square_side_length,
        spacing=square_spacing,
        fraction=params.square_fraction,
        n_before_start=params.n_before_start,
        n_after_apex=params.n_after_apex,
    )

    points = sample_all_squares(
        square_centers=square_centers,
        axis=axis,
        side_length=square_side_length,
        n_points_per_square=params.n_points_per_square,
        min_dist=min_dist,
    )

    sdf_epi_raw = signed_slab_sdf(
        query_points=points,
        surface=epi,
        axis_origin=c_area,
        axis=axis,
        slab_half_width=slab_half_width,
    )

    sdf_lv_raw = signed_slab_sdf(
        query_points=points,
        surface=lv,
        axis_origin=c_area,
        axis=axis,
        slab_half_width=slab_half_width,
    )

    sdf_rv_raw = signed_slab_sdf(
        query_points=points,
        surface=rv,
        axis_origin=c_area,
        axis=axis,
        slab_half_width=slab_half_width,
    )

    mask_epi = np.isfinite(sdf_epi_raw).astype(float)
    mask_lv = np.isfinite(sdf_lv_raw).astype(float)
    mask_rv = np.isfinite(sdf_rv_raw).astype(float)

    sdf_epi = np.nan_to_num(sdf_epi_raw, nan=0.0)
    sdf_lv = np.nan_to_num(sdf_lv_raw, nan=0.0)
    sdf_rv = np.nan_to_num(sdf_rv_raw, nan=0.0)

    samples = np.column_stack([
        points,
        sdf_epi,
        sdf_lv,
        sdf_rv,
        mask_epi,
        mask_lv,
        mask_rv,
    ])

    out_npy = output_dir / f"{patient}_mri_samples.npy"
    out_csv = output_dir / f"{patient}_mri_samples.csv"

    if params.save_npy:
        np.save(out_npy, samples)

    if params.save_csv:
        out_df = pd.DataFrame(
            samples,
            columns=[
                "x", "y", "z",
                "sdf_epi", "sdf_lv", "sdf_rv",
                "mask_epi", "mask_lv", "mask_rv",
            ],
        )
        out_df.to_csv(out_csv, index=False)

    stats = {
        "patient": patient,
        "n_squares": len(square_centers),
        "n_samples": len(samples),
        "square_side_length_norm": square_side_length,
        "square_side_length_mm": square_side_length * scale_to_original_mm,
        "scale_to_original_mm": scale_to_original_mm,
        "mask_epi_count": int(mask_epi.sum()),
        "mask_lv_count": int(mask_lv.sum()),
        "mask_rv_count": int(mask_rv.sum()),
        "mask_epi_fraction": float(mask_epi.mean()),
        "mask_lv_fraction": float(mask_lv.mean()),
        "mask_rv_fraction": float(mask_rv.mean()),
        "out_npy": str(out_npy),
    }

    debug = {
        "patient": patient,
        "lv": lv,
        "rv": rv,
        "epi": epi,
        "c_area": c_area,
        "apex": apex,
        "axis": axis,
        "squares": squares,
        "axis_points": axis_points,
        "square_centers": square_centers,
        "points": points,
        "samples": samples,
        "sdf_epi_raw": sdf_epi_raw,
        "sdf_lv_raw": sdf_lv_raw,
        "sdf_rv_raw": sdf_rv_raw,
        "epi_points_used_for_side": epi_points_used_for_side,
        "reference_slice_point": reference_slice_point,
        "stats": stats,
    }

    if params.plot_debug:
        plot_patient_debug(debug)

    return samples, stats, debug


# ============================================================
# DEBUG PLOT
# ============================================================

def plot_patient_debug(debug, scalar_name="sdf_epi_raw"):
    patient = debug["patient"]
    lv = debug["lv"]
    rv = debug["rv"]
    epi = debug["epi"]
    c_area = debug["c_area"]
    apex = debug["apex"]
    axis = debug["axis"]
    squares = debug["squares"]
    axis_points = debug["axis_points"]
    square_centers = debug["square_centers"]
    points = debug["points"]

    scalars = debug[scalar_name]

    valid_mask = np.isfinite(scalars)

    points_plot = points[valid_mask]
    scalars_plot = scalars[valid_mask]

    cloud = pv.PolyData(points_plot)
    cloud[scalar_name] = scalars_plot

    vmin = np.nanmin(scalars_plot)
    vmax = np.nanmax(scalars_plot)

    from matplotlib.colors import LinearSegmentedColormap

    if vmin < 0 < vmax:
        zero_pos = (0.0 - vmin) / (vmax - vmin)
        cmap = LinearSegmentedColormap.from_list(
            "custom_bwr",
            [
                (0.0, "blue"),
                (zero_pos, "white"),
                (1.0, "red"),
            ],
        )
    else:
        cmap = "bwr"

    plotter = pv.Plotter(off_screen=False, notebook=False)

    plotter.add_mesh(lv, color="lightgray", opacity=0.35)
    plotter.add_mesh(rv, color="lightblue", opacity=0.18)
    plotter.add_mesh(epi, color="salmon", opacity=0.18)

    plotter.add_mesh(
        cloud,
        scalars=scalar_name,
        cmap=cmap,
        clim=[vmin, vmax],
        point_size=5,
        render_points_as_spheres=True,
        show_scalar_bar=True,
    )

    for square in squares:
        plotter.add_mesh(
            square,
            color="green",
            opacity=0.08,
            show_edges=True,
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

    plotter.add_mesh(pv.Sphere(radius=0.02, center=c_area), color="magenta")
    plotter.add_mesh(pv.Sphere(radius=0.015, center=apex), color="yellow")

    plotter.add_mesh(
        pv.Line(c_area, apex),
        color="yellow",
        line_width=7,
    )

    plotter.add_text(
        f"{patient}\n{scalar_name}",
        font_size=12,
    )

    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
    )

    plotter.add_axes()
    plotter.show(interactive=True, auto_close=False)