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
class ThreeAxisMRIParams:
    square_spacing_mm: float = 6.0
    slab_width_mm: float = 0.75

    n_before_mitral: int = 3
    n_after_apex: int = 3

    square_margin_factor: float = 1.5

    n_points_per_square: int = 1000
    min_dist_mm: float = 1.0

    contour_expansion_mm: float = 5.0
    batch_size: int = 5000

    plane_23_shift_mm: float = 25.0

    save_npy: bool = True
    save_csv: bool = False
    plot_debug: bool = False


# ============================================================
# BASIC GEOMETRY
# ============================================================

def normalize(v, name="vector"):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n == 0:
        raise ValueError(f"{name} has zero norm.")

    return v / n


def make_square(center, u, v, side):
    h = side / 2.0

    p0 = center - h * u - h * v
    p1 = center + h * u - h * v
    p2 = center + h * u + h * v
    p3 = center - h * u + h * v

    points = np.array([p0, p1, p2, p3])
    faces = np.hstack([[4, 0, 1, 2, 3]])

    return pv.PolyData(points, faces)


def max_in_plane_extent(points, axis_a, axis_b):
    if points.shape[0] == 0:
        return 0.0, 0.0, 0.0

    proj_a = np.dot(points, axis_a)
    proj_b = np.dot(points, axis_b)

    range_a = proj_a.max() - proj_a.min()
    range_b = proj_b.max() - proj_b.min()

    return max(range_a, range_b), range_a, range_b


def make_parallel_plane_points(
    start_point,
    apex_point,
    axis,
    spacing,
    n_before_start=3,
    n_after_apex=3,
):
    start_point = np.asarray(start_point, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    axis = normalize(axis, "axis")

    axis_length = np.linalg.norm(apex_point - start_point)

    if spacing <= 0:
        raise ValueError("Spacing must be positive.")

    n_full_steps = int(np.floor(axis_length / spacing))

    plane_points = []

    for i in range(n_before_start, 0, -1):
        plane_points.append(start_point - i * spacing * axis)

    for i in range(n_full_steps + 1):
        plane_points.append(start_point + i * spacing * axis)

    last_distance = n_full_steps * spacing

    for i in range(1, n_after_apex + 1):
        plane_points.append(start_point + (last_distance + i * spacing) * axis)

    return np.asarray(plane_points)


# ============================================================
# CSV HELPERS
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

    raise ValueError(f"Could not find patient column. Columns: {list(df.columns)}")


def read_point(row, cols, point_name):
    if not all(c in row.index for c in cols):
        raise ValueError(f"Missing columns for {point_name}: {cols}")

    return np.array(
        [row[cols[0]], row[cols[1]], row[cols[2]]],
        dtype=float,
    )


def read_patient_three_points(df, patient):
    patient_col = find_patient_column(df)

    row_df = df[df[patient_col].astype(str) == str(patient)]

    if row_df.empty:
        raise ValueError(f"Patient '{patient}' not found in CSV.")

    row = row_df.iloc[0]

    c_area = read_point(
        row,
        ["C_area_x", "C_area_y", "C_area_z"],
        "C_area",
    )

    a_maxd = read_point(
        row,
        ["A_maxD_x", "A_maxD_y", "A_maxD_z"],
        "A_maxD",
    )

    t_area = read_point(
        row,
        ["T_area_x", "T_area_y", "T_area_z"],
        "T_area",
    )

    return c_area, a_maxd, t_area


# ============================================================
# AXES
# ============================================================

def build_three_axes(c_area, a_maxd, t_area):
    e1 = normalize(a_maxd - c_area, "e1 apex-base")

    raw_e2 = normalize(t_area - c_area, "raw e2 mitral-tricuspid")

    e2 = raw_e2 - np.dot(raw_e2, e1) * e1
    e2 = normalize(e2, "e2 orthogonalized")

    e3 = np.cross(e1, e2)
    e3 = normalize(e3, "e3")

    e2 = np.cross(e3, e1)
    e2 = normalize(e2, "e2 right-handed")

    return e1, e2, e3


# ============================================================
# SIGN
# ============================================================

def compute_sign_libigl(mesh, query_points):
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


# ============================================================
# SAMPLING
# ============================================================

def estimate_max_points_in_square(side_length, min_dist):
    return int(
        (2.0 / np.sqrt(3.0))
        * side_length**2
        / min_dist**2
    )


def sample_points_in_oriented_square_min_dist(
    square_center,
    u,
    v,
    side_length,
    n_points,
    min_dist,
    seed=42,
    max_trials=1_000_000,
):
    rng = np.random.default_rng(seed)

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

        if len(points_2d) > 0:
            existing_2d = np.asarray(points_2d)
            dists = np.linalg.norm(existing_2d - candidate_2d, axis=1)

            if np.any(dists < min_dist):
                trials += 1
                continue

        candidate_3d = square_center + a * u + b * v

        points_2d.append(candidate_2d)
        points_3d.append(candidate_3d)

        trials += 1

    if len(points_3d) < n_points:
        raise RuntimeError(
            f"Uniform sampling failed: generated only "
            f"{len(points_3d)} / {n_points} after {max_trials} trials."
        )

    return np.asarray(points_3d)


def sample_inside_epi_or_near_contour_min_dist(
    square_center,
    u,
    v,
    side_length,
    epi_slice_points,
    epi_mesh,
    n_points,
    min_dist,
    contour_expansion,
    seed=42,
    max_trials=2_000_000,
    batch_size=5000,
):
    """
    Accetta un sample se:
    - è dentro l'epicardio
      oppure
    - è vicino alla traccia dell'epicardio nel piano.

    Se la slice non interseca l'epicardio, usa sampling uniforme.
    """

    if epi_slice_points.shape[0] == 0:
        return sample_points_in_oriented_square_min_dist(
            square_center=square_center,
            u=u,
            v=v,
            side_length=side_length,
            n_points=n_points,
            min_dist=min_dist,
            seed=seed,
            max_trials=max_trials,
        )

    rng = np.random.default_rng(seed)

    half = side_length / 2.0

    local_epi = epi_slice_points - square_center

    epi_2d = np.column_stack([
        np.dot(local_epi, u),
        np.dot(local_epi, v),
    ])

    inside_epi_square_mask = (
        (epi_2d[:, 0] >= -half) &
        (epi_2d[:, 0] <=  half) &
        (epi_2d[:, 1] >= -half) &
        (epi_2d[:, 1] <=  half)
    )

    epi_2d = epi_2d[inside_epi_square_mask]

    if epi_2d.shape[0] == 0:
        return sample_points_in_oriented_square_min_dist(
            square_center=square_center,
            u=u,
            v=v,
            side_length=side_length,
            n_points=n_points,
            min_dist=min_dist,
            seed=seed,
            max_trials=max_trials,
        )

    points_2d = []
    points_3d = []

    trials = 0

    while len(points_3d) < n_points and trials < max_trials:
        current_batch = min(batch_size, max_trials - trials)

        ab = rng.uniform(
            -half,
            half,
            size=(current_batch, 2),
        )

        candidate_3d = (
            square_center[None, :]
            + ab[:, 0:1] * u[None, :]
            + ab[:, 1:2] * v[None, :]
        )

        sign_epi = compute_sign_libigl(
            mesh=epi_mesh,
            query_points=candidate_3d,
        )

        inside_epi = sign_epi < 0

        near_contour = np.zeros(current_batch, dtype=bool)

        for j in range(current_batch):
            dists_to_contour = np.linalg.norm(
                epi_2d - ab[j],
                axis=1,
            )

            near_contour[j] = (
                dists_to_contour.min()
                <= contour_expansion
            )

        valid_region = inside_epi | near_contour

        valid_ab = ab[valid_region]
        valid_3d = candidate_3d[valid_region]

        for candidate_2d, candidate_point_3d in zip(valid_ab, valid_3d):

            if len(points_2d) > 0:
                existing_2d = np.asarray(points_2d)

                dists_samples = np.linalg.norm(
                    existing_2d - candidate_2d,
                    axis=1,
                )

                if np.any(dists_samples < min_dist):
                    continue

            points_2d.append(candidate_2d)
            points_3d.append(candidate_point_3d)

            if len(points_3d) >= n_points:
                break

        trials += current_batch

    if len(points_3d) < n_points:
        raise RuntimeError(
            f"Boundary-aware sampling failed: generated only "
            f"{len(points_3d)} / {n_points}. "
            f"Increase contour_expansion_mm, reduce min_dist_mm, "
            f"or reduce n_points_per_square."
        )

    return np.asarray(points_3d)


# ============================================================
# SDF
# ============================================================

def l2_distance_to_surface_in_slab(
    query_points,
    surface_points,
    plane_centers,
    plane_normals,
    slab_half_width,
):
    distances = np.full(len(query_points), np.nan, dtype=float)

    for i, p in enumerate(query_points):
        center = plane_centers[i]
        normal = plane_normals[i]

        q_surface = np.dot(surface_points - center, normal)

        slab_mask = np.abs(q_surface) <= slab_half_width
        slab_points = surface_points[slab_mask]

        if slab_points.shape[0] == 0:
            continue

        dists = np.linalg.norm(slab_points - p[None, :], axis=1)
        distances[i] = dists.min()

    return distances


def signed_multi_plane_sdf(
    query_points,
    surface,
    plane_centers,
    plane_normals,
    slab_half_width,
):
    unsigned_distance = l2_distance_to_surface_in_slab(
        query_points=query_points,
        surface_points=surface.points,
        plane_centers=plane_centers,
        plane_normals=plane_normals,
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

def generate_patient_three_axis_mri_dataset(
    patient,
    all_processed_dir,
    csv_path,
    output_dir,
    params=ThreeAxisMRIParams(),
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

    df = pd.read_csv(csv_path, sep=";")

    c_area, a_maxd, t_area = read_patient_three_points(
        df,
        patient,
    )

    e1, e2, e3 = build_three_axes(
        c_area,
        a_maxd,
        t_area,
    )

    c_short = np.dot(c_area, e1) * e1

    scale_to_original_um = epi.field_data["scale-tooriginalrange"][0]
    scale_to_original_mm = scale_to_original_um / 1000.0

    slab_half_width = (
        params.slab_width_mm / 2.0
    ) / scale_to_original_mm

    square_spacing = (
        params.square_spacing_mm
        / scale_to_original_mm
    )

    min_dist = (
        params.min_dist_mm
        / scale_to_original_mm
    )

    contour_expansion = (
        params.contour_expansion_mm
        / scale_to_original_mm
    )

    plane_23_shift = (
        params.plane_23_shift_mm
        / scale_to_original_mm
    )

    c_long = c_area + plane_23_shift * e1

    epi_points = epi.points

    # --------------------------------------------------------
    # Estimate common square side
    # --------------------------------------------------------

    d_short = np.dot(epi_points - c_short, e1)
    d_e2 = np.dot(epi_points - c_long, e2)
    d_e3 = np.dot(epi_points - c_long, e3)

    epi_points_short = epi_points[np.abs(d_short) <= slab_half_width]
    epi_points_e2 = epi_points[np.abs(d_e2) <= slab_half_width]
    epi_points_e3 = epi_points[np.abs(d_e3) <= slab_half_width]

    _, short_range_e2, short_range_e3 = max_in_plane_extent(
        epi_points_short,
        e2,
        e3,
    )

    _, normal_e2_range_e1, normal_e2_range_e3 = max_in_plane_extent(
        epi_points_e2,
        e1,
        e3,
    )

    _, normal_e3_range_e1, normal_e3_range_e2 = max_in_plane_extent(
        epi_points_e3,
        e1,
        e2,
    )

    all_ranges = [
        short_range_e2,
        short_range_e3,
        normal_e2_range_e1,
        normal_e2_range_e3,
        normal_e3_range_e1,
        normal_e3_range_e2,
    ]

    max_range = max(all_ranges)

    if max_range <= 0:
        raise ValueError("Cannot estimate square side: all ranges are zero.")

    square_side = params.square_margin_factor * max_range

    # --------------------------------------------------------
    # Build planes
    # --------------------------------------------------------

    short_axis_plane_points = make_parallel_plane_points(
        start_point=c_short,
        apex_point=a_maxd,
        axis=e1,
        spacing=square_spacing,
        n_before_start=params.n_before_mitral,
        n_after_apex=params.n_after_apex,
    )

    plane_specs = []

    for i, center in enumerate(short_axis_plane_points):
        plane_specs.append({
            "type": "short_axis",
            "center": center,
            "normal": e1,
            "u": e2,
            "v": e3,
        })

    plane_specs.append({
        "type": "normal_e2",
        "center": c_long,
        "normal": e2,
        "u": e1,
        "v": e3,
    })

    plane_specs.append({
        "type": "normal_e3",
        "center": c_long,
        "normal": e3,
        "u": e1,
        "v": e2,
    })

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    all_points = []
    all_plane_ids = []
    all_plane_type_ids = []
    all_plane_centers_per_point = []
    all_plane_normals_per_point = []

    type_to_id = {
        "short_axis": 0,
        "normal_e2": 1,
        "normal_e3": 2,
    }

    for plane_id, spec in enumerate(plane_specs):
        center = spec["center"]
        normal = spec["normal"]
        u = spec["u"]
        v = spec["v"]

        d = np.dot(epi_points - center, normal)
        epi_slice_points = epi_points[np.abs(d) <= slab_half_width]

        points_i = sample_inside_epi_or_near_contour_min_dist(
            square_center=center,
            u=u,
            v=v,
            side_length=square_side,
            epi_slice_points=epi_slice_points,
            epi_mesh=epi,
            n_points=params.n_points_per_square,
            min_dist=min_dist,
            contour_expansion=contour_expansion,
            seed=42 + plane_id,
            batch_size=params.batch_size,
        )

        all_points.append(points_i)

        all_plane_ids.append(
            np.full(points_i.shape[0], plane_id, dtype=int)
        )

        all_plane_type_ids.append(
            np.full(
                points_i.shape[0],
                type_to_id[spec["type"]],
                dtype=int,
            )
        )

        all_plane_centers_per_point.append(
            np.repeat(center[None, :], points_i.shape[0], axis=0)
        )

        all_plane_normals_per_point.append(
            np.repeat(normal[None, :], points_i.shape[0], axis=0)
        )

    points = np.vstack(all_points)
    plane_ids = np.concatenate(all_plane_ids)
    plane_type_ids = np.concatenate(all_plane_type_ids)
    plane_centers_per_point = np.vstack(all_plane_centers_per_point)
    plane_normals_per_point = np.vstack(all_plane_normals_per_point)

    # --------------------------------------------------------
    # SDF
    # --------------------------------------------------------

    sdf_epi_raw = signed_multi_plane_sdf(
        query_points=points,
        surface=epi,
        plane_centers=plane_centers_per_point,
        plane_normals=plane_normals_per_point,
        slab_half_width=slab_half_width,
    )

    sdf_lv_raw = signed_multi_plane_sdf(
        query_points=points,
        surface=lv,
        plane_centers=plane_centers_per_point,
        plane_normals=plane_normals_per_point,
        slab_half_width=slab_half_width,
    )

    sdf_rv_raw = signed_multi_plane_sdf(
        query_points=points,
        surface=rv,
        plane_centers=plane_centers_per_point,
        plane_normals=plane_normals_per_point,
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

    out_npy = output_dir / f"{patient}_three_axis_mri_samples.npy"

    if params.save_npy:
        np.save(out_npy, samples)

    if params.save_csv:
        out_csv = output_dir / f"{patient}_three_axis_mri_samples.csv"

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
        "n_planes": len(plane_specs),
        "n_short_axis_planes": len(short_axis_plane_points),
        "n_samples": len(samples),
        "n_points_per_square": params.n_points_per_square,
        "square_side_norm": square_side,
        "square_side_mm": square_side * scale_to_original_mm,
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
        "a_maxd": a_maxd,
        "t_area": t_area,
        "c_short": c_short,
        "c_long": c_long,
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "plane_specs": plane_specs,
        "short_axis_plane_points": short_axis_plane_points,
        "points": points,
        "samples": samples,
        "plane_ids": plane_ids,
        "plane_type_ids": plane_type_ids,
        "sdf_epi_raw": sdf_epi_raw,
        "sdf_lv_raw": sdf_lv_raw,
        "sdf_rv_raw": sdf_rv_raw,
        "stats": stats,
    }

    if params.plot_debug:
        plot_three_axis_debug(debug)

    return samples, stats, debug


# ============================================================
# DEBUG PLOT
# ============================================================

def plot_three_axis_debug(debug):
    lv = debug["lv"]
    rv = debug["rv"]
    epi = debug["epi"]
    c_area = debug["c_area"]
    a_maxd = debug["a_maxd"]
    t_area = debug["t_area"]
    plane_specs = debug["plane_specs"]
    points = debug["points"]
    plane_type_ids = debug["plane_type_ids"]

    plotter = pv.Plotter()

    plotter.add_mesh(epi, color="lightgray", opacity=0.15)
    plotter.add_mesh(lv, color="red", opacity=0.15)
    plotter.add_mesh(rv, color="blue", opacity=0.15)

    plotter.add_mesh(pv.Sphere(radius=0.03, center=c_area), color="magenta")
    plotter.add_mesh(pv.Sphere(radius=0.03, center=a_maxd), color="black")
    plotter.add_mesh(pv.Sphere(radius=0.03, center=t_area), color="cyan")

    for spec in plane_specs:
        square = make_square(
            spec["center"],
            spec["u"],
            spec["v"],
            debug["stats"]["square_side_norm"],
        )

        color = {
            "short_axis": "green",
            "normal_e2": "orange",
            "normal_e3": "purple",
        }[spec["type"]]

        opacity = 0.08 if spec["type"] == "short_axis" else 0.25

        plotter.add_mesh(
            square,
            color=color,
            opacity=opacity,
            show_edges=True,
        )

    plotter.add_mesh(
        pv.PolyData(points[plane_type_ids == 0]),
        color="green",
        point_size=3,
        render_points_as_spheres=True,
    )

    plotter.add_mesh(
        pv.PolyData(points[plane_type_ids == 1]),
        color="orange",
        point_size=3,
        render_points_as_spheres=True,
    )

    plotter.add_mesh(
        pv.PolyData(points[plane_type_ids == 2]),
        color="purple",
        point_size=3,
        render_points_as_spheres=True,
    )

    plotter.add_axes()
    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
    )

    plotter.show()