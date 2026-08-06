from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyvista as pv
import igl

from vtk import vtkClipPolyData, vtkPlane


@dataclass
class ThreeAxisMRIParams:
    square_spacing_mm: float = 6.0
    short_axis_slab_width_mm: float = 0.75
    long_axis_volume_width_mm: float = 2.0

    n_before_mitral: int = 3
    n_after_apex: int = 3
    square_margin_factor: float = 1.5

    n_points_per_short_axis_plane: int = 1000
    n_points_per_long_axis_volume: int = 1000

    min_dist_short_axis_mm: float = 1.0
    min_dist_long_axis_mm: float = 1.0

    contour_expansion_mm: float = 25.0
    batch_size: int = 5000
    max_sampling_trials: int = 2_000_000

    plane_23_shift_mm: float = 25.0

    save_npy: bool = True
    save_csv: bool = False
    plot_debug: bool = False


def normalize(v, name="vector"):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        raise ValueError(f"{name} has zero norm.")
    return v / norm


def make_square(center, u, v, side):
    center = np.asarray(center, dtype=float)
    u = normalize(u, "square axis u")
    v = normalize(v, "square axis v")
    half = side / 2.0
    points = np.array([
        center - half * u - half * v,
        center + half * u - half * v,
        center + half * u + half * v,
        center - half * u + half * v,
    ])
    faces = np.hstack([[4, 0, 1, 2, 3]])
    return pv.PolyData(points, faces)


def max_in_plane_extent(points, axis_a, axis_b):
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0:
        return 0.0, 0.0, 0.0
    axis_a = normalize(axis_a, "axis_a")
    axis_b = normalize(axis_b, "axis_b")
    proj_a = np.dot(points, axis_a)
    proj_b = np.dot(points, axis_b)
    range_a = float(np.ptp(proj_a))
    range_b = float(np.ptp(proj_b))
    return max(range_a, range_b), range_a, range_b


def make_parallel_plane_points(start_point, apex_point, axis, spacing,
                               n_before_start=3, n_after_apex=3):
    start_point = np.asarray(start_point, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    axis = normalize(axis, "short-axis translation axis")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")

    apex_vector = apex_point - start_point
    axial_distance = float(np.dot(apex_vector, axis))
    lateral_error = float(np.linalg.norm(apex_vector - axial_distance * axis))
    if axial_distance <= 0.0:
        raise ValueError("The apex must lie in the positive e1 direction.")

    tolerance = max(1e-10, 1e-8 * axial_distance)
    if lateral_error > tolerance:
        raise ValueError(
            "The supplied apex is not on the supplied anatomical axis. "
            f"Lateral error: {lateral_error:.6e}."
        )

    n_full_steps = int(np.floor(axial_distance / spacing))
    plane_points = []

    for i in range(n_before_start, 0, -1):
        plane_points.append(start_point - i * spacing * axis)
    for i in range(n_full_steps + 1):
        plane_points.append(start_point + i * spacing * axis)

    last_regular_distance = n_full_steps * spacing
    for i in range(1, n_after_apex + 1):
        plane_points.append(
            start_point + (last_regular_distance + i * spacing) * axis
        )

    return np.asarray(plane_points, dtype=float)


def find_patient_column(df):
    for name in ["patient", "Patient", "PATIENT", "patient_id",
                 "PatientID", "id", "ID"]:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find patient column. Columns: {list(df.columns)}")


def read_point(row, cols, point_name):
    if not all(column in row.index for column in cols):
        raise ValueError(f"Missing columns for {point_name}: {cols}")
    point = np.array([row[cols[0]], row[cols[1]], row[cols[2]]], dtype=float)
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{point_name} contains non-finite coordinates: {point}")
    return point


def read_patient_three_points(df, patient):
    patient_col = find_patient_column(df)
    row_df = df[df[patient_col].astype(str) == str(patient)]
    if row_df.empty:
        raise ValueError(f"Patient '{patient}' not found in CSV.")
    if len(row_df) > 1:
        raise ValueError(f"Patient '{patient}' appears {len(row_df)} times in CSV.")
    row = row_df.iloc[0]
    c_area = read_point(row, ["C_area_x", "C_area_y", "C_area_z"], "C_area")
    a_maxd = read_point(row, ["A_maxD_x", "A_maxD_y", "A_maxD_z"], "A_maxD")
    t_area = read_point(row, ["T_area_x", "T_area_y", "T_area_z"], "T_area")
    return c_area, a_maxd, t_area


def build_three_axes(c_area, a_maxd, t_area):
    e1 = normalize(a_maxd - c_area, "e1 apex-base")
    raw_e2 = normalize(t_area - c_area, "raw e2 mitral-tricuspid")
    e2 = normalize(raw_e2 - np.dot(raw_e2, e1) * e1, "e2 orthogonalized")
    e3 = normalize(np.cross(e1, e2), "e3")
    e2 = normalize(np.cross(e3, e1), "e2 right-handed")
    basis = np.column_stack([e1, e2, e3])
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-10, rtol=1e-10):
        raise RuntimeError("The anatomical basis is not orthonormal.")
    if np.linalg.det(basis) <= 0.0:
        raise RuntimeError("The anatomical basis is not right-handed.")
    return e1, e2, e3


def prepare_surface(mesh, surface_name):
    surface = mesh.extract_surface(algorithm=None).triangulate().clean()
    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError(f"{surface_name} is empty after preprocessing.")
    if not surface.is_all_triangles:
        raise ValueError(f"{surface_name} is not fully triangulated.")
    try:
        n_open_edges = int(surface.n_open_edges)
    except Exception:
        n_open_edges = -1
    if n_open_edges > 0:
        print(f"WARNING: {surface_name} has {n_open_edges} open edges.")
    return surface


def _clip_keep_positive(polydata, origin, normal):
    plane = vtkPlane()
    plane.SetOrigin(*np.asarray(origin, dtype=float))
    plane.SetNormal(*normalize(normal, "clip plane normal"))
    clipper = vtkClipPolyData()
    clipper.SetInputData(polydata)
    clipper.SetClipFunction(plane)
    clipper.SetValue(0.0)
    clipper.SetInsideOut(False)
    clipper.GenerateClippedOutputOff()
    clipper.Update()
    return pv.wrap(clipper.GetOutput()).copy()


def clip_surface_to_slab(surface, center, normal, slab_half_width):
    if slab_half_width <= 0.0:
        raise ValueError("slab_half_width must be positive.")
    center = np.asarray(center, dtype=float)
    normal = normalize(normal, "slab normal")
    tolerance = max(1e-12, 1e-9 * slab_half_width)
    effective_half_width = slab_half_width + tolerance
    lower_origin = center - effective_half_width * normal
    upper_origin = center + effective_half_width * normal

    clipped = _clip_keep_positive(surface, lower_origin, normal)
    if clipped.n_cells == 0:
        return None
    clipped = _clip_keep_positive(clipped, upper_origin, -normal)
    if clipped.n_cells == 0:
        return None
    clipped = clipped.extract_surface(algorithm=None).triangulate().clean()
    if clipped.n_cells == 0 or clipped.n_points == 0:
        return None
    return clipped


def build_slab_patches(surface, plane_specs, slab_half_width=None):
    patches = []
    for spec in plane_specs:
        current_half_width = spec.get("slab_half_width", slab_half_width)
        if current_half_width is None:
            raise ValueError("Missing slab_half_width.")
        patches.append(
            clip_surface_to_slab(
                surface,
                spec["center"],
                spec["normal"],
                current_half_width,
            )
        )
    return patches


def point_to_patch_distances(query_points, patch):
    query_points = np.asarray(query_points, dtype=float)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError(f"query_points must have shape (N, 3), got {query_points.shape}.")
    if query_points.shape[0] == 0:
        return np.empty(0, dtype=float)
    if patch is None or patch.n_cells == 0:
        return np.full(query_points.shape[0], np.nan, dtype=float)

    cloud = pv.PolyData(query_points)
    evaluated = cloud.compute_implicit_distance(patch, inplace=False)
    return np.abs(np.asarray(evaluated.point_data["implicit_distance"], dtype=float))


def compute_sign_libigl(mesh, query_points):
    query_points = np.asarray(query_points, dtype=float)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError(f"query_points must have shape (N, 3), got {query_points.shape}.")

    vertices = np.asarray(mesh.points, dtype=np.float64)
    faces_raw = np.asarray(mesh.faces)
    if faces_raw.size == 0 or faces_raw.size % 4 != 0:
        raise ValueError("Expected a non-empty triangular PolyData mesh.")
    faces = faces_raw.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("compute_sign_libigl requires triangular faces.")
    faces = faces[:, 1:4].astype(np.int32)

    winding = igl.fast_winding_number(
        V=vertices,
        F=faces,
        Q=query_points.astype(np.float64),
    )
    sign = np.sign(0.5 - np.abs(winding))

    bounds = mesh.bounds
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2],
                 bounds[5] - bounds[4], 1.0)
    outside_point = np.array([[
        bounds[1] + 10.0 * extent,
        bounds[3] + 10.0 * extent,
        bounds[5] + 10.0 * extent,
    ]])
    winding_outside = igl.fast_winding_number(
        V=vertices,
        F=faces,
        Q=outside_point.astype(np.float64),
    )[0]
    outside_sign = np.sign(0.5 - np.abs(winding_outside))
    if outside_sign == 0.0:
        raise RuntimeError("Could not determine the outside sign.")
    if outside_sign < 0.0:
        sign *= -1.0
    return sign


def sample_points_in_oriented_square_min_dist(
    square_center,
    u,
    v,
    side_length,
    n_points,
    min_dist,
    seed=42,
    max_trials=2_000_000,
):
    """
    Campionamento uniforme nel quadrato orientato,
    con distanza minima calcolata nelle coordinate 2D del piano.
    """
    rng = np.random.default_rng(seed)

    square_center = np.asarray(square_center, dtype=float)
    u = normalize(u, "uniform square axis u")
    v = normalize(v, "uniform square axis v")

    half = side_length / 2.0

    accepted_2d = []
    accepted_3d = []

    trials = 0

    while len(accepted_3d) < n_points and trials < max_trials:
        candidate_2d = rng.uniform(
            -half,
            half,
            size=2,
        )

        if accepted_2d:
            distances = np.linalg.norm(
                np.asarray(accepted_2d)
                - candidate_2d[None, :],
                axis=1,
            )

            if np.any(distances < min_dist):
                trials += 1
                continue

        candidate_3d = (
            square_center
            + candidate_2d[0] * u
            + candidate_2d[1] * v
        )

        accepted_2d.append(candidate_2d)
        accepted_3d.append(candidate_3d)

        trials += 1

    if len(accepted_3d) < n_points:
        raise RuntimeError(
            f"Uniform short-axis sampling failed: generated "
            f"{len(accepted_3d)} / {n_points} points."
        )

    return np.asarray(accepted_3d)



def sample_inside_epi_or_near_slab_surface_min_dist(
    square_center,
    u,
    v,
    side_length,
    epi_mesh,
    epi_slab_patch,
    n_points,
    min_dist,
    surface_expansion,
    seed=42,
    max_trials=2_000_000,
    batch_size=5000,
):
    """
    Campiona punti sul piano short-axis centrale.

    Un punto viene accettato quando:
    - è interno all'epicardio, oppure
    - è vicino alla patch epicardica contenuta nello slab.

    Se lo slab non interseca l'epicardio, viene usato il
    campionamento uniforme nel quadrato, come nel generatore originale.
    """

    square_center = np.asarray(square_center, dtype=float)
    u = normalize(u, "sampling axis u")
    v = normalize(v, "sampling axis v")

    # ========================================================
    # FALLBACK: nessuna superficie epicardica nello slab
    # ========================================================

    if (
        epi_slab_patch is None
        or epi_slab_patch.n_points == 0
        or epi_slab_patch.n_cells == 0
    ):
        print(
            "WARNING: no epicardial patch in short-axis slab. "
            "Using uniform square sampling."
        )

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

    # Controllo che la patch intersechi effettivamente il quadrato
    local_patch = (
        np.asarray(epi_slab_patch.points)
        - square_center[None, :]
    )

    patch_u = np.dot(local_patch, u)
    patch_v = np.dot(local_patch, v)

    inside_square = (
        (patch_u >= -half)
        & (patch_u <= half)
        & (patch_v >= -half)
        & (patch_v <= half)
    )

    if not np.any(inside_square):
        print(
            "WARNING: epicardial slab patch does not intersect "
            "the short-axis square. Using uniform square sampling."
        )

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

    accepted_2d = []
    accepted_3d = []

    trials = 0

    while (
        len(accepted_3d) < n_points
        and trials < max_trials
    ):
        current_batch = min(
            batch_size,
            max_trials - trials,
        )

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

        # Punto interno alla superficie epicardica completa
        inside_epi = (
            compute_sign_libigl(
                mesh=epi_mesh,
                query_points=candidate_3d,
            )
            < 0.0
        )

        # Punto vicino alla patch epicardica nello slab
        distance = point_to_patch_distances(
            query_points=candidate_3d,
            patch=epi_slab_patch,
        )

        near_patch = (
            np.isfinite(distance)
            & (distance <= surface_expansion)
        )

        valid = inside_epi | near_patch

        for candidate_ab, candidate_xyz in zip(
            ab[valid],
            candidate_3d[valid],
        ):
            if accepted_2d:
                distances = np.linalg.norm(
                    np.asarray(accepted_2d)
                    - candidate_ab[None, :],
                    axis=1,
                )

                if np.any(distances < min_dist):
                    continue

            accepted_2d.append(candidate_ab)
            accepted_3d.append(candidate_xyz)

            if len(accepted_3d) >= n_points:
                break

        trials += current_batch

    if len(accepted_3d) < n_points:
        raise RuntimeError(
            f"Short-axis boundary-aware sampling failed: generated "
            f"{len(accepted_3d)} / {n_points}. "
            f"Increase contour_expansion_mm, reduce "
            f"min_dist_short_axis_mm, or reduce the number of points."
        )

    return np.asarray(accepted_3d)

sample_inside_epi_or_near_contour_min_dist = (
    sample_inside_epi_or_near_slab_surface_min_dist
)


def sample_inside_epi_or_near_long_axis_volume_min_dist(
    volume_center, normal, u, v, side_length, volume_half_width,
    epi_mesh, epi_slab_patch, n_points, min_dist, surface_expansion,
    seed=42, max_trials=2_000_000, batch_size=5000,
):
    """
    Uniform sampling in the oriented long-axis volume:
        p = center + a*u + b*v + t*normal,
    with t in [-volume_half_width, +volume_half_width].

    Minimum separation is the full 3-D Euclidean distance.
    """
    rng = np.random.default_rng(seed)
    volume_center = np.asarray(volume_center, dtype=float)
    normal = normalize(normal, "long-axis volume normal")
    u = normalize(u, "long-axis volume axis u")
    v = normalize(v, "long-axis volume axis v")
    half = side_length / 2.0
    accepted_3d = []
    trials = 0

    while len(accepted_3d) < n_points and trials < max_trials:
        current_batch = min(batch_size, max_trials - trials)
        ab = rng.uniform(-half, half, size=(current_batch, 2))
        t = rng.uniform(-volume_half_width, volume_half_width, size=current_batch)
        candidate_3d = (
            volume_center[None, :]
            + ab[:, 0:1] * u[None, :]
            + ab[:, 1:2] * v[None, :]
            + t[:, None] * normal[None, :]
        )

        inside_epi = compute_sign_libigl(epi_mesh, candidate_3d) < 0.0
        if epi_slab_patch is None:
            near_patch = np.zeros(current_batch, dtype=bool)
        else:
            distance = point_to_patch_distances(candidate_3d, epi_slab_patch)
            near_patch = np.isfinite(distance) & (distance <= surface_expansion)
        valid = inside_epi | near_patch

        for candidate_xyz in candidate_3d[valid]:
            if accepted_3d:
                distances = np.linalg.norm(
                    np.asarray(accepted_3d) - candidate_xyz[None, :], axis=1
                )
                if np.any(distances < min_dist):
                    continue
            accepted_3d.append(candidate_xyz)
            if len(accepted_3d) >= n_points:
                break
        trials += current_batch

    if len(accepted_3d) < n_points:
        raise RuntimeError(
            f"Long-axis volume sampling failed: generated "
            f"{len(accepted_3d)} / {n_points}."
        )
    return np.asarray(accepted_3d)


def signed_multi_plane_sdf(query_points, surface, plane_ids, slab_patches):
    query_points = np.asarray(query_points, dtype=float)
    plane_ids = np.asarray(plane_ids, dtype=int)
    if query_points.shape[0] != plane_ids.shape[0]:
        raise ValueError("query_points and plane_ids must have the same length.")

    unsigned_distance = np.full(query_points.shape[0], np.nan, dtype=float)
    for plane_id in np.unique(plane_ids):
        if plane_id < 0 or plane_id >= len(slab_patches):
            raise IndexError(f"Invalid plane_id: {plane_id}")
        indices = np.where(plane_ids == plane_id)[0]
        patch = slab_patches[plane_id]
        if patch is None or patch.n_cells == 0:
            continue
        unsigned_distance[indices] = point_to_patch_distances(
            query_points[indices], patch
        )

    global_sign = compute_sign_libigl(surface, query_points)
    return unsigned_distance * global_sign


def generate_patient_three_axis_mri_dataset(
    patient, all_processed_dir, csv_path, output_dir,
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
    for surface_path in [lv_path, rv_path, epi_path]:
        if not surface_path.exists():
            raise FileNotFoundError(surface_path)

    lv_raw, rv_raw, epi_raw = pv.read(lv_path), pv.read(rv_path), pv.read(epi_path)
    if "scale-tooriginalrange" not in epi_raw.field_data:
        raise KeyError(f"'scale-tooriginalrange' missing from {epi_path}")
    scale_values = np.asarray(epi_raw.field_data["scale-tooriginalrange"]).ravel()
    if scale_values.size == 0:
        raise ValueError("'scale-tooriginalrange' is empty.")
    scale_to_original_um = float(scale_values[0])
    if not np.isfinite(scale_to_original_um) or scale_to_original_um <= 0.0:
        raise ValueError(f"Invalid scale-tooriginalrange: {scale_to_original_um}")
    scale_to_original_mm = scale_to_original_um / 1000.0

    lv = prepare_surface(lv_raw, f"{patient} LV endocardium")
    rv = prepare_surface(rv_raw, f"{patient} RV endocardium")
    epi = prepare_surface(epi_raw, f"{patient} epicardium")

    df = pd.read_csv(csv_path, sep=";")
    c_area, a_maxd, t_area = read_patient_three_points(df, patient)
    e1, e2, e3 = build_three_axes(c_area, a_maxd, t_area)

    short_half_width = (params.short_axis_slab_width_mm / 2.0) / scale_to_original_mm
    long_half_width = (params.long_axis_volume_width_mm / 2.0) / scale_to_original_mm
    square_spacing = params.square_spacing_mm / scale_to_original_mm
    min_dist_short = params.min_dist_short_axis_mm / scale_to_original_mm
    min_dist_long = params.min_dist_long_axis_mm / scale_to_original_mm
    surface_expansion = params.contour_expansion_mm / scale_to_original_mm
    plane_23_shift = params.plane_23_shift_mm / scale_to_original_mm
    c_long = c_area + plane_23_shift * e1

    reference_specs = [
        {"center": c_area, "normal": e1, "u": e2, "v": e3,
         "slab_half_width": short_half_width},
        {"center": c_long, "normal": e2, "u": e1, "v": e3,
         "slab_half_width": long_half_width},
        {"center": c_long, "normal": e3, "u": e1, "v": e2,
         "slab_half_width": long_half_width},
    ]
    reference_patches = build_slab_patches(epi, reference_specs)
    reference_points = [
        np.empty((0, 3), dtype=float) if p is None else np.asarray(p.points)
        for p in reference_patches
    ]

    _, r0a, r0b = max_in_plane_extent(reference_points[0], e2, e3)
    _, r1a, r1b = max_in_plane_extent(reference_points[1], e1, e3)
    _, r2a, r2b = max_in_plane_extent(reference_points[2], e1, e2)
    all_ranges = np.array([r0a, r0b, r1a, r1b, r2a, r2b])
    valid_ranges = all_ranges[np.isfinite(all_ranges) & (all_ranges > 0.0)]
    if valid_ranges.size == 0:
        raise ValueError("Cannot estimate square side.")
    square_side = params.square_margin_factor * float(valid_ranges.max())

    short_axis_plane_points = make_parallel_plane_points(
        c_area, a_maxd, e1, square_spacing,
        params.n_before_mitral, params.n_after_apex,
    )

    plane_specs = []
    for center in short_axis_plane_points:
        plane_specs.append({
            "type": "short_axis", "center": center, "normal": e1,
            "u": e2, "v": e3,
            "slab_half_width": short_half_width,
            "sampling_mode": "plane",
        })
    plane_specs.append({
        "type": "normal_e2", "center": c_long, "normal": e2,
        "u": e1, "v": e3,
        "slab_half_width": long_half_width,
        "sampling_mode": "volume",
    })
    plane_specs.append({
        "type": "normal_e3", "center": c_long, "normal": e3,
        "u": e1, "v": e2,
        "slab_half_width": long_half_width,
        "sampling_mode": "volume",
    })

    epi_patches = build_slab_patches(epi, plane_specs)
    lv_patches = build_slab_patches(lv, plane_specs)
    rv_patches = build_slab_patches(rv, plane_specs)

    all_points, all_plane_ids, all_type_ids = [], [], []
    type_to_id = {"short_axis": 0, "normal_e2": 1, "normal_e3": 2}

    for plane_id, spec in enumerate(plane_specs):
        if spec["sampling_mode"] == "plane":
            points_i = sample_inside_epi_or_near_slab_surface_min_dist(
                spec["center"], spec["u"], spec["v"], square_side,
                epi, epi_patches[plane_id],
                params.n_points_per_short_axis_plane,
                min_dist_short, surface_expansion,
                seed=42 + plane_id,
                max_trials=params.max_sampling_trials,
                batch_size=params.batch_size,
            )
        else:
            points_i = sample_inside_epi_or_near_long_axis_volume_min_dist(
                spec["center"], spec["normal"], spec["u"], spec["v"],
                square_side, spec["slab_half_width"],
                epi, epi_patches[plane_id],
                params.n_points_per_long_axis_volume,
                min_dist_long, surface_expansion,
                seed=42 + plane_id,
                max_trials=params.max_sampling_trials,
                batch_size=params.batch_size,
            )

        all_points.append(points_i)
        all_plane_ids.append(np.full(points_i.shape[0], plane_id, dtype=int))
        all_type_ids.append(np.full(points_i.shape[0], type_to_id[spec["type"]], dtype=int))

    points = np.vstack(all_points)
    plane_ids = np.concatenate(all_plane_ids)
    plane_type_ids = np.concatenate(all_type_ids)

    sdf_epi_raw = signed_multi_plane_sdf(points, epi, plane_ids, epi_patches)
    sdf_lv_raw = signed_multi_plane_sdf(points, lv, plane_ids, lv_patches)
    sdf_rv_raw = signed_multi_plane_sdf(points, rv, plane_ids, rv_patches)

    mask_epi = np.isfinite(sdf_epi_raw).astype(float)
    mask_lv = np.isfinite(sdf_lv_raw).astype(float)
    mask_rv = np.isfinite(sdf_rv_raw).astype(float)

    sdf_epi = np.nan_to_num(sdf_epi_raw, nan=0.0, posinf=0.0, neginf=0.0)
    sdf_lv = np.nan_to_num(sdf_lv_raw, nan=0.0, posinf=0.0, neginf=0.0)
    sdf_rv = np.nan_to_num(sdf_rv_raw, nan=0.0, posinf=0.0, neginf=0.0)

    samples = np.column_stack([
        points, sdf_epi, sdf_lv, sdf_rv, mask_epi, mask_lv, mask_rv
    ]).astype(np.float32)

    out_npy = output_dir / f"{patient}_three_axis_mri_samples.npy"
    if params.save_npy:
        np.save(out_npy, samples)
    if params.save_csv:
        pd.DataFrame(samples, columns=[
            "x", "y", "z", "sdf_epi", "sdf_lv", "sdf_rv",
            "mask_epi", "mask_lv", "mask_rv",
        ]).to_csv(output_dir / f"{patient}_three_axis_mri_samples.csv", index=False)

    long_mask = plane_type_ids > 0
    long_depth_mm = np.full(len(points), np.nan, dtype=float)
    for plane_id, spec in enumerate(plane_specs):
        if spec["sampling_mode"] == "volume":
            idx = plane_ids == plane_id
            long_depth_mm[idx] = (
                np.dot(points[idx] - spec["center"], spec["normal"])
                * scale_to_original_mm
            )

    stats = {
        "patient": patient,
        "n_planes": len(plane_specs),
        "n_short_axis_planes": len(short_axis_plane_points),
        "n_long_axis_volumes": 2,
        "n_samples": len(samples),
        "n_short_axis_samples": int((~long_mask).sum()),
        "n_long_axis_samples": int(long_mask.sum()),
        "short_axis_slab_width_mm": params.short_axis_slab_width_mm,
        "long_axis_volume_width_mm": params.long_axis_volume_width_mm,
        "long_axis_depth_min_mm": float(np.nanmin(long_depth_mm)),
        "long_axis_depth_max_mm": float(np.nanmax(long_depth_mm)),
        "square_side_mm": float(square_side * scale_to_original_mm),
        "mask_epi_fraction": float(mask_epi.mean()),
        "mask_lv_fraction": float(mask_lv.mean()),
        "mask_rv_fraction": float(mask_rv.mean()),
        "out_npy": str(out_npy),
    }

    debug = {
        "patient": patient,
        "lv": lv, "rv": rv, "epi": epi,
        "c_area": c_area, "a_maxd": a_maxd, "t_area": t_area,
        "c_long": c_long, "e1": e1, "e2": e2, "e3": e3,
        "plane_specs": plane_specs,
        "plane_ids": plane_ids,
        "plane_type_ids": plane_type_ids,
        "epi_slab_patches": epi_patches,
        "lv_slab_patches": lv_patches,
        "rv_slab_patches": rv_patches,
        "points": points,
        "samples": samples,
        "long_axis_depth_mm": long_depth_mm,
        "square_side": square_side,
        "scale_to_original_mm": scale_to_original_mm,
        "stats": stats,
    }

    if params.plot_debug:
        plot_patient_debug(debug)

    return samples, stats, debug


def plot_patient_debug(debug):
    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 850))
    long_specs = [
        (i, spec) for i, spec in enumerate(debug["plane_specs"])
        if spec["sampling_mode"] == "volume"
    ]

    for subplot_id, (plane_id, spec) in enumerate(long_specs):
        plotter.subplot(0, subplot_id)
        width_mm = 2.0 * spec["slab_half_width"] * debug["scale_to_original_mm"]
        plotter.add_text(
            f'{debug["patient"]} - {spec["type"]} - {width_mm:.2f} mm',
            font_size=14,
        )
        plotter.add_mesh(debug["epi"], color="lightgray", opacity=0.08)
        patch = debug["epi_slab_patches"][plane_id]
        if patch is not None:
            plotter.add_mesh(
                patch, color="orange", opacity=0.40,
                show_edges=True, edge_color="black",
            )

        idx = debug["plane_ids"] == plane_id
        cloud = pv.PolyData(debug["points"][idx])
        depth_mm = (
            np.dot(
                debug["points"][idx] - spec["center"],
                spec["normal"],
            ) * debug["scale_to_original_mm"]
        )
        cloud["depth_mm"] = depth_mm
        plotter.add_mesh(
            cloud, scalars="depth_mm", point_size=7,
            render_points_as_spheres=True,
            scalar_bar_args={"title": "depth [mm]"},
        )

        lower = make_square(
            spec["center"] - spec["slab_half_width"] * spec["normal"],
            spec["u"], spec["v"], debug["square_side"],
        )
        upper = make_square(
            spec["center"] + spec["slab_half_width"] * spec["normal"],
            spec["u"], spec["v"], debug["square_side"],
        )
        plotter.add_mesh(lower, color="blue", opacity=0.12, show_edges=True)
        plotter.add_mesh(upper, color="red", opacity=0.12, show_edges=True)
        plotter.add_axes()
        plotter.show_bounds(grid="front", location="outer", all_edges=True)

    plotter.link_views()
    plotter.show()