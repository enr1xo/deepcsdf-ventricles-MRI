from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyvista as pv
import igl

from vtk import vtkClipPolyData, vtkPlane


# ============================================================
# PARAMETERS
# ============================================================

@dataclass
class ThreeAxisMRIParams:
    """
    Parameters for the generation of a three-axis MRI-like dataset.

    Geometric interpretation:
    - Samples lie on a regular grid of central acquisition planes.
    - Each slice has a finite thickness, represented by a slab.
    - The SDF magnitude is the Euclidean point-to-triangle distance
      from a sample to the portion of the cardiac surface contained
      in the corresponding slab.
    - The sign is computed with respect to the complete closed surface.
    """

    square_spacing_mm: float = 6.0
    slab_width_mm: float = 0.75

    n_before_mitral: int = 3
    n_after_apex: int = 3
    square_margin_factor: float = 1.5

    n_points_per_square: int = 1000
    min_dist_mm: float = 1.0

    contour_expansion_mm: float = 25.0
    batch_size: int = 5000
    max_sampling_trials: int = 2_000_000

    plane_23_shift_mm: float = 25.0

    save_npy: bool = True
    save_csv: bool = False
    plot_debug: bool = False


# ============================================================
# BASIC GEOMETRY
# ============================================================

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

    p0 = center - half * u - half * v
    p1 = center + half * u - half * v
    p2 = center + half * u + half * v
    p3 = center - half * u + half * v

    points = np.array([p0, p1, p2, p3])
    faces = np.hstack([[4, 0, 1, 2, 3]])

    return pv.PolyData(points, faces)


def max_in_plane_extent(points, axis_a, axis_b):
    """
    Return the maximum of the two in-plane ranges and the individual ranges.
    """
    points = np.asarray(points, dtype=float)

    if points.shape[0] == 0:
        return 0.0, 0.0, 0.0

    axis_a = normalize(axis_a, "axis_a")
    axis_b = normalize(axis_b, "axis_b")

    proj_a = np.dot(points, axis_a)
    proj_b = np.dot(points, axis_b)

    range_a = float(proj_a.max() - proj_a.min())
    range_b = float(proj_b.max() - proj_b.min())

    return max(range_a, range_b), range_a, range_b


def make_parallel_plane_points(
    start_point,
    apex_point,
    axis,
    spacing,
    n_before_start=3,
    n_after_apex=3,
):
    """
    Generate a single regularly spaced sequence of parallel short-axis planes.

    The sequence contains:
    - n_before_start planes before the mitral centroid;
    - planes every `spacing` from the mitral centroid towards the apex;
    - n_after_apex additional planes continuing the same regular grid.

    No additional plane is inserted exactly through the apex. Therefore, the
    apex generally lies between two consecutive planes unless its axial
    distance from the mitral centroid is an exact multiple of `spacing`.

    This reproduces the original plane-generation logic while keeping the
    anatomical mitral centroid as the starting point.
    """
    start_point = np.asarray(start_point, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    axis = normalize(axis, "short-axis translation axis")

    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")
    if n_before_start < 0:
        raise ValueError("n_before_start must be non-negative.")
    if n_after_apex < 0:
        raise ValueError("n_after_apex must be non-negative.")

    apex_vector = apex_point - start_point
    axial_distance = float(np.dot(apex_vector, axis))

    lateral_component = apex_vector - axial_distance * axis
    lateral_error = float(np.linalg.norm(lateral_component))

    if axial_distance <= 0.0:
        raise ValueError(
            "The apex must lie in the positive direction of the "
            "mitral-to-apex axis."
        )

    tolerance = max(1e-10, 1e-8 * axial_distance)

    if lateral_error > tolerance:
        raise ValueError(
            "The supplied apex is not on the supplied anatomical axis. "
            f"Lateral error: {lateral_error:.6e}."
        )

    n_full_steps = int(np.floor(axial_distance / spacing))

    plane_points = []

    # Regular-grid planes before the mitral centroid.
    for i in range(n_before_start, 0, -1):
        plane_points.append(
            start_point - i * spacing * axis
        )

    # Regular grid from the mitral centroid towards the apex.
    for i in range(n_full_steps + 1):
        plane_points.append(
            start_point + i * spacing * axis
        )

    # Continue the same grid beyond its final pre-apical position.
    last_regular_distance = n_full_steps * spacing

    for i in range(1, n_after_apex + 1):
        plane_points.append(
            start_point
            + (last_regular_distance + i * spacing) * axis
        )

    return np.asarray(plane_points, dtype=float)


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

    raise ValueError(
        f"Could not find patient column. Columns: {list(df.columns)}"
    )


def read_point(row, cols, point_name):
    if not all(column in row.index for column in cols):
        raise ValueError(f"Missing columns for {point_name}: {cols}")

    point = np.array(
        [row[cols[0]], row[cols[1]], row[cols[2]]],
        dtype=float,
    )

    if not np.all(np.isfinite(point)):
        raise ValueError(f"{point_name} contains non-finite coordinates: {point}")

    return point


def read_patient_three_points(df, patient):
    patient_col = find_patient_column(df)

    row_df = df[df[patient_col].astype(str) == str(patient)]

    if row_df.empty:
        raise ValueError(f"Patient '{patient}' not found in CSV.")

    if len(row_df) > 1:
        raise ValueError(
            f"Patient '{patient}' appears {len(row_df)} times in the CSV."
        )

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
    """
    Build an orthonormal right-handed anatomical reference frame.

    e1: mitral centroid -> LV apex
    e2: orthogonalized mitral centroid -> tricuspid centroid
    e3: e1 x e2
    """
    e1 = normalize(a_maxd - c_area, "e1 apex-base")

    raw_e2 = normalize(
        t_area - c_area,
        "raw e2 mitral-tricuspid",
    )

    e2 = raw_e2 - np.dot(raw_e2, e1) * e1
    e2 = normalize(e2, "e2 orthogonalized")

    e3 = np.cross(e1, e2)
    e3 = normalize(e3, "e3")

    # Recompute e2 to minimize numerical loss of orthogonality.
    e2 = np.cross(e3, e1)
    e2 = normalize(e2, "e2 right-handed")

    basis = np.column_stack([e1, e2, e3])
    gram = basis.T @ basis

    if not np.allclose(gram, np.eye(3), atol=1e-10, rtol=1e-10):
        raise RuntimeError(
            "The anatomical basis is not sufficiently orthonormal.\n"
            f"Gram matrix:\n{gram}"
        )

    if np.linalg.det(basis) <= 0.0:
        raise RuntimeError("The anatomical basis is not right-handed.")

    return e1, e2, e3


# ============================================================
# SURFACE PREPARATION AND SLAB CLIPPING
# ============================================================

def prepare_surface(mesh, surface_name):
    """
    Convert a mesh to clean triangular PolyData.

    The winding-number sign is most reliable for closed surfaces. Open edges
    are reported but do not automatically stop execution, because some source
    meshes may intentionally contain valve openings.
    """
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
        print(
            f"WARNING: {surface_name} has {n_open_edges} open edges. "
            "Winding-number signs should be checked carefully."
        )

    return surface


def _clip_keep_positive(polydata, origin, normal):
    """
    Keep the part of polydata for which vtkPlane(x) >= 0.
    """
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


def clip_surface_to_slab(
    surface,
    center,
    normal,
    slab_half_width,
):
    """
    Intersect a triangular surface with a finite-thickness slab.

    The slab is:
        |(x - center) dot normal| <= slab_half_width

    Triangles crossing either slab boundary are geometrically clipped, so the
    result does not depend only on whether original mesh vertices happen to lie
    inside the slab.
    """
    if slab_half_width <= 0.0:
        raise ValueError("slab_half_width must be positive.")

    center = np.asarray(center, dtype=float)
    normal = normalize(normal, "slab normal")

    # Tiny expansion avoids dropping cells lying numerically on a boundary.
    tolerance = max(1e-12, 1e-9 * slab_half_width)
    effective_half_width = slab_half_width + tolerance

    lower_origin = center - effective_half_width * normal
    upper_origin = center + effective_half_width * normal

    # Keep points above the lower boundary.
    clipped = _clip_keep_positive(
        surface,
        origin=lower_origin,
        normal=normal,
    )

    if clipped.n_cells == 0:
        return None

    # Keep points below the upper boundary by reversing the plane normal.
    clipped = _clip_keep_positive(
        clipped,
        origin=upper_origin,
        normal=-normal,
    )

    if clipped.n_cells == 0:
        return None

    clipped = clipped.extract_surface(algorithm=None).triangulate().clean()

    if clipped.n_cells == 0 or clipped.n_points == 0:
        return None

    return clipped


def build_slab_patches(surface, plane_specs, slab_half_width):
    """
    Build one clipped triangular surface patch for every acquisition plane.
    """
    patches = []

    for spec in plane_specs:
        patch = clip_surface_to_slab(
            surface=surface,
            center=spec["center"],
            normal=spec["normal"],
            slab_half_width=slab_half_width,
        )
        patches.append(patch)

    return patches


def point_to_patch_distances(query_points, patch):
    """
    Euclidean point-to-triangle distance to an open triangular surface patch.

    PyVista/VTK evaluates the distance to polygonal cells, not only to mesh
    vertices. The absolute value is used because the slab patch is open and its
    local sign is not meaningful.
    """
    query_points = np.asarray(query_points, dtype=float)

    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError(
            f"query_points must have shape (N, 3), got {query_points.shape}."
        )

    if query_points.shape[0] == 0:
        return np.empty(0, dtype=float)

    if patch is None or patch.n_cells == 0:
        return np.full(query_points.shape[0], np.nan, dtype=float)

    cloud = pv.PolyData(query_points)

    evaluated = cloud.compute_implicit_distance(
        patch,
        inplace=False,
    )

    distances = np.asarray(
        evaluated.point_data["implicit_distance"],
        dtype=float,
    )

    return np.abs(distances)


# ============================================================
# SIGN
# ============================================================

def compute_sign_libigl(mesh, query_points):
    """
    Return +1 outside and -1 inside the complete surface.

    The sign is calibrated using a point far outside the surface bounding box.
    """
    query_points = np.asarray(query_points, dtype=float)

    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError(
            f"query_points must have shape (N, 3), got {query_points.shape}."
        )

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
    extent = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
        1.0,
    )

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
        raise RuntimeError(
            "Could not determine the outside sign for the surface."
        )

    if outside_sign < 0.0:
        sign *= -1.0

    return sign


# ============================================================
# SAMPLING
# ============================================================

def estimate_max_points_in_square(side_length, min_dist):
    """
    Upper estimate based on hexagonal packing.
    """
    if side_length <= 0.0:
        raise ValueError("side_length must be positive.")
    if min_dist <= 0.0:
        raise ValueError("min_dist must be positive.")

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
    """
    Uniform rejection sampling in an oriented square with a minimum mutual
    in-plane distance.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive.")

    rng = np.random.default_rng(seed)

    square_center = np.asarray(square_center, dtype=float)
    u = normalize(u, "sampling axis u")
    v = normalize(v, "sampling axis v")

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
        candidate_2d = rng.uniform(-half, half, size=2)

        if points_2d:
            existing_2d = np.asarray(points_2d)
            distances = np.linalg.norm(
                existing_2d - candidate_2d,
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

        points_2d.append(candidate_2d)
        points_3d.append(candidate_3d)

        trials += 1

    if len(points_3d) < n_points:
        raise RuntimeError(
            f"Uniform sampling failed: generated only "
            f"{len(points_3d)} / {n_points} after {max_trials} trials."
        )

    return np.asarray(points_3d)


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
    Sample points on the central slice plane.

    A candidate is accepted when:
    - it lies inside the complete epicardial volume; or
    - its Euclidean distance to the epicardial triangular patch contained in
      the slice slab is <= surface_expansion.

    The second condition is a finite-thickness generalization of a
    point-to-contour distance. The distance is computed to triangles, not only
    to original mesh vertices.

    If the epicardium does not intersect the slab, uniform sampling is used.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if surface_expansion < 0.0:
        raise ValueError("surface_expansion must be non-negative.")

    n_max_estimated = estimate_max_points_in_square(
        side_length=side_length,
        min_dist=min_dist,
    )

    if n_points > n_max_estimated:
        raise ValueError(
            f"Requested {n_points} points, but only about "
            f"{n_max_estimated} can fit with min_dist={min_dist:.6f}."
        )

    if epi_slab_patch is None or epi_slab_patch.n_cells == 0:
        print(
            "WARNING: epicardium does not intersect this slab. "
            "Using uniform sampling in the square."
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

    square_center = np.asarray(square_center, dtype=float)
    u = normalize(u, "sampling axis u")
    v = normalize(v, "sampling axis v")

    half = side_length / 2.0

    points_2d = []
    points_3d = []

    trials = 0

    while len(points_3d) < n_points and trials < max_trials:
        current_batch = min(batch_size, max_trials - trials)

        candidate_2d_batch = rng.uniform(
            -half,
            half,
            size=(current_batch, 2),
        )

        candidate_3d_batch = (
            square_center[None, :]
            + candidate_2d_batch[:, 0:1] * u[None, :]
            + candidate_2d_batch[:, 1:2] * v[None, :]
        )

        sign_epi = compute_sign_libigl(
            mesh=epi_mesh,
            query_points=candidate_3d_batch,
        )

        inside_epi = sign_epi < 0.0

        distance_to_slab_surface = point_to_patch_distances(
            query_points=candidate_3d_batch,
            patch=epi_slab_patch,
        )

        near_slab_surface = (
            np.isfinite(distance_to_slab_surface)
            & (distance_to_slab_surface <= surface_expansion)
        )

        valid_region = inside_epi | near_slab_surface

        valid_2d = candidate_2d_batch[valid_region]
        valid_3d = candidate_3d_batch[valid_region]

        for candidate_2d, candidate_3d in zip(valid_2d, valid_3d):
            if points_2d:
                existing_2d = np.asarray(points_2d)

                distances = np.linalg.norm(
                    existing_2d - candidate_2d,
                    axis=1,
                )

                if np.any(distances < min_dist):
                    continue

            points_2d.append(candidate_2d)
            points_3d.append(candidate_3d)

            if len(points_3d) >= n_points:
                break

        trials += current_batch

    if len(points_3d) < n_points:
        raise RuntimeError(
            f"Slab-aware sampling failed: generated only "
            f"{len(points_3d)} / {n_points}. "
            "Increase contour_expansion_mm, reduce min_dist_mm, "
            "increase max_sampling_trials, or reduce n_points_per_square."
        )

    return np.asarray(points_3d)


# Backward-compatible alias.
sample_inside_epi_or_near_contour_min_dist = (
    sample_inside_epi_or_near_slab_surface_min_dist
)


# ============================================================
# SLAB-RESTRICTED SIGNED DISTANCE
# ============================================================

def signed_multi_plane_sdf(
    query_points,
    surface,
    plane_ids,
    slab_patches,
):
    """
    Compute a signed distance for samples belonging to multiple MRI slices.

    Magnitude:
        Euclidean distance from each sample to the triangular surface patch
        contained in its own finite-thickness slab.

    Sign:
        Inside/outside sign with respect to the complete cardiac surface.

    Samples whose corresponding slab does not intersect the surface receive
    NaN. These NaNs are subsequently converted to zero and masked out.
    """
    query_points = np.asarray(query_points, dtype=float)
    plane_ids = np.asarray(plane_ids, dtype=int)

    if query_points.shape[0] != plane_ids.shape[0]:
        raise ValueError(
            "query_points and plane_ids must contain the same number of rows."
        )

    unsigned_distance = np.full(
        query_points.shape[0],
        np.nan,
        dtype=float,
    )

    for plane_id in np.unique(plane_ids):
        if plane_id < 0 or plane_id >= len(slab_patches):
            raise IndexError(f"Invalid plane_id: {plane_id}")

        indices = np.where(plane_ids == plane_id)[0]
        patch = slab_patches[plane_id]

        if patch is None or patch.n_cells == 0:
            continue

        unsigned_distance[indices] = point_to_patch_distances(
            query_points=query_points[indices],
            patch=patch,
        )

    global_sign = compute_sign_libigl(
        mesh=surface,
        query_points=query_points,
    )

    return unsigned_distance * global_sign


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

    for surface_path in [lv_path, rv_path, epi_path]:
        if not surface_path.exists():
            raise FileNotFoundError(surface_path)

    lv_raw = pv.read(lv_path)
    rv_raw = pv.read(rv_path)
    epi_raw = pv.read(epi_path)

    if "scale-tooriginalrange" not in epi_raw.field_data:
        raise KeyError(
            f"'scale-tooriginalrange' missing from {epi_path}"
        )

    scale_values = np.asarray(
        epi_raw.field_data["scale-tooriginalrange"]
    ).ravel()

    if scale_values.size == 0:
        raise ValueError(
            f"'scale-tooriginalrange' is empty in {epi_path}"
        )

    scale_to_original_um = float(scale_values[0])

    if not np.isfinite(scale_to_original_um) or scale_to_original_um <= 0.0:
        raise ValueError(
            f"Invalid scale-tooriginalrange: {scale_to_original_um}"
        )

    scale_to_original_mm = scale_to_original_um / 1000.0

    lv = prepare_surface(lv_raw, f"{patient} LV endocardium")
    rv = prepare_surface(rv_raw, f"{patient} RV endocardium")
    epi = prepare_surface(epi_raw, f"{patient} epicardium")

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

    surface_expansion = (
        params.contour_expansion_mm
        / scale_to_original_mm
    )

    plane_23_shift = (
        params.plane_23_shift_mm
        / scale_to_original_mm
    )

    # The long-axis slices are translated along the anatomical e1 axis.
    c_long = c_area + plane_23_shift * e1

    # --------------------------------------------------------
    # Estimate a common square side from true slab-clipped
    # epicardial patches.
    # --------------------------------------------------------

    reference_plane_specs = [
        {
            "type": "short_axis_reference",
            "center": c_area,
            "normal": e1,
            "u": e2,
            "v": e3,
        },
        {
            "type": "normal_e2_reference",
            "center": c_long,
            "normal": e2,
            "u": e1,
            "v": e3,
        },
        {
            "type": "normal_e3_reference",
            "center": c_long,
            "normal": e3,
            "u": e1,
            "v": e2,
        },
    ]

    reference_epi_patches = build_slab_patches(
        surface=epi,
        plane_specs=reference_plane_specs,
        slab_half_width=slab_half_width,
    )

    reference_points = [
        np.empty((0, 3), dtype=float)
        if patch is None
        else np.asarray(patch.points)
        for patch in reference_epi_patches
    ]

    _, short_range_e2, short_range_e3 = max_in_plane_extent(
        reference_points[0],
        e2,
        e3,
    )

    _, normal_e2_range_e1, normal_e2_range_e3 = max_in_plane_extent(
        reference_points[1],
        e1,
        e3,
    )

    _, normal_e3_range_e1, normal_e3_range_e2 = max_in_plane_extent(
        reference_points[2],
        e1,
        e2,
    )

    all_ranges = np.array([
        short_range_e2,
        short_range_e3,
        normal_e2_range_e1,
        normal_e2_range_e3,
        normal_e3_range_e1,
        normal_e3_range_e2,
    ], dtype=float)

    finite_positive_ranges = all_ranges[
        np.isfinite(all_ranges) & (all_ranges > 0.0)
    ]

    if finite_positive_ranges.size == 0:
        raise ValueError(
            "Cannot estimate square side: all slab-based ranges are zero."
        )

    max_range = float(finite_positive_ranges.max())
    square_side = params.square_margin_factor * max_range

    # --------------------------------------------------------
    # Build all acquisition planes.
    # --------------------------------------------------------

    short_axis_plane_points = make_parallel_plane_points(
        start_point=c_area,
        apex_point=a_maxd,
        axis=e1,
        spacing=square_spacing,
        n_before_start=params.n_before_mitral,
        n_after_apex=params.n_after_apex,
    )

    plane_specs = []

    for center in short_axis_plane_points:
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

    # Build patches once and reuse them for sampling and SDF calculation.
    epi_slab_patches = build_slab_patches(
        surface=epi,
        plane_specs=plane_specs,
        slab_half_width=slab_half_width,
    )

    lv_slab_patches = build_slab_patches(
        surface=lv,
        plane_specs=plane_specs,
        slab_half_width=slab_half_width,
    )

    rv_slab_patches = build_slab_patches(
        surface=rv,
        plane_specs=plane_specs,
        slab_half_width=slab_half_width,
    )

    # --------------------------------------------------------
    # Sampling.
    # --------------------------------------------------------

    all_points = []
    all_plane_ids = []
    all_plane_type_ids = []

    type_to_id = {
        "short_axis": 0,
        "normal_e2": 1,
        "normal_e3": 2,
    }

    for plane_id, spec in enumerate(plane_specs):
        points_i = sample_inside_epi_or_near_slab_surface_min_dist(
            square_center=spec["center"],
            u=spec["u"],
            v=spec["v"],
            side_length=square_side,
            epi_mesh=epi,
            epi_slab_patch=epi_slab_patches[plane_id],
            n_points=params.n_points_per_square,
            min_dist=min_dist,
            surface_expansion=surface_expansion,
            seed=42 + plane_id,
            max_trials=params.max_sampling_trials,
            batch_size=params.batch_size,
        )

        all_points.append(points_i)

        all_plane_ids.append(
            np.full(
                points_i.shape[0],
                plane_id,
                dtype=int,
            )
        )

        all_plane_type_ids.append(
            np.full(
                points_i.shape[0],
                type_to_id[spec["type"]],
                dtype=int,
            )
        )

    points = np.vstack(all_points)
    plane_ids = np.concatenate(all_plane_ids)
    plane_type_ids = np.concatenate(all_plane_type_ids)

    # --------------------------------------------------------
    # Slab-restricted signed distances.
    # --------------------------------------------------------

    sdf_epi_raw = signed_multi_plane_sdf(
        query_points=points,
        surface=epi,
        plane_ids=plane_ids,
        slab_patches=epi_slab_patches,
    )

    sdf_lv_raw = signed_multi_plane_sdf(
        query_points=points,
        surface=lv,
        plane_ids=plane_ids,
        slab_patches=lv_slab_patches,
    )

    sdf_rv_raw = signed_multi_plane_sdf(
        query_points=points,
        surface=rv,
        plane_ids=plane_ids,
        slab_patches=rv_slab_patches,
    )

    mask_epi = np.isfinite(sdf_epi_raw).astype(float)
    mask_lv = np.isfinite(sdf_lv_raw).astype(float)
    mask_rv = np.isfinite(sdf_rv_raw).astype(float)

    sdf_epi = np.nan_to_num(
        sdf_epi_raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sdf_lv = np.nan_to_num(
        sdf_lv_raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sdf_rv = np.nan_to_num(
        sdf_rv_raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    samples = np.column_stack([
        points,
        sdf_epi,
        sdf_lv,
        sdf_rv,
        mask_epi,
        mask_lv,
        mask_rv,
    ])

    if not np.all(np.isfinite(samples)):
        raise RuntimeError(
            f"Non-finite values remain in the final samples for {patient}."
        )

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
        "slab_width_mm": params.slab_width_mm,
        "mask_epi_count": int(mask_epi.sum()),
        "mask_lv_count": int(mask_lv.sum()),
        "mask_rv_count": int(mask_rv.sum()),
        "mask_epi_fraction": float(mask_epi.mean()),
        "mask_lv_fraction": float(mask_lv.mean()),
        "mask_rv_fraction": float(mask_rv.mean()),
        "epi_valid_planes": int(
            sum(patch is not None for patch in epi_slab_patches)
        ),
        "lv_valid_planes": int(
            sum(patch is not None for patch in lv_slab_patches)
        ),
        "rv_valid_planes": int(
            sum(patch is not None for patch in rv_slab_patches)
        ),
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
        "c_long": c_long,
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "plane_specs": plane_specs,
        "short_axis_plane_points": short_axis_plane_points,
        "epi_slab_patches": epi_slab_patches,
        "lv_slab_patches": lv_slab_patches,
        "rv_slab_patches": rv_slab_patches,
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

    epi_slab_patches = debug["epi_slab_patches"]

    square_side = debug["stats"]["square_side_norm"]

    plotter = pv.Plotter()

    plotter.add_mesh(
        epi,
        color="lightgray",
        opacity=0.10,
        label="Epicardium",
    )

    plotter.add_mesh(
        lv,
        color="red",
        opacity=0.10,
        label="LV endocardium",
    )

    plotter.add_mesh(
        rv,
        color="blue",
        opacity=0.10,
        label="RV endocardium",
    )

    plotter.add_mesh(
        pv.Sphere(radius=0.03, center=c_area),
        color="magenta",
        label="Mitral centroid",
    )

    plotter.add_mesh(
        pv.Sphere(radius=0.03, center=a_maxd),
        color="black",
        label="Apex",
    )

    plotter.add_mesh(
        pv.Sphere(radius=0.03, center=t_area),
        color="cyan",
        label="Tricuspid centroid",
    )

    plane_colors = {
        "short_axis": "green",
        "normal_e2": "orange",
        "normal_e3": "purple",
    }

    for plane_id, spec in enumerate(plane_specs):
        square = make_square(
            spec["center"],
            spec["u"],
            spec["v"],
            square_side,
        )

        color = plane_colors[spec["type"]]
        opacity = 0.05 if spec["type"] == "short_axis" else 0.18

        plotter.add_mesh(
            square,
            color=color,
            opacity=opacity,
            show_edges=True,
        )

        patch = epi_slab_patches[plane_id]

        if patch is not None:
            plotter.add_mesh(
                patch,
                color=color,
                opacity=0.30,
                show_edges=False,
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

    plotter.add_legend()
    plotter.show()


if __name__ == "__main__":
    pass