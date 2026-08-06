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


il lato di ogni quadrato è 1.5 volte il massimo tra le distanze massime delle proiezini dei punti nelle slice rispetto agili altri due assi, per ogni combianzione.

Ora costruiamo anche tutti gli altri piani paralleli al piano 1 e distanti 6mm. ne mettiamo anche 3 prima di quello sul centroide della mitrale e  2 oltre l'apice.
samplaimo anche con distanza minima di 1 mm tra punti e al più a 35mm dalla intersezione epicardio-slice in direzione transmurale;
se non esiste intersezione epicardio-slice, sampliamo uniformemente a distanza minima di 1 mm.
"""


from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv

import igl

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
patient = "LEU_BBB_21248"
# # patient = "yrm9981_v1"
# patient = "LEU_NORM_0194"
# patient = "S65"

# ============================================================
# PARAMETRI
# ============================================================

# SQUARE_SIDE = 3.0
AXIS_LENGTH = 4.0
SQUARE_MARGIN_FACTOR = 1.5

SLICE_THICKNESS_MM = 0.75

plane_23_shitf_mm = 25.0

SQUARE_SPACING_MM = 6

N_BEFORE_MITRAL = 4
N_AFTER_APEX = 3

# N_BEFORE_MITRAL = 2
# N_AFTER_APEX = 2

N_SAMPLED_POINTS_PER_SQUARE = 1000  
MIN_DIST_MM = 1

CONTOUR_EXPANSION_MM = 25.0
BATCH_SIZE = 5000

# ============================================================
# PARAMETRI SETTORE ECOGRAFICO / TRIANGOLO CURVILINEO
# ============================================================

ECHO_SPHERE_RADIUS = 1.5
ECHO_ANGLE_DEG = 60.0
ECHO_ARC_RESOLUTION = 150

# Centro della sfera sulla quale si trova la sonda
ECHO_SPHERE_CENTER = np.array([0.0, 0.0, 0.0])

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


def max_in_plane_extent(points, axis_a, axis_b):
    """
    Massima estensione dei punti lungo i due assi del piano.
    """

    if points.shape[0] == 0:
        return 0.0

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
    n_after_apex=2,
):
    """
    Genera punti lungo l'asse start_point -> apex_point.

    Include:
    - n_before_start piani prima di start_point
    - piani da start_point verso apex_point
    - n_after_apex piani oltre apex_point
    """

    start_point = np.asarray(start_point, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    axis = normalize(axis)

    axis_length = np.linalg.norm(apex_point - start_point)

    if spacing <= 0:
        raise ValueError("Spacing must be positive")

    n_full_steps = int(np.floor(axis_length / spacing))

    plane_points = []

    # piani prima della mitrale
    for i in range(n_before_start, 0, -1):
        plane_points.append(
            start_point - i * spacing * axis
        )

    # piani dalla mitrale verso apex
    for i in range(n_full_steps + 1):
        plane_points.append(
            start_point + i * spacing * axis
        )

    # piani oltre l'apex
    last_distance = n_full_steps * spacing

    for i in range(1, n_after_apex + 1):
        plane_points.append(
            start_point + (last_distance + i * spacing) * axis
        )

    return np.asarray(plane_points)

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

        if len(points_2d) == 0:
            accept = True
        else:
            existing_2d = np.asarray(points_2d)

            dists = np.linalg.norm(
                existing_2d - candidate_2d,
                axis=1,
            )

            accept = np.all(dists >= min_dist)

        if accept:
            candidate_3d = (
                square_center
                + a * u
                + b * v
            )

            points_2d.append(candidate_2d)
            points_3d.append(candidate_3d)

        trials += 1

    if len(points_3d) < n_points:
        raise RuntimeError(
            f"Sampling failed: generated only "
            f"{len(points_3d)} / {n_points} "
            f"after {max_trials} trials."
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
    Campiona punti nel quadrato accettando un punto se:
    - è dentro l'epicardio
      oppure
    - è entro contour_expansion dalla traccia 2D dell'epicardio nel piano

    Se non ci sono punti epicardici nella slice, oppure nessuno cade dentro
    il quadrato, usa sampling uniforme classico.

    Inoltre impone distanza minima reciproca tra i sample.
    """

    if epi_slice_points.shape[0] == 0:
        print(
            "WARNING: no epicardial points in this slice. "
            "Using uniform sampling without boundary."
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
        print(
            "WARNING: no epicardial contour points inside this square. "
            "Using uniform sampling without boundary."
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
    
    if epi_slice_points.shape[0] == 0:
        print(
            "WARNING: no epicardial points in this slice. "
            "Using uniform sampling without boundary."
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

    points_2d = []
    points_3d = []

    trials = 0

    while len(points_3d) < n_points and trials < max_trials:

        current_batch = min(
            batch_size,
            max_trials - trials
        )

        # ----------------------------------------------------
        # Genero candidati uniformi nel quadrato
        # ----------------------------------------------------

        ab = rng.uniform(
            -half,
            half,
            size=(current_batch, 2)
        )

        candidate_3d = (
            square_center[None, :]
            + ab[:, 0:1] * u[None, :]
            + ab[:, 1:2] * v[None, :]
        )

        # ----------------------------------------------------
        # Condizione 1: dentro epicardio
        # ----------------------------------------------------

        sign_epi = compute_sign_libigl(
            mesh=epi_mesh,
            query_points=candidate_3d,
        )

        inside_epi = sign_epi < 0

        # ----------------------------------------------------
        # Condizione 2: vicino al contour epi nel piano
        # ----------------------------------------------------

        near_contour = np.zeros(
            current_batch,
            dtype=bool
        )

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

        # ----------------------------------------------------
        # Distanza minima tra samples
        # ----------------------------------------------------

        for candidate_2d, candidate_point_3d in zip(
            valid_ab,
            valid_3d,
        ):

            if len(points_2d) == 0:
                accept = True
            else:
                existing_2d = np.asarray(points_2d)

                dists_samples = np.linalg.norm(
                    existing_2d - candidate_2d,
                    axis=1,
                )

                accept = np.all(
                    dists_samples >= min_dist
                )

            if accept:
                points_2d.append(candidate_2d)
                points_3d.append(candidate_point_3d)

            if len(points_3d) >= n_points:
                break

        trials += current_batch

    if len(points_3d) < n_points:
        raise RuntimeError(
            f"Sampling failed: generated only "
            f"{len(points_3d)} / {n_points} points. "
            f"Try increasing CONTOUR_EXPANSION_MM, "
            f"reducing MIN_DIST_MM, or reducing "
            f"N_SAMPLED_POINTS_PER_SQUARE."
        )

    return np.asarray(points_3d)

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

def rotate_vector_in_plane(
    bisector_direction,
    lateral_direction,
    angle_rad,
):
    """
    Ruota bisector_direction nel piano generato da
    bisector_direction e lateral_direction.

    Entrambi i vettori devono essere ortonormali.
    """

    bisector_direction = normalize(
        np.asarray(bisector_direction, dtype=float)
    )

    lateral_direction = normalize(
        np.asarray(lateral_direction, dtype=float)
    )

    return normalize(
        np.cos(angle_rad) * bisector_direction
        + np.sin(angle_rad) * lateral_direction
    )


def second_ray_sphere_intersection(
    vertex,
    ray_direction,
    sphere_center,
    sphere_radius,
    tolerance=1e-10,
):
    """
    Trova la seconda intersezione tra una semiretta e una sfera.

    La semiretta parte da vertex, che deve trovarsi sulla sfera:

        x(t) = vertex + t * ray_direction

    L'intersezione con t=0 è vertex.
    La funzione restituisce l'altra intersezione.
    """

    vertex = np.asarray(vertex, dtype=float)
    ray_direction = normalize(
        np.asarray(ray_direction, dtype=float)
    )
    sphere_center = np.asarray(sphere_center, dtype=float)

    relative_vertex = vertex - sphere_center

    vertex_radius = np.linalg.norm(relative_vertex)

    if not np.isclose(
        vertex_radius,
        sphere_radius,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "The sector vertex is not on the sphere: "
            f"|V-C|={vertex_radius:.8f}, "
            f"radius={sphere_radius:.8f}"
        )

    # Risolvendo l'intersezione:
    #
    # |V + t*d - C|² = r²
    #
    # dato che |V-C|² = r²:
    #
    # t [t + 2(V-C)·d] = 0
    #
    # t=0 è V; l'altra soluzione è:
    second_t = -2.0 * np.dot(
        relative_vertex,
        ray_direction,
    )

    if second_t <= tolerance:
        raise ValueError(
            "The ray does not point toward the interior of the sphere. "
            f"Second intersection parameter t={second_t:.8f}"
        )

    return vertex + second_t * ray_direction


def make_curvilinear_triangle_on_sphere(
    sphere_center,
    sphere_radius,
    vertex_direction,
    lateral_direction,
    opening_angle_deg,
    arc_resolution=100,
):
    """
    Costruisce un triangolo curvilineo piano inscritto nella sfera.

    Il triangolo è delimitato da:
    - due segmenti con origine nel vertice V;
    - un arco della circonferenza ottenuta intersecando la sfera
      con il piano di acquisizione.

    Parametri
    ---------
    sphere_center:
        Centro della sfera.

    sphere_radius:
        Raggio della sfera.

    vertex_direction:
        Direzione dal centro della sfera verso il vertice V.

    lateral_direction:
        Secondo asse appartenente al piano di acquisizione.

    opening_angle_deg:
        Angolo totale al vertice.

    arc_resolution:
        Numero di punti utilizzati per discretizzare l'arco.

    Returns
    -------
    sector:
        Superficie PyVista triangolata.

    boundary:
        Bordo completo del triangolo curvilineo.

    vertex:
        Vertice V sulla sfera.

    intersection_1, intersection_2:
        Intersezioni dei lati con la sfera.
    """

    sphere_center = np.asarray(
        sphere_center,
        dtype=float,
    )

    vertex_direction = normalize(
        np.asarray(vertex_direction, dtype=float)
    )

    # Rendiamo lateral_direction ortogonale a vertex_direction
    lateral_direction = np.asarray(
        lateral_direction,
        dtype=float,
    )

    lateral_direction = (
        lateral_direction
        - np.dot(
            lateral_direction,
            vertex_direction,
        ) * vertex_direction
    )

    lateral_direction = normalize(lateral_direction)

    alpha = np.deg2rad(opening_angle_deg)
    beta = alpha / 2.0

    if not 0.0 < alpha < np.pi:
        raise ValueError(
            "opening_angle_deg must be between 0 and 180 degrees."
        )

    # Vertice sulla sfera
    vertex = (
        sphere_center
        + sphere_radius * vertex_direction
    )

    # La bisettrice è orientata dal vertice verso il centro
    bisector_direction = normalize(
        sphere_center - vertex
    )

    # Direzioni dei due lati
    ray_direction_1 = rotate_vector_in_plane(
        bisector_direction=bisector_direction,
        lateral_direction=lateral_direction,
        angle_rad=-beta,
    )

    ray_direction_2 = rotate_vector_in_plane(
        bisector_direction=bisector_direction,
        lateral_direction=lateral_direction,
        angle_rad=beta,
    )

    intersection_1 = second_ray_sphere_intersection(
        vertex=vertex,
        ray_direction=ray_direction_1,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
    )

    intersection_2 = second_ray_sphere_intersection(
        vertex=vertex,
        ray_direction=ray_direction_2,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
    )

    # Il punto diametralmente opposto a V è nella direzione
    # della bisettrice rispetto al centro.
    arc_axis = bisector_direction

    # Gli estremi dell'arco risultano agli angoli -alpha e +alpha
    # rispetto all'asse arc_axis.
    arc_angles = np.linspace(
        -alpha,
        alpha,
        arc_resolution,
    )

    arc_points = (
        sphere_center[None, :]
        + sphere_radius
        * (
            np.cos(arc_angles)[:, None]
            * arc_axis[None, :]
            + np.sin(arc_angles)[:, None]
            * lateral_direction[None, :]
        )
    )

    # Forziamo esattamente i punti estremi, per evitare
    # piccoli errori numerici.
    arc_points[0] = intersection_1
    arc_points[-1] = intersection_2

    # Punto 0 = vertice
    # Punti 1...N = arco
    points = np.vstack([
        vertex,
        arc_points,
    ])

    # Ventaglio di triangoli:
    # V, arco_i, arco_i+1
    faces = []

    for i in range(arc_resolution - 1):
        faces.extend([
            3,
            0,
            i + 1,
            i + 2,
        ])

    sector = pv.PolyData(
        points,
        np.asarray(faces, dtype=np.int64),
    )

    sector = sector.clean().triangulate()

    # Bordo: V -> P1 -> arco -> P2 -> V
    boundary_points = np.vstack([
        vertex,
        arc_points,
        vertex,
    ])

    boundary = pv.lines_from_points(
        boundary_points,
        close=False,
    )

    return (
        sector,
        boundary,
        vertex,
        intersection_1,
        intersection_2,
    )


def intersect_surface_with_sector(
    anatomical_surface,
    sector,
):
    """
    Calcola la curva di intersezione tra una superficie anatomica
    chiusa e il triangolo curvilineo.

    Restituisce None se non viene trovata alcuna intersezione.
    """

    if anatomical_surface is None:
        return None

    anatomical_surface = (
        anatomical_surface
        .extract_surface()
        .triangulate()
        .clean()
    )

    sector_clean = (
        sector
        .extract_surface()
        .triangulate()
        .clean()
    )

    try:
        intersection, _, _ = anatomical_surface.intersection(
            sector_clean,
            split_first=False,
            split_second=False,
        )
    except Exception as error:
        print(
            "WARNING: sector/surface intersection failed:",
            error,
        )
        return None

    if intersection.n_points == 0:
        return None

    return intersection.clean()
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
# CENTRO DEL PIANO SHORT-AXIS
# ============================================================
# Il piano short-axis resta normale a e1.
# Però il suo centro non è C_area.
#
# Vogliamo il punto sull'asse e1 passante per l'origine
# che abbia la stessa quota/proiezione di C_area lungo e1.

C_short = np.dot(C_area, e1) * e1

print("\nShort-axis center")
print("C_area:", C_area)
print("C_short:", C_short)
print("projection C_area on e1:", np.dot(C_area, e1))
print("projection C_short on e1:", np.dot(C_short, e1))

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

square_spacing_norm = SQUARE_SPACING_MM / scale_to_original_mm

print("square_spacing_mm:", SQUARE_SPACING_MM)
print("square_spacing_norm:", square_spacing_norm)

print("\nScale")
print("scale_to_original_range:", scale_to_original_range)
print("scale_to_original_mm:", scale_to_original_mm)
print("slice_thickness_mm:", SLICE_THICKNESS_MM)
print("slice_thickness_norm:", slice_thickness_norm)
print("slice_half_thickness_norm:", slice_half_thickness_norm)

min_dist_norm = MIN_DIST_MM / scale_to_original_mm

print("\nSampling parameters")
print("N_SAMPLED_POINTS_PER_SQUARE:", N_SAMPLED_POINTS_PER_SQUARE)
print("MIN_DIST_MM:", MIN_DIST_MM)
print("min_dist_norm:", min_dist_norm)


# shifting dei piani 2 e 3 
plane_23_shift_norm = plane_23_shitf_mm / scale_to_original_mm

C_long = C_area + plane_23_shift_norm * e1

print("\nPlane 2/3 shift")
print("PLANE_23_SHIFT_MM:", plane_23_shitf_mm)
print("plane_23_shift_norm:", plane_23_shift_norm)
print("C_area:", C_area)
print("C_long:", C_long)

#---

contour_expansion_norm = CONTOUR_EXPANSION_MM / scale_to_original_mm

print("CONTOUR_EXPANSION_MM:", CONTOUR_EXPANSION_MM)
print("contour_expansion_norm:", contour_expansion_norm)

#-------

epi_points = epi.points

# distanza signed dai tre piani
# ogni piano passa per C_area
# normale piano short-axis = e1
# normale piano normal-e2 = e2
# normale piano normal-e3 = e3

# d_short = np.dot(
#     epi_points - C_area,
#     e1
# )

d_short = np.dot(
    epi_points - C_short,
    e1
)

# d_e2 = np.dot(
#     epi_points - C_area,
#     e2
# )

# d_e3 = np.dot(
#     epi_points - C_area,
#     e3
# )

# d_e2 = np.dot(epi_points - C_area, e2)
# d_e3 = np.dot(epi_points - C_area, e3)

d_e2 = np.dot(epi_points - C_long, e2)
d_e3 = np.dot(epi_points - C_long, e3)

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
# CALCOLO LATO QUADRATI
# ============================================================
# Per ogni slice proiettiamo i punti sui due assi che generano il piano.
#
# Piano short-axis: normale e1 -> assi nel piano: e2, e3
# Piano normal-e2:  normale e2 -> assi nel piano: e1, e3
# Piano normal-e3:  normale e3 -> assi nel piano: e1, e2
#
# Otteniamo 6 range:
# short: e2, e3
# normal-e2: e1, e3
# normal-e3: e1, e2
#
# Il lato comune dei quadrati è:
# SQUARE_SIDE = 1.5 * max(dei 6 range)

extent_short, short_range_e2, short_range_e3 = max_in_plane_extent(
    epi_points_short,
    e2,
    e3
)

extent_normal_e2, normal_e2_range_e1, normal_e2_range_e3 = max_in_plane_extent(
    epi_points_e2,
    e1,
    e3
)

extent_normal_e3, normal_e3_range_e1, normal_e3_range_e2 = max_in_plane_extent(
    epi_points_e3,
    e1,
    e2
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

SQUARE_SIDE = SQUARE_MARGIN_FACTOR * max_range

print("\nSquare side estimation")

print("Short-axis slice:")
print("  range on e2:", short_range_e2)
print("  range on e3:", short_range_e3)

print("Normal-e2 slice:")
print("  range on e1:", normal_e2_range_e1)
print("  range on e3:", normal_e2_range_e3)

print("Normal-e3 slice:")
print("  range on e1:", normal_e3_range_e1)
print("  range on e2:", normal_e3_range_e2)

print("max_range:", max_range)
print("SQUARE_MARGIN_FACTOR:", SQUARE_MARGIN_FACTOR)
print("SQUARE_SIDE:", SQUARE_SIDE)


# ============================================================
# COSTRUISCI QUADRATI / PIANI
# ============================================================
origin = (0,0,0)

center = C_long

# A. Short-axis: normale e1, quindi piano generato da e2, e3
square_short_axis = make_square(
    C_short,
    e2,
    e3,
    SQUARE_SIDE
)

# B. Piano normale a e2, generato da e1, e3
square_normal_e2 = make_square(
    C_long,
    e1,
    e3,
    SQUARE_SIDE
)

# C. Piano normale a e3, generato da e1, e2
square_normal_e3 = make_square(
    C_long,
    e1,
    e2,
    SQUARE_SIDE
)

# ============================================================
# TRIANGOLO CURVILINEO ECOGRAFICO
# ============================================================

# Posizioniamo il vertice sul lato opposto all'apice.
#
# e1 punta dalla mitrale verso l'apice.
# Quindi -e1 porta il vertice verso la regione basale/esterna.
echo_vertex_direction = -e1

# Il secondo asse del piano di acquisizione è e2.
# Il triangolo curvilineo giace quindi nel piano e1-e2,
# normale a e3.
echo_lateral_direction = e2

(
    echo_sector,
    echo_sector_boundary,
    echo_vertex,
    echo_intersection_1,
    echo_intersection_2,
) = make_curvilinear_triangle_on_sphere(
    sphere_center=ECHO_SPHERE_CENTER,
    sphere_radius=ECHO_SPHERE_RADIUS,
    vertex_direction=echo_vertex_direction,
    lateral_direction=echo_lateral_direction,
    opening_angle_deg=ECHO_ANGLE_DEG,
    arc_resolution=ECHO_ARC_RESOLUTION,
)

print("\nEcho curvilinear triangle")
print("Vertex:", echo_vertex)
print("Intersection 1:", echo_intersection_1)
print("Intersection 2:", echo_intersection_2)
print(
    "Distance V-P1:",
    np.linalg.norm(echo_intersection_1 - echo_vertex),
)
print(
    "Distance V-P2:",
    np.linalg.norm(echo_intersection_2 - echo_vertex),
)

# Intersezioni con le tre superfici
echo_epi_intersection = intersect_surface_with_sector(
    anatomical_surface=epi,
    sector=echo_sector,
)

echo_lv_intersection = intersect_surface_with_sector(
    anatomical_surface=lv,
    sector=echo_sector,
)

echo_rv_intersection = intersect_surface_with_sector(
    anatomical_surface=rv,
    sector=echo_sector,
)

# ============================================================
# PIANI PARALLELI AL PIANO 1 / SHORT-AXIS
# ============================================================

short_axis_plane_points = make_parallel_plane_points(
    start_point=C_area,      # oppure C_area, vedi nota sotto
    apex_point=A_maxD,
    axis=e1,
    spacing=square_spacing_norm,
    n_before_start=N_BEFORE_MITRAL,
    n_after_apex=N_AFTER_APEX,
)

short_axis_squares = []

for plane_point in short_axis_plane_points:

    square = make_square(
        plane_point,
        e2,
        e3,
        SQUARE_SIDE
    )

    short_axis_squares.append(square)

print("\nParallel short-axis planes")
print("Number of short-axis planes:", len(short_axis_squares))
print("Number before mitral:", N_BEFORE_MITRAL)
print("Number after apex:", N_AFTER_APEX)


# ============================================================
# SAMPLING DEI PIANI
# ============================================================

all_sampled_points = []
all_sampled_plane_ids = []
all_sampled_plane_types = []

plane_counter = 0

# ------------------------------------------------------------
# 1. Sampling sui piani short-axis paralleli
# ------------------------------------------------------------

for i, center_i in enumerate(short_axis_plane_points):

    d = np.dot(
    epi_points - center_i,
    e1
    )

    mask = np.abs(d) <= slice_half_thickness_norm

    epi_slice_points_i = epi_points[mask]

    sampled_i = sample_inside_epi_or_near_contour_min_dist(
        square_center=center_i,
        u=e2,
        v=e3,
        side_length=SQUARE_SIDE,
        epi_slice_points=epi_slice_points_i,
        epi_mesh=epi,
        n_points=N_SAMPLED_POINTS_PER_SQUARE,
        min_dist=min_dist_norm,
        contour_expansion=contour_expansion_norm,
        seed=42 + plane_counter,
        batch_size=BATCH_SIZE,
    )

    all_sampled_points.append(sampled_i)

    all_sampled_plane_ids.append(
        np.full(
            sampled_i.shape[0],
            plane_counter,
            dtype=int,
        )
    )

    all_sampled_plane_types.append(
        np.full(
            sampled_i.shape[0],
            "short_axis",
            dtype=object,
        )
    )

    plane_counter += 1


# ------------------------------------------------------------
# 2. Sampling piano normal-e2
# ------------------------------------------------------------

d = np.dot(
    epi_points - C_long,
    e2
)

mask = np.abs(d) <= slice_half_thickness_norm

epi_slice_points_e2 = epi_points[mask]

sampled_e2 = sample_inside_epi_or_near_contour_min_dist(
    square_center=C_long,
    u=e1,
    v=e3,
    side_length=SQUARE_SIDE,
    epi_slice_points=epi_slice_points_e2,
    epi_mesh=epi,
    n_points=N_SAMPLED_POINTS_PER_SQUARE,
    min_dist=min_dist_norm,
    contour_expansion=contour_expansion_norm,
    seed=42 + plane_counter,
    batch_size=BATCH_SIZE,
)

all_sampled_points.append(sampled_e2)

all_sampled_plane_ids.append(
    np.full(
        sampled_e2.shape[0],
        plane_counter,
        dtype=int,
    )
)

all_sampled_plane_types.append(
    np.full(
        sampled_e2.shape[0],
        "normal_e2",
        dtype=object,
    )
)

plane_counter += 1


# ------------------------------------------------------------
# 3. Sampling piano normal-e3
# ------------------------------------------------------------

d = np.dot(
    epi_points - C_long,
    e3
)

mask = np.abs(d) <= slice_half_thickness_norm

epi_slice_points_e3 = epi_points[mask]

sampled_e3 = sample_inside_epi_or_near_contour_min_dist(
    square_center=C_long,
    u=e1,
    v=e2,
    side_length=SQUARE_SIDE,
    epi_slice_points=epi_slice_points_e3,
    epi_mesh=epi,
    n_points=N_SAMPLED_POINTS_PER_SQUARE,
    min_dist=min_dist_norm,
    contour_expansion=contour_expansion_norm,
    seed=42 + plane_counter,
    batch_size=BATCH_SIZE,
)

all_sampled_points.append(sampled_e3)

all_sampled_plane_ids.append(
    np.full(
        sampled_e3.shape[0],
        plane_counter,
        dtype=int,
    )
)

all_sampled_plane_types.append(
    np.full(
        sampled_e3.shape[0],
        "normal_e3",
        dtype=object,
    )
)

plane_counter += 1


# ------------------------------------------------------------
# Stack finale
# ------------------------------------------------------------

all_sampled_points = np.vstack(all_sampled_points)
all_sampled_plane_ids = np.concatenate(all_sampled_plane_ids)
all_sampled_plane_types = np.concatenate(all_sampled_plane_types)

print("\nSampling")
print("Total planes:", plane_counter)
print("Points per plane:", N_SAMPLED_POINTS_PER_SQUARE)
print("Total sampled points:", all_sampled_points.shape[0])

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

# plotter.add_mesh(
#     axis_e1,
#     color="green",
#     line_width=5,
#     label="e1 apex-base"
# )

# plotter.add_mesh(
#     axis_e2,
#     color="orange",
#     line_width=5,
#     label="e2 mitral-tricuspid"
# )

# plotter.add_mesh(
#     axis_e3,
#     color="purple",
#     line_width=5,
#     label="e3 third axis"
# )

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
# for i, square in enumerate(short_axis_squares):

#     opacity = 0.22 if i == N_BEFORE_MITRAL else 0.08
#     if i == 4:
#         plotter.add_mesh(
#             square,
#             color="green",
#             opacity=opacity,
#             show_edges=True,
#         )
#         break
#     else:
#         continue
    

# plotter.add_mesh(
#     square_normal_e2,
#     color="orange",
#     opacity=0.30,
#     show_edges=True,
#     label="Vertical Long-axis Plane normal e2"
# )

# plotter.add_mesh(
#     square_normal_e3,
#     color="purple",
#     opacity=0.30,
#     show_edges=True,
#     label="Horizonral Long-axis Plane normal e3"
# )

# centri dei piani
# plotter.add_points(
#     short_axis_plane_points,
#     color="black",
#     point_size=9,
#     render_points_as_spheres=True,
#     label="Short-axis plane centers"
# )

# punti di intersezione dell'epi con i piani
# plotter.add_points(
#     epi_points_short,
#     color="green",
#     point_size=8,
#     render_points_as_spheres=True,
#     label="Epi points short-axis slice"
# )

# plotter.add_points(
#     epi_points_e2,
#     color="orange",
#     point_size=8,
#     render_points_as_spheres=True,
#     label="Epi points normal-e2 slice"
# )

plotter.add_points(
    epi_points_e3,
    color="purple",
    point_size=8,
    render_points_as_spheres=True,
    label="Epi points normal-e3 slice"
)


# origine
plotter.add_mesh(
    pv.Sphere(radius=0.04, center=(0,0,0)),
    color="red",
    label="origin",
    opacity=0.4
)

plotter.add_mesh(
    pv.Sphere(radius=1.5, center=(0,0,0)),
    color="red",
    label="origin",
    opacity=0.1
)

# samples
# plotter.add_mesh(
#     pv.PolyData(all_sampled_points[all_sampled_plane_types == "short_axis"]),
#     color="green",
#     point_size=4,
#     render_points_as_spheres=True,
# )

# plotter.add_mesh(
#     pv.PolyData(all_sampled_points[all_sampled_plane_types == "normal_e2"]),
#     color="orange",
#     point_size=4,
#     render_points_as_spheres=True,
# )

# plotter.add_mesh(
#     pv.PolyData(all_sampled_points[all_sampled_plane_types == "normal_e3"]),
#     color="purple",
#     point_size=4,
#     render_points_as_spheres=True,
# )

#---------------

# ============================================================
# PLOT TRIANGOLO CURVILINEO
# ============================================================

plotter.add_mesh(
    echo_sector,
    color="gold",
    opacity=0.20,
    show_edges=False,
    label="Echo fan-shaped sector",
)

plotter.add_mesh(
    echo_sector_boundary,
    color="black",
    line_width=5,
    label="Echo sector boundary",
)

# Vertice del settore / posizione della sonda
plotter.add_mesh(
    pv.Sphere(
        radius=0.045,
        center=echo_vertex,
    ),
    color="yellow",
    label="Echo probe vertex",
)

# Intersezioni dei due lati con la sfera
plotter.add_points(
    np.vstack([
        echo_intersection_1,
        echo_intersection_2,
    ]),
    color="black",
    point_size=12,
    render_points_as_spheres=True,
    label="Sector-sphere intersections",
)

# Asse centrale del settore
echo_bisector_line = pv.Line(
    echo_vertex,
    ECHO_SPHERE_CENTER
    + ECHO_SPHERE_RADIUS * e1,
)

plotter.add_mesh(
    echo_bisector_line,
    color="yellow",
    line_width=3,
    label="Echo sector bisector",
)

# ============================================================
# INTERSEZIONI SETTORE - SUPERFICI CARDIACHE
# ============================================================

if (
    echo_epi_intersection is not None
    and echo_epi_intersection.n_points > 0
):
    plotter.add_mesh(
        echo_epi_intersection,
        color="green",
        line_width=8,
        render_lines_as_tubes=True,
        label="Echo sector / epicardium",
    )
else:
    print("No intersection between echo sector and epicardium")

if (
    echo_lv_intersection is not None
    and echo_lv_intersection.n_points > 0
):
    plotter.add_mesh(
        echo_lv_intersection,
        color="red",
        line_width=8,
        render_lines_as_tubes=True,
        label="Echo sector / LV endocardium",
    )
else:
    print("No intersection between echo sector and LV")

if (
    echo_rv_intersection is not None
    and echo_rv_intersection.n_points > 0
):
    plotter.add_mesh(
        echo_rv_intersection,
        color="blue",
        line_width=8,
        render_lines_as_tubes=True,
        label="Echo sector / RV endocardium",
    )
else:
    print("No intersection between echo sector and RV")

plotter.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)

plotter.add_axes()
plotter.add_legend()

plotter.show()