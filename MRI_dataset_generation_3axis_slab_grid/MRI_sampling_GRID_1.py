"""
GRID-BASED MRI-LIKE SAMPLING PER UN SINGOLO PAZIENTE
=====================================================

OBIETTIVO
---------
Generare un file NPY con campioni MRI-like relativi a:

1. una sequenza di piani short-axis;
2. un volume long-axis normale a e2;
3. un volume long-axis normale a e3.

Il file finale ha colonne:

    x, y, z,
    sdf_epi, sdf_lv, sdf_rv,
    mask_epi, mask_lv, mask_rv


REGIONE AMMESSA PER IL SAMPLING
-------------------------------
La regione valida rimane uguale a quella del codice precedente.

Un nodo della griglia viene considerato utilizzabile quando:

    valid = inside_epi OR near_epi_patch

ovvero:

- il nodo si trova all'interno dell'epicardio completo;
- oppure si trova entro `contour_expansion_mm` dalla patch epicardica
  contenuta nello slab relativo al piano.

La presenza di LV e RV NON amplia la regione valida.

Le superfici LV e RV vengono utilizzate solamente per distribuire meglio
i punti validi attorno ai rispettivi contour.


GRIGLIA
-------
Per gli short-axis viene costruita una griglia 2D sul piano:

    p = center + a*u + b*v

con passo:

    grid_spacing_mm = 1.0 mm

La distanza fra due nodi orizzontalmente o verticalmente adiacenti è quindi
esattamente 1 mm. Non è necessario calcolare tutte le distanze reciproche.

Per i volumi long-axis viene costruita una griglia 3D:

    p = center + a*u + b*v + t*normal

sempre con passo di 1 mm.


NUMERO DI PUNTI DISPONIBILI
---------------------------
Dopo aver costruito e filtrato la griglia viene calcolato esattamente:

    n_available = numero di nodi validi

Se:

    n_available <= n_requested

vengono utilizzati tutti i nodi disponibili.

Se:

    n_available > n_requested

vengono selezionati `n_requested` nodi.


DISTRIBUZIONE ATTORNO AI CONTOUR
--------------------------------
Una percentuale del budget viene riservata ai contour delle superfici:

- epicardio;
- endocardio LV;
- endocardio RV.

Per ciascuna superficie si cercano inizialmente nodi entro una banda stretta,
per esempio 2 mm.

Se non sono sufficienti, la banda viene aumentata progressivamente:

    2, 4, 6, 8, 12 mm

La ricerca è sempre limitata ai nodi appartenenti alla regione valida:

    inside_epi OR near_epi_patch

Quindi LV e RV non introducono nuovi nodi esterni alla regione epicardica
ammessa.

Se una superficie non è presente nello slab, oppure non contiene abbastanza
nodi, il suo deficit viene recuperato selezionando nodi dalla restante
griglia valida.


SELEZIONE DISTRIBUITA
---------------------
La selezione dei nodi non avviene prendendo semplicemente i primi elementi.

I candidati vengono ordinati usando una griglia stratificata:

1. le coordinate locali vengono divise in macro-celle;
2. viene scelto inizialmente un nodo per macro-cella;
3. se servono altri punti, vengono aggiunti nodi dalle celle ancora occupate;
4. l'ordine all'interno delle celle viene randomizzato con seed riproducibile.

Questo riduce il rischio di accumulare i punti in una sola regione.


FALLIMENTI
----------
Il metodo non fallisce se ci sono meno nodi del numero richiesto.

In quel caso vengono salvati tutti i nodi disponibili e viene stampato
un warning.

Viene sollevato un errore soltanto se un piano o volume non contiene
alcun nodo valido.


UNITÀ DI MISURA
---------------
Le superfici e i landmark sono espressi nel sistema normalizzato originale.

I parametri espressi in millimetri vengono convertiti usando il campo:

    scale-tooriginalrange

presente nell'epicardio.
"""

from dataclasses import dataclass
from pathlib import Path

import igl
import numpy as np
import pandas as pd
import pyvista as pv

from vtk import vtkClipPolyData, vtkPlane


# ============================================================
# CONFIGURAZIONE
# ============================================================

@dataclass
class GridMRIParams:

    # Distanza fra i piani short-axis
    square_spacing_mm: float = 6.0

    # Larghezza totale dello slab short-axis utilizzato per le SDF
    short_axis_slab_width_mm: float = 0.75

    # Larghezza totale dei volumi long-axis
    long_axis_volume_width_mm: float = 2.0

    # Numero di piani prima del mitrale e dopo l'apice
    n_before_mitral: int = 3
    n_after_apex: int = 3

    # Margine applicato alla dimensione del quadrato
    square_margin_factor: float = 1.5

    # Numero desiderato di punti
    n_points_per_short_axis_plane: int = 500
    n_points_per_long_axis_volume: int = 1000

    # Passo della griglia
    grid_spacing_mm: float = 1.0

    # Distanza massima dalla patch epicardica
    contour_expansion_mm: float = 25.0

    # Bande progressive usate per trovare nodi attorno ai contour
    profile_bands_mm: tuple = (2.0, 4.0, 6.0, 8.0, 12.0)

    # Quota totale desiderata attorno alle superfici.
    # Il resto viene selezionato dall'intera regione valida.
    surface_sampling_fraction: float = 0.80

    # Peso relativo delle tre superfici
    epi_fraction: float = 1.0 / 3.0
    lv_fraction: float = 1.0 / 3.0
    rv_fraction: float = 1.0 / 3.0

    # Numero indicativo di celle per asse nella selezione stratificata
    stratification_bins_2d: int = 20
    stratification_bins_3d: int = 10

    # Traslazione dei due volumi long-axis lungo e1
    plane_23_shift_mm: float = 25.0

    # Offset casuale della griglia.
    # La distanza tra nodi resta invariata.
    random_grid_offset: bool = True

    random_seed: int = 42

    save_npy: bool = False
    save_csv: bool = False
    plot_debug: bool = False


# ============================================================
# PATH DEL SINGOLO PAZIENTE
# ============================================================

PATIENT = "AF001"

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

LANDMARKS_CSV = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/mitral_apex_tricuspid_locations.csv"
)

OUTPUT_DIR = Path(
    "/home/rizzardi/Schreibtisch/"
    # "MRI_model/generated_npy_three_axis_grid"
)

PARAMS = GridMRIParams()


# ============================================================
# FUNZIONI GEOMETRICHE DI BASE
# ============================================================

def normalize(vector, name="vector"):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm <= 0.0:
        raise ValueError(f"{name} has zero norm.")

    return vector / norm


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

    projection_a = np.dot(points, axis_a)
    projection_b = np.dot(points, axis_b)

    range_a = float(np.ptp(projection_a))
    range_b = float(np.ptp(projection_b))

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
    axis = normalize(axis, "short-axis translation axis")

    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")

    apex_vector = apex_point - start_point
    axial_distance = float(np.dot(apex_vector, axis))

    lateral_error = float(
        np.linalg.norm(
            apex_vector - axial_distance * axis
        )
    )

    if axial_distance <= 0.0:
        raise ValueError(
            "The apex must lie in the positive e1 direction."
        )

    tolerance = max(1e-10, 1e-8 * axial_distance)

    if lateral_error > tolerance:
        raise ValueError(
            "The supplied apex is not on the anatomical axis. "
            f"Lateral error: {lateral_error:.6e}."
        )

    n_full_steps = int(
        np.floor(axial_distance / spacing)
    )

    plane_points = []

    for i in range(n_before_start, 0, -1):
        plane_points.append(
            start_point - i * spacing * axis
        )

    for i in range(n_full_steps + 1):
        plane_points.append(
            start_point + i * spacing * axis
        )

    last_regular_distance = n_full_steps * spacing

    for i in range(1, n_after_apex + 1):
        plane_points.append(
            start_point
            + (last_regular_distance + i * spacing) * axis
        )

    return np.asarray(plane_points, dtype=float)


# ============================================================
# LETTURA LANDMARK
# ============================================================

def find_patient_column(dataframe):
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
        if name in dataframe.columns:
            return name

    raise ValueError(
        "Could not find patient column. "
        f"Columns: {list(dataframe.columns)}"
    )


def read_point(row, columns, point_name):
    if not all(column in row.index for column in columns):
        raise ValueError(
            f"Missing columns for {point_name}: {columns}"
        )

    point = np.array([
        row[columns[0]],
        row[columns[1]],
        row[columns[2]],
    ], dtype=float)

    if not np.all(np.isfinite(point)):
        raise ValueError(
            f"{point_name} contains non-finite coordinates: {point}"
        )

    return point


def read_patient_three_points(dataframe, patient):
    patient_column = find_patient_column(dataframe)

    patient_rows = dataframe[
        dataframe[patient_column].astype(str) == str(patient)
    ]

    if patient_rows.empty:
        raise ValueError(
            f"Patient '{patient}' not found in CSV."
        )

    if len(patient_rows) > 1:
        raise ValueError(
            f"Patient '{patient}' appears "
            f"{len(patient_rows)} times in CSV."
        )

    row = patient_rows.iloc[0]

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


def build_three_axes(c_area, a_maxd, t_area):
    e1 = normalize(
        a_maxd - c_area,
        "e1 apex-base",
    )

    raw_e2 = normalize(
        t_area - c_area,
        "raw e2 mitral-tricuspid",
    )

    e2 = normalize(
        raw_e2 - np.dot(raw_e2, e1) * e1,
        "e2 orthogonalized",
    )

    e3 = normalize(
        np.cross(e1, e2),
        "e3",
    )

    e2 = normalize(
        np.cross(e3, e1),
        "e2 right-handed",
    )

    basis = np.column_stack([e1, e2, e3])

    if not np.allclose(
        basis.T @ basis,
        np.eye(3),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise RuntimeError(
            "The anatomical basis is not orthonormal."
        )

    if np.linalg.det(basis) <= 0.0:
        raise RuntimeError(
            "The anatomical basis is not right-handed."
        )

    return e1, e2, e3


# ============================================================
# PREPARAZIONE E CLIPPING DELLE SUPERFICI
# ============================================================

def prepare_surface(mesh, surface_name):
    surface = (
        mesh
        .extract_surface(algorithm=None)
        .triangulate()
        .clean()
    )

    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError(
            f"{surface_name} is empty after preprocessing."
        )

    if not surface.is_all_triangles:
        raise ValueError(
            f"{surface_name} is not fully triangulated."
        )

    try:
        n_open_edges = int(surface.n_open_edges)
    except Exception:
        n_open_edges = -1

    if n_open_edges > 0:
        print(
            f"WARNING: {surface_name} has "
            f"{n_open_edges} open edges."
        )

    return surface


def clip_keep_positive(polydata, origin, normal):
    plane = vtkPlane()

    plane.SetOrigin(
        *np.asarray(origin, dtype=float)
    )

    plane.SetNormal(
        *normalize(normal, "clip plane normal")
    )

    clipper = vtkClipPolyData()
    clipper.SetInputData(polydata)
    clipper.SetClipFunction(plane)
    clipper.SetValue(0.0)
    clipper.SetInsideOut(False)
    clipper.GenerateClippedOutputOff()
    clipper.Update()

    return pv.wrap(
        clipper.GetOutput()
    ).copy()


def clip_surface_to_slab(
    surface,
    center,
    normal,
    slab_half_width,
):
    if slab_half_width <= 0.0:
        raise ValueError(
            "slab_half_width must be positive."
        )

    center = np.asarray(center, dtype=float)
    normal = normalize(normal, "slab normal")

    tolerance = max(
        1e-12,
        1e-9 * slab_half_width,
    )

    effective_half_width = (
        slab_half_width + tolerance
    )

    lower_origin = (
        center
        - effective_half_width * normal
    )

    upper_origin = (
        center
        + effective_half_width * normal
    )

    clipped = clip_keep_positive(
        surface,
        lower_origin,
        normal,
    )

    if clipped.n_cells == 0:
        return None

    clipped = clip_keep_positive(
        clipped,
        upper_origin,
        -normal,
    )

    if clipped.n_cells == 0:
        return None

    clipped = (
        clipped
        .extract_surface(algorithm=None)
        .triangulate()
        .clean()
    )

    if clipped.n_cells == 0 or clipped.n_points == 0:
        return None

    return clipped


def build_slab_patches(
    surface,
    plane_specs,
    slab_half_width=None,
):
    patches = []

    for spec in plane_specs:
        current_half_width = spec.get(
            "slab_half_width",
            slab_half_width,
        )

        if current_half_width is None:
            raise ValueError(
                "Missing slab_half_width."
            )

        patch = clip_surface_to_slab(
            surface=surface,
            center=spec["center"],
            normal=spec["normal"],
            slab_half_width=current_half_width,
        )

        patches.append(patch)

    return patches


# ============================================================
# DISTANZE E SEGNO DELLA SDF
# ============================================================

def point_to_patch_distances(query_points, patch):
    query_points = np.asarray(
        query_points,
        dtype=float,
    )

    if (
        query_points.ndim != 2
        or query_points.shape[1] != 3
    ):
        raise ValueError(
            "query_points must have shape (N, 3), "
            f"got {query_points.shape}."
        )

    if query_points.shape[0] == 0:
        return np.empty(0, dtype=float)

    if patch is None or patch.n_cells == 0:
        return np.full(
            query_points.shape[0],
            np.nan,
            dtype=float,
        )

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


def compute_sign_libigl(mesh, query_points):
    query_points = np.asarray(
        query_points,
        dtype=float,
    )

    if (
        query_points.ndim != 2
        or query_points.shape[1] != 3
    ):
        raise ValueError(
            "query_points must have shape (N, 3), "
            f"got {query_points.shape}."
        )

    vertices = np.asarray(
        mesh.points,
        dtype=np.float64,
    )

    faces_raw = np.asarray(mesh.faces)

    if (
        faces_raw.size == 0
        or faces_raw.size % 4 != 0
    ):
        raise ValueError(
            "Expected a non-empty triangular PolyData mesh."
        )

    faces = faces_raw.reshape(-1, 4)

    if not np.all(faces[:, 0] == 3):
        raise ValueError(
            "compute_sign_libigl requires triangular faces."
        )

    faces = faces[:, 1:4].astype(np.int32)

    winding = igl.fast_winding_number(
        V=vertices,
        F=faces,
        Q=query_points.astype(np.float64),
    )

    sign = np.sign(
        0.5 - np.abs(winding)
    )

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

    outside_sign = np.sign(
        0.5 - np.abs(winding_outside)
    )

    if outside_sign == 0.0:
        raise RuntimeError(
            "Could not determine the outside sign."
        )

    if outside_sign < 0.0:
        sign *= -1.0

    return sign


# ============================================================
# COSTRUZIONE DELLA GRIGLIA
# ============================================================

def make_grid_axis_values(
    half_extent,
    spacing,
    rng,
    random_offset,
):
    """
    Costruisce le coordinate lungo un asse locale.

    Con random_offset=True la griglia viene traslata di una
    quantità casuale inferiore al passo. La distanza reciproca
    fra nodi rimane uguale a spacing.
    """

    if spacing <= 0.0:
        raise ValueError(
            "Grid spacing must be positive."
        )

    if random_offset:
        offset = rng.uniform(
            0.0,
            spacing,
        )
    else:
        offset = 0.0

    start = -half_extent + offset

    values = np.arange(
        start,
        half_extent + 1e-12,
        spacing,
    )

    return values


def build_oriented_plane_grid(
    center,
    u,
    v,
    side_length,
    spacing,
    seed,
    random_offset=True,
):
    """
    Costruisce una griglia 2D sul piano orientato.
    """

    rng = np.random.default_rng(seed)

    center = np.asarray(center, dtype=float)
    u = normalize(u, "plane-grid axis u")
    v = normalize(v, "plane-grid axis v")

    half = side_length / 2.0

    a_values = make_grid_axis_values(
        half_extent=half,
        spacing=spacing,
        rng=rng,
        random_offset=random_offset,
    )

    b_values = make_grid_axis_values(
        half_extent=half,
        spacing=spacing,
        rng=rng,
        random_offset=random_offset,
    )

    aa, bb = np.meshgrid(
        a_values,
        b_values,
        indexing="ij",
    )

    local_coordinates = np.column_stack([
        aa.ravel(),
        bb.ravel(),
    ])

    points = (
        center[None, :]
        + local_coordinates[:, 0:1] * u[None, :]
        + local_coordinates[:, 1:2] * v[None, :]
    )

    return points, local_coordinates


def build_oriented_volume_grid(
    center,
    normal,
    u,
    v,
    side_length,
    volume_half_width,
    spacing,
    seed,
    random_offset=True,
):
    """
    Costruisce una griglia 3D nel volume orientato.
    """

    rng = np.random.default_rng(seed)

    center = np.asarray(center, dtype=float)
    normal = normalize(normal, "volume-grid normal")
    u = normalize(u, "volume-grid axis u")
    v = normalize(v, "volume-grid axis v")

    half = side_length / 2.0

    a_values = make_grid_axis_values(
        half_extent=half,
        spacing=spacing,
        rng=rng,
        random_offset=random_offset,
    )

    b_values = make_grid_axis_values(
        half_extent=half,
        spacing=spacing,
        rng=rng,
        random_offset=random_offset,
    )

    t_values = make_grid_axis_values(
        half_extent=volume_half_width,
        spacing=spacing,
        rng=rng,
        random_offset=random_offset,
    )

    aa, bb, tt = np.meshgrid(
        a_values,
        b_values,
        t_values,
        indexing="ij",
    )

    local_coordinates = np.column_stack([
        aa.ravel(),
        bb.ravel(),
        tt.ravel(),
    ])

    points = (
        center[None, :]
        + local_coordinates[:, 0:1] * u[None, :]
        + local_coordinates[:, 1:2] * v[None, :]
        + local_coordinates[:, 2:3] * normal[None, :]
    )

    return points, local_coordinates


# ============================================================
# SELEZIONE STRATIFICATA DEI NODI
# ============================================================

def stratified_select_indices(
    candidate_indices,
    local_coordinates,
    n_select,
    n_bins,
    seed,
):
    """
    Seleziona indici distribuiti nello spazio locale.

    Funziona sia per coordinate 2D sia per coordinate 3D.
    """

    candidate_indices = np.asarray(
        candidate_indices,
        dtype=int,
    )

    if n_select <= 0 or len(candidate_indices) == 0:
        return np.empty(0, dtype=int)

    if len(candidate_indices) <= n_select:
        return candidate_indices.copy()

    rng = np.random.default_rng(seed)

    candidate_coordinates = local_coordinates[
        candidate_indices
    ]

    n_dimensions = candidate_coordinates.shape[1]

    minimum = candidate_coordinates.min(axis=0)
    maximum = candidate_coordinates.max(axis=0)

    ranges = maximum - minimum

    ranges[ranges == 0.0] = 1.0

    normalized = (
        candidate_coordinates - minimum[None, :]
    ) / ranges[None, :]

    bin_coordinates = np.floor(
        normalized * n_bins
    ).astype(int)

    bin_coordinates = np.clip(
        bin_coordinates,
        0,
        n_bins - 1,
    )

    if n_dimensions == 2:
        bin_ids = (
            bin_coordinates[:, 0] * n_bins
            + bin_coordinates[:, 1]
        )

    elif n_dimensions == 3:
        bin_ids = (
            bin_coordinates[:, 0] * n_bins * n_bins
            + bin_coordinates[:, 1] * n_bins
            + bin_coordinates[:, 2]
        )

    else:
        raise ValueError(
            "Only 2D and 3D coordinates are supported."
        )

    unique_bins = np.unique(bin_ids)

    rng.shuffle(unique_bins)

    indices_by_bin = {}

    for current_bin in unique_bins:
        positions = np.flatnonzero(
            bin_ids == current_bin
        )

        rng.shuffle(positions)

        indices_by_bin[current_bin] = list(
            candidate_indices[positions]
        )

    selected = []

    # Primo passaggio: un nodo per cella
    for current_bin in unique_bins:
        if indices_by_bin[current_bin]:
            selected.append(
                indices_by_bin[current_bin].pop()
            )

        if len(selected) >= n_select:
            return np.asarray(
                selected,
                dtype=int,
            )

    # Passaggi successivi: round-robin tra le celle
    active_bins = list(unique_bins)

    while len(selected) < n_select and active_bins:
        rng.shuffle(active_bins)

        next_active_bins = []

        for current_bin in active_bins:
            values = indices_by_bin[current_bin]

            if values:
                selected.append(values.pop())

            if values:
                next_active_bins.append(current_bin)

            if len(selected) >= n_select:
                break

        active_bins = next_active_bins

    return np.asarray(
        selected,
        dtype=int,
    )


# ============================================================
# CANDIDATI ATTORNO A UN CONTOUR
# ============================================================

def get_profile_candidate_indices(
    valid_mask,
    distances,
    target_points,
    profile_bands,
    scale_to_original_mm,
    already_used,
):
    """
    Trova nodi validi vicini a una superficie.

    Le bande sono espresse in millimetri e vengono convertite
    nel sistema normalizzato.
    """

    if distances is None:
        return np.empty(0, dtype=int), None

    finite = np.isfinite(distances)

    available_mask = (
        valid_mask
        & finite
        & ~already_used
    )

    if not np.any(available_mask):
        return np.empty(0, dtype=int), None

    final_indices = np.empty(0, dtype=int)
    final_band_mm = None

    for band_mm in profile_bands:
        band_normalized = (
            band_mm / scale_to_original_mm
        )

        current_mask = (
            available_mask
            & (distances <= band_normalized)
        )

        current_indices = np.flatnonzero(
            current_mask
        )

        final_indices = current_indices
        final_band_mm = band_mm

        if len(current_indices) >= target_points:
            break

    return final_indices, final_band_mm


# ============================================================
# SAMPLING DA GRIGLIA
# ============================================================

def sample_grid_with_surface_priorities(
    grid_points,
    local_coordinates,
    epi_mesh,
    epi_patch,
    lv_patch,
    rv_patch,
    n_requested,
    surface_expansion,
    profile_bands_mm,
    scale_to_original_mm,
    surface_sampling_fraction,
    surface_fractions,
    n_stratification_bins,
    seed,
):
    """
    Filtra la griglia usando esclusivamente:

        inside_epi OR near_epi_patch

    Successivamente privilegia nodi attorno ai contour EPI,
    LV e RV senza ampliare la regione valida.
    """

    n_grid = len(grid_points)

    if n_grid == 0:
        raise RuntimeError(
            "The generated grid contains no nodes."
        )

    # --------------------------------------------------------
    # Regione valida
    # --------------------------------------------------------

    inside_epi = (
        compute_sign_libigl(
            epi_mesh,
            grid_points,
        )
        < 0.0
    )

    epi_patch_available = (
        epi_patch is not None
        and epi_patch.n_points > 0
        and epi_patch.n_cells > 0
    )

    # ========================================================
    # CASO 1:
    # la patch epicardica è presente nello slab
    # ========================================================

    if epi_patch_available:

        distance_epi = point_to_patch_distances(
            grid_points,
            epi_patch,
        )

        near_epi = (
            np.isfinite(distance_epi)
            & (distance_epi <= surface_expansion)
        )

        valid_mask = inside_epi | near_epi

        # La patch esiste, ma nessun nodo della griglia è valido.
        # In questo caso usiamo l'intera griglia come fallback.
        if not np.any(valid_mask):

            print(
                "  WARNING: epicardial patch is present, "
                "but no grid node is inside the epicardium "
                "or sufficiently close to the epicardial patch. "
                "Using random sampling on the complete grid."
            )

            sampling_region_mode = (
                "uniform_grid_fallback_empty_epi_region"
            )

            valid_mask = np.ones(
                len(grid_points),
                dtype=bool,
            )

    # ========================================================
    # CASO 2:
    # lo slab non contiene una patch epicardica
    # ========================================================

    else:

        print(
            "  WARNING: no epicardial patch is present "
            "in this slab. "
            "Using random sampling on the complete grid."
        )

        sampling_region_mode = (
            "uniform_grid_fallback_no_epi_patch"
        )

        distance_epi = np.full(
            len(grid_points),
            np.nan,
            dtype=float,
        )

        near_epi = np.zeros(
            len(grid_points),
            dtype=bool,
        )

        valid_mask = np.ones(
            len(grid_points),
            dtype=bool,
        )

    # Se la regione epicardica normale è valida,
    # registriamo la modalità standard.
    if (
        epi_patch_available
        and np.any(inside_epi | near_epi)
    ):
        sampling_region_mode = "inside_or_near_epi"

    valid_indices = np.flatnonzero(
        valid_mask
    )

    n_available = len(valid_indices)

    if n_available == 0:
        raise RuntimeError(
            "The generated grid contains no nodes."
        )

    n_final = min(
        n_requested,
        n_available,
    )

    # ========================================================
    # FALLBACK SULL'INTERA GRIGLIA
    #
    # Questo ramo viene usato quando:
    # - non esiste una patch epicardica nello slab;
    # - oppure nessun nodo soddisfa inside_epi | near_epi.
    #
    # Non usiamo tutti i nodi.
    # Selezioniamo casualmente al massimo n_requested nodi.
    # ========================================================

    if sampling_region_mode.startswith(
        "uniform_grid_fallback"
    ):

        rng = np.random.default_rng(seed)

        selected_indices = rng.choice(
            valid_indices,
            size=n_final,
            replace=False,
        )

        print(
            f"  Random grid fallback: selected "
            f"{len(selected_indices)} / "
            f"{n_available} available grid nodes."
        )

        return grid_points[selected_indices], {
            "n_grid_nodes": n_grid,
            "n_valid_nodes": n_available,
            "n_requested": n_requested,
            "n_selected": len(selected_indices),
            "n_epi_selected": 0,
            "n_lv_selected": 0,
            "n_rv_selected": 0,
            "n_background_selected": len(selected_indices),
            "sampling_region_mode": sampling_region_mode,
        }

    # --------------------------------------------------------
    # Distanze dai contour LV e RV
    # --------------------------------------------------------

    if lv_patch is None:
        distance_lv = np.full(
            n_grid,
            np.nan,
            dtype=float,
        )
    else:
        distance_lv = point_to_patch_distances(
            grid_points,
            lv_patch,
        )

    if rv_patch is None:
        distance_rv = np.full(
            n_grid,
            np.nan,
            dtype=float,
        )
    else:
        distance_rv = point_to_patch_distances(
            grid_points,
            rv_patch,
        )

    # --------------------------------------------------------
    # Se tutti i nodi disponibili sono necessari
    # --------------------------------------------------------

    n_final = min(
        n_requested,
        n_available,
    )

    if n_available <= n_requested:
        print(
            f"  WARNING: available grid nodes "
            f"{n_available} < requested {n_requested}. "
            "Using every valid node."
        )

        return grid_points[valid_indices], {
            "n_grid_nodes": n_grid,
            "n_valid_nodes": n_available,
            "n_requested": n_requested,
            "n_selected": n_available,
            "n_epi_selected": None,
            "n_lv_selected": None,
            "n_rv_selected": None,
            "n_background_selected": None,
        }

    # --------------------------------------------------------
    # Budget attorno alle superfici
    # --------------------------------------------------------

    total_surface_target = int(
        round(
            n_final * surface_sampling_fraction
        )
    )

    epi_weight, lv_weight, rv_weight = surface_fractions

    weight_sum = (
        epi_weight
        + lv_weight
        + rv_weight
    )

    if weight_sum <= 0.0:
        raise ValueError(
            "The sum of surface fractions must be positive."
        )

    epi_target = int(
        round(
            total_surface_target
            * epi_weight
            / weight_sum
        )
    )

    lv_target = int(
        round(
            total_surface_target
            * lv_weight
            / weight_sum
        )
    )

    rv_target = (
        total_surface_target
        - epi_target
        - lv_target
    )

    used = np.zeros(
        n_grid,
        dtype=bool,
    )

    selected_groups = {}

    surface_definitions = [
        (
            "epi",
            distance_epi,
            epi_target,
        ),
        (
            "lv",
            distance_lv,
            lv_target,
        ),
        (
            "rv",
            distance_rv,
            rv_target,
        ),
    ]

    # --------------------------------------------------------
    # Selezione attorno ai singoli contour
    # --------------------------------------------------------

    for group_number, (
        group_name,
        distances,
        target,
    ) in enumerate(surface_definitions):

        candidates, band_used_mm = (
            get_profile_candidate_indices(
                valid_mask=valid_mask,
                distances=distances,
                target_points=target,
                profile_bands=profile_bands_mm,
                scale_to_original_mm=scale_to_original_mm,
                already_used=used,
            )
        )

        n_to_select = min(
            target,
            len(candidates),
        )

        selected = stratified_select_indices(
            candidate_indices=candidates,
            local_coordinates=local_coordinates,
            n_select=n_to_select,
            n_bins=n_stratification_bins,
            seed=seed + 100 + group_number,
        )

        used[selected] = True

        selected_groups[group_name] = selected

        print(
            f"  {group_name.upper():3s}: "
            f"target={target:4d}, "
            f"available={len(candidates):4d}, "
            f"selected={len(selected):4d}, "
            f"band={band_used_mm}"
        )

    # --------------------------------------------------------
    # Recupero del deficit dal resto della griglia valida
    # --------------------------------------------------------

    selected_surface = np.concatenate([
        selected_groups["epi"],
        selected_groups["lv"],
        selected_groups["rv"],
    ])

    n_remaining = (
        n_final - len(selected_surface)
    )

    remaining_candidates = np.flatnonzero(
        valid_mask & ~used
    )

    selected_background = stratified_select_indices(
        candidate_indices=remaining_candidates,
        local_coordinates=local_coordinates,
        n_select=n_remaining,
        n_bins=n_stratification_bins,
        seed=seed + 500,
    )

    used[selected_background] = True

    selected_indices = np.concatenate([
        selected_surface,
        selected_background,
    ])

    if len(selected_indices) != n_final:
        raise RuntimeError(
            "Internal sampling error: selected "
            f"{len(selected_indices)} points instead of {n_final}."
        )

    return grid_points[selected_indices], {
        "n_grid_nodes": n_grid,
        "n_valid_nodes": n_available,
        "n_requested": n_requested,
        "n_selected": len(selected_indices),
        "n_epi_selected": len(
            selected_groups["epi"]
        ),
        "n_lv_selected": len(
            selected_groups["lv"]
        ),
        "n_rv_selected": len(
            selected_groups["rv"]
        ),
        "n_background_selected": len(
            selected_background
        ),
    }


# ============================================================
# SDF MULTI-PIANO
# ============================================================

def signed_multi_plane_sdf(
    query_points,
    surface,
    plane_ids,
    slab_patches,
):
    query_points = np.asarray(
        query_points,
        dtype=float,
    )

    plane_ids = np.asarray(
        plane_ids,
        dtype=int,
    )

    if query_points.shape[0] != plane_ids.shape[0]:
        raise ValueError(
            "query_points and plane_ids must have "
            "the same length."
        )

    unsigned_distance = np.full(
        query_points.shape[0],
        np.nan,
        dtype=float,
    )

    for plane_id in np.unique(plane_ids):
        if (
            plane_id < 0
            or plane_id >= len(slab_patches)
        ):
            raise IndexError(
                f"Invalid plane_id: {plane_id}"
            )

        indices = np.where(
            plane_ids == plane_id
        )[0]

        patch = slab_patches[plane_id]

        if patch is None or patch.n_cells == 0:
            continue

        unsigned_distance[indices] = (
            point_to_patch_distances(
                query_points[indices],
                patch,
            )
        )

    global_sign = compute_sign_libigl(
        surface,
        query_points,
    )

    return unsigned_distance * global_sign


# ============================================================
# GENERAZIONE DEL SINGOLO PAZIENTE
# ============================================================

def generate_single_patient_grid_dataset(
    patient,
    all_processed_dir,
    csv_path,
    output_dir,
    params,
):
    all_processed_dir = Path(all_processed_dir)
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_dir = (
        all_processed_dir / patient
    )

    lv_path = (
        patient_dir
        / "lv_endo-processed.vtp"
    )

    rv_path = (
        patient_dir
        / "rv_endo-processed.vtp"
    )

    epi_path = (
        patient_dir
        / "epicardium-processed.vtp"
    )

    for surface_path in [
        lv_path,
        rv_path,
        epi_path,
    ]:
        if not surface_path.exists():
            raise FileNotFoundError(
                surface_path
            )

    # --------------------------------------------------------
    # Superfici
    # --------------------------------------------------------

    lv_raw = pv.read(lv_path)
    rv_raw = pv.read(rv_path)
    epi_raw = pv.read(epi_path)

    if "scale-tooriginalrange" not in epi_raw.field_data:
        raise KeyError(
            "'scale-tooriginalrange' missing from "
            f"{epi_path}"
        )

    scale_values = np.asarray(
        epi_raw.field_data[
            "scale-tooriginalrange"
        ]
    ).ravel()

    if scale_values.size == 0:
        raise ValueError(
            "'scale-tooriginalrange' is empty."
        )

    scale_to_original_um = float(
        scale_values[0]
    )

    if (
        not np.isfinite(scale_to_original_um)
        or scale_to_original_um <= 0.0
    ):
        raise ValueError(
            "Invalid scale-tooriginalrange: "
            f"{scale_to_original_um}"
        )

    scale_to_original_mm = (
        scale_to_original_um / 1000.0
    )

    lv = prepare_surface(
        lv_raw,
        f"{patient} LV endocardium",
    )

    rv = prepare_surface(
        rv_raw,
        f"{patient} RV endocardium",
    )

    epi = prepare_surface(
        epi_raw,
        f"{patient} epicardium",
    )

    # --------------------------------------------------------
    # Landmark e assi
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        csv_path,
        sep=";",
    )

    c_area, a_maxd, t_area = (
        read_patient_three_points(
            dataframe,
            patient,
        )
    )

    e1, e2, e3 = build_three_axes(
        c_area,
        a_maxd,
        t_area,
    )

    # --------------------------------------------------------
    # Conversione millimetri -> coordinate normalizzate
    # --------------------------------------------------------

    short_half_width = (
        params.short_axis_slab_width_mm / 2.0
    ) / scale_to_original_mm

    long_half_width = (
        params.long_axis_volume_width_mm / 2.0
    ) / scale_to_original_mm

    square_spacing = (
        params.square_spacing_mm
        / scale_to_original_mm
    )

    grid_spacing = (
        params.grid_spacing_mm
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

    c_long = (
        c_area
        + plane_23_shift * e1
    )

    # --------------------------------------------------------
    # Stima della dimensione comune del quadrato
    # --------------------------------------------------------

    reference_specs = [
        {
            "center": c_area,
            "normal": e1,
            "u": e2,
            "v": e3,
            "slab_half_width": short_half_width,
        },
        {
            "center": c_long,
            "normal": e2,
            "u": e1,
            "v": e3,
            "slab_half_width": long_half_width,
        },
        {
            "center": c_long,
            "normal": e3,
            "u": e1,
            "v": e2,
            "slab_half_width": long_half_width,
        },
    ]

    reference_patches = build_slab_patches(
        epi,
        reference_specs,
    )

    reference_points = [
        (
            np.empty((0, 3), dtype=float)
            if patch is None
            else np.asarray(patch.points)
        )
        for patch in reference_patches
    ]

    _, r0a, r0b = max_in_plane_extent(
        reference_points[0],
        e2,
        e3,
    )

    _, r1a, r1b = max_in_plane_extent(
        reference_points[1],
        e1,
        e3,
    )

    _, r2a, r2b = max_in_plane_extent(
        reference_points[2],
        e1,
        e2,
    )

    all_ranges = np.array([
        r0a,
        r0b,
        r1a,
        r1b,
        r2a,
        r2b,
    ])

    valid_ranges = all_ranges[
        np.isfinite(all_ranges)
        & (all_ranges > 0.0)
    ]

    if valid_ranges.size == 0:
        raise ValueError(
            "Cannot estimate square side."
        )

    square_side = (
        params.square_margin_factor
        * float(valid_ranges.max())
    )

    print("\n" + "=" * 70)
    print(f"PATIENT: {patient}")
    print("=" * 70)
    print(
        f"Scale to original range: "
        f"{scale_to_original_mm:.6f} mm/unit"
    )
    print(
        f"Square side: "
        f"{square_side * scale_to_original_mm:.2f} mm"
    )
    print(
        f"Grid spacing: "
        f"{params.grid_spacing_mm:.2f} mm"
    )

    # --------------------------------------------------------
    # Piani short-axis
    # --------------------------------------------------------

    short_axis_centers = (
        make_parallel_plane_points(
            start_point=c_area,
            apex_point=a_maxd,
            axis=e1,
            spacing=square_spacing,
            n_before_start=params.n_before_mitral,
            n_after_apex=params.n_after_apex,
        )
    )

    plane_specs = []

    # --------------------------------------------------------
    # Short-axis: sempre sampling planare
    # --------------------------------------------------------

    for center in short_axis_centers:
        plane_specs.append({
            "type": "short_axis",
            "center": center,
            "normal": e1,
            "u": e2,
            "v": e3,
            "slab_half_width": short_half_width,
            "sampling_mode": "plane",
        })


    # --------------------------------------------------------
    # Long-axis:
    # se lo spessore è minore del grid spacing,
    # campioniamo su un singolo piano centrale.
    # Altrimenti manteniamo il volume 3D.
    # --------------------------------------------------------

    if params.long_axis_volume_width_mm < params.grid_spacing_mm:
        long_axis_sampling_mode = "plane"
    else:
        long_axis_sampling_mode = "volume"


    plane_specs.append({
        "type": "normal_e2",
        "center": c_long,
        "normal": e2,
        "u": e1,
        "v": e3,
        "slab_half_width": long_half_width,
        "sampling_mode": long_axis_sampling_mode,
    })

    plane_specs.append({
        "type": "normal_e3",
        "center": c_long,
        "normal": e3,
        "u": e1,
        "v": e2,
        "slab_half_width": long_half_width,
        "sampling_mode": long_axis_sampling_mode,
    })

    # --------------------------------------------------------
    # Patch delle tre superfici
    # --------------------------------------------------------

    epi_patches = build_slab_patches(
        epi,
        plane_specs,
    )

    lv_patches = build_slab_patches(
        lv,
        plane_specs,
    )

    rv_patches = build_slab_patches(
        rv,
        plane_specs,
    )

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    all_points = []
    all_plane_ids = []
    all_type_ids = []
    sampling_stats = []

    type_to_id = {
        "short_axis": 0,
        "normal_e2": 1,
        "normal_e3": 2,
    }

    for plane_id, spec in enumerate(plane_specs):

        print("\n" + "-" * 70)
        print(
            f"Plane {plane_id:02d}: "
            f"{spec['type']}"
        )

        current_seed = (
            params.random_seed
            + plane_id * 1000
        )

        if spec["sampling_mode"] == "plane":

            grid_points, local_coordinates = (
                build_oriented_plane_grid(
                    center=spec["center"],
                    u=spec["u"],
                    v=spec["v"],
                    side_length=square_side,
                    spacing=grid_spacing,
                    seed=current_seed,
                    random_offset=params.random_grid_offset,
                )
            )

            # Short-axis e long-axis usano budget diversi
            if spec["type"] == "short_axis":
                n_requested = (
                    params.n_points_per_short_axis_plane
                )
            else:
                n_requested = (
                    params.n_points_per_long_axis_volume
                )

            stratification_bins = (
                params.stratification_bins_2d
            )

        else:

            grid_points, local_coordinates = (
                build_oriented_volume_grid(
                    center=spec["center"],
                    normal=spec["normal"],
                    u=spec["u"],
                    v=spec["v"],
                    side_length=square_side,
                    volume_half_width=spec[
                        "slab_half_width"
                    ],
                    spacing=grid_spacing,
                    seed=current_seed,
                    random_offset=params.random_grid_offset,
                )
            )

            n_requested = (
                params.n_points_per_long_axis_volume
            )

            stratification_bins = (
                params.stratification_bins_3d
            )

        points_i, stats_i = (
            sample_grid_with_surface_priorities(
                grid_points=grid_points,
                local_coordinates=local_coordinates,
                epi_mesh=epi,
                epi_patch=epi_patches[plane_id],
                lv_patch=lv_patches[plane_id],
                rv_patch=rv_patches[plane_id],
                n_requested=n_requested,
                surface_expansion=surface_expansion,
                profile_bands_mm=params.profile_bands_mm,
                scale_to_original_mm=scale_to_original_mm,
                surface_sampling_fraction=(
                    params.surface_sampling_fraction
                ),
                surface_fractions=(
                    params.epi_fraction,
                    params.lv_fraction,
                    params.rv_fraction,
                ),
                n_stratification_bins=(
                    stratification_bins
                ),
                seed=current_seed,
            )
        )

        stats_i["plane_id"] = plane_id
        stats_i["plane_type"] = spec["type"]

        sampling_stats.append(stats_i)

        print(
            f"  Grid nodes:     "
            f"{stats_i['n_grid_nodes']}"
        )
        print(
            f"  Valid nodes:    "
            f"{stats_i['n_valid_nodes']}"
        )
        print(
            f"  Selected nodes: "
            f"{stats_i['n_selected']}"
        )

        all_points.append(points_i)

        all_plane_ids.append(
            np.full(
                points_i.shape[0],
                plane_id,
                dtype=int,
            )
        )

        all_type_ids.append(
            np.full(
                points_i.shape[0],
                type_to_id[spec["type"]],
                dtype=int,
            )
        )

    # --------------------------------------------------------
    # Concatenazione
    # --------------------------------------------------------

    points = np.vstack(all_points)

    plane_ids = np.concatenate(
        all_plane_ids
    )

    plane_type_ids = np.concatenate(
        all_type_ids
    )

    # --------------------------------------------------------
    # Calcolo delle SDF
    # --------------------------------------------------------

    print("\nCalculating SDF values...")

    sdf_epi_raw = signed_multi_plane_sdf(
        points,
        epi,
        plane_ids,
        epi_patches,
    )

    sdf_lv_raw = signed_multi_plane_sdf(
        points,
        lv,
        plane_ids,
        lv_patches,
    )

    sdf_rv_raw = signed_multi_plane_sdf(
        points,
        rv,
        plane_ids,
        rv_patches,
    )

    mask_epi = np.isfinite(
        sdf_epi_raw
    ).astype(float)

    mask_lv = np.isfinite(
        sdf_lv_raw
    ).astype(float)

    mask_rv = np.isfinite(
        sdf_rv_raw
    ).astype(float)

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
    ]).astype(np.float32)

    # --------------------------------------------------------
    # Salvataggio
    # --------------------------------------------------------

    output_npy = (
        output_dir
        / f"{patient}_three_axis_mri_grid_samples.npy"
    )

    if params.save_npy:
        np.save(
            output_npy,
            samples,
        )

        print(
            f"\nSaved NPY:\n{output_npy}"
        )

    if params.save_csv:
        output_csv = (
            output_dir
            / f"{patient}_three_axis_mri_grid_samples.csv"
        )

        pd.DataFrame(
            samples,
            columns=[
                "x",
                "y",
                "z",
                "sdf_epi",
                "sdf_lv",
                "sdf_rv",
                "mask_epi",
                "mask_lv",
                "mask_rv",
            ],
        ).to_csv(
            output_csv,
            index=False,
        )

        print(
            f"Saved CSV:\n{output_csv}"
        )

    stats_dataframe = pd.DataFrame(
        sampling_stats
    )

    stats_path = (
        output_dir
        / f"{patient}_grid_sampling_stats.csv"
    )

    stats_dataframe.to_csv(
        stats_path,
        index=False,
    )

    print(
        f"Saved sampling statistics:\n{stats_path}"
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Patient: {patient}")
    print(f"Planes/volumes: {len(plane_specs)}")
    print(f"Total samples: {len(samples)}")
    print(
        f"EPI mask fraction: "
        f"{mask_epi.mean():.4f}"
    )
    print(
        f"LV mask fraction:  "
        f"{mask_lv.mean():.4f}"
    )
    print(
        f"RV mask fraction:  "
        f"{mask_rv.mean():.4f}"
    )

    debug = {
        "patient": patient,
        "lv": lv,
        "rv": rv,
        "epi": epi,
        "plane_specs": plane_specs,
        "epi_patches": epi_patches,
        "lv_patches": lv_patches,
        "rv_patches": rv_patches,
        "points": points,
        "plane_ids": plane_ids,
        "plane_type_ids": plane_type_ids,
        "samples": samples,
        "sampling_stats": sampling_stats,
        "square_side": square_side,
        "scale_to_original_mm": scale_to_original_mm,
    }

    if params.plot_debug:
        plot_sampling_debug(debug)

    return samples, stats_dataframe, debug


# ============================================================
# VISUALIZZAZIONE
# ============================================================

def plot_sampling_debug(debug):
    plotter = pv.Plotter(
        window_size=(1600, 1000)
    )

    plotter.set_background("white")

    plotter.add_mesh(
        debug["epi"],
        color="lightgray",
        opacity=0.15,
        smooth_shading=True,
    )

    plotter.add_mesh(
        debug["lv"],
        color="lightcoral",
        opacity=0.20,
        smooth_shading=True,
    )

    plotter.add_mesh(
        debug["rv"],
        color="lightblue",
        opacity=0.20,
        smooth_shading=True,
    )

    cloud = pv.PolyData(
        debug["points"]
    )

    cloud["plane_id"] = debug[
        "plane_ids"
    ].astype(float)

    plotter.add_mesh(
        cloud,
        scalars="plane_id",
        point_size=6,
        render_points_as_spheres=True,
        cmap="turbo",
        scalar_bar_args={
            "title": "Plane ID",
        },
    )

    for plane_id, spec in enumerate(
        debug["plane_specs"]
    ):
        if spec["sampling_mode"] == "plane":

            square = make_square(
                center=spec["center"],
                u=spec["u"],
                v=spec["v"],
                side=debug["square_side"],
            )

            plotter.add_mesh(
                square,
                color="black",
                opacity=0.03,
                show_edges=True,
            )

    plotter.add_axes()

    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
    )

    plotter.view_isometric()
    plotter.reset_camera()

    plotter.show(
        title=(
            f"{debug['patient']} - "
            "grid-based MRI sampling"
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_single_patient_grid_dataset(
        patient=PATIENT,
        all_processed_dir=ALL_PROCESSED_DIR,
        csv_path=LANDMARKS_CSV,
        output_dir=OUTPUT_DIR,
        params=PARAMS,
    )