#!/usr/bin/env python3
"""
MRI-like grid sampling con SDF calcolata rispetto al contour ESATTO sul piano.

Output NPY:
[x,y,z,sdf_epi,sdf_lv,sdf_rv,mask_epi,mask_lv,mask_rv]

Ordine:
1) tutti gli short-axis
2) LAX 1 (normale e2)
3) LAX 2 (normale e3)

Se n_points_per_long_axis_volume=2000:
    samples[-4000:-2000] -> LAX 1
    samples[-2000:]      -> LAX 2

La magnitudine della SDF e' 2D, cioe' la distanza minima dal contour
mesh ∩ plane. Il segno resta inside/outside rispetto alla mesh 3D completa.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
import igl


@dataclass
class GridMRIParams:
    square_spacing_mm: float = 6.0
    short_axis_slab_width_mm: float = 0.75      # compatibilita'; non usato per |SDF|
    long_axis_volume_width_mm: float = 0.75     # compatibilita'; LAX qui e' planare
    n_before_mitral: int = 3
    n_after_apex: int = 3
    square_margin_factor: float = 1.5
    n_points_per_short_axis_plane: int = 500
    n_points_per_long_axis_volume: int = 2000
    grid_spacing_mm: float = 1.0
    contour_expansion_mm: float = 25.0
    profile_bands_mm: tuple = (2.0, 4.0, 6.0, 8.0, 12.0)
    surface_sampling_fraction: float = 0.80
    epi_fraction: float = 1.0 / 3.0
    lv_fraction: float = 1.0 / 3.0
    rv_fraction: float = 1.0 / 3.0
    stratification_bins_2d: int = 20
    stratification_bins_3d: int = 10            # compatibilita'
    plane_23_shift_mm: float = 25.0
    random_grid_offset: bool = True
    random_seed: int = 42
    save_npy: bool = False
    save_csv: bool = False
    plot_debug: bool = False


def normalize(v, name="vector"):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError(f"{name} has zero norm")
    return v / n


def find_patient_column(df):
    for c in ["patient", "Patient", "PATIENT", "patient_id", "PatientID", "id", "ID"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find patient column. Columns: {list(df.columns)}")


def _read_point(row, cols, name):
    if not all(c in row.index for c in cols):
        raise ValueError(f"Missing columns for {name}: {cols}")
    p = np.asarray([row[cols[0]], row[cols[1]], row[cols[2]]], dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values: {p}")
    return p


def read_patient_three_points(df, patient):
    pc = find_patient_column(df)
    rows = df[df[pc].astype(str) == str(patient)]
    if rows.empty:
        raise ValueError(f"Patient {patient} not found in CSV")
    if len(rows) != 1:
        raise ValueError(f"Patient {patient} appears {len(rows)} times in CSV")
    row = rows.iloc[0]
    c_area = _read_point(row, ["C_area_x", "C_area_y", "C_area_z"], "C_area")
    apex = _read_point(row, ["A_maxD_x", "A_maxD_y", "A_maxD_z"], "A_maxD")
    t_area = _read_point(row, ["T_area_x", "T_area_y", "T_area_z"], "T_area")
    return c_area, apex, t_area


def build_three_axes(c_area, apex, t_area):
    e1 = normalize(apex - c_area, "e1")
    raw_e2 = normalize(t_area - c_area, "raw e2")
    e2 = normalize(raw_e2 - np.dot(raw_e2, e1) * e1, "e2")
    e3 = normalize(np.cross(e1, e2), "e3")
    e2 = normalize(np.cross(e3, e1), "e2 right handed")
    return e1, e2, e3


def prepare_surface(mesh, name):
    s = mesh.extract_surface(algorithm=None).triangulate().clean()
    if s.n_points == 0 or s.n_cells == 0:
        raise ValueError(f"{name} empty after preprocessing")
    if not s.is_all_triangles:
        raise ValueError(f"{name} is not triangulated")
    return s


def make_parallel_plane_points(start, apex, axis, spacing, n_before, n_after):
    axis = normalize(axis)
    length = float(np.dot(apex - start, axis))
    if length <= 0 or spacing <= 0:
        raise ValueError("Invalid apex-axis geometry or spacing")
    n_full = int(np.floor(length / spacing))
    pts = []
    for i in range(n_before, 0, -1):
        pts.append(start - i * spacing * axis)
    for i in range(n_full + 1):
        pts.append(start + i * spacing * axis)
    last = n_full * spacing
    for i in range(1, n_after + 1):
        pts.append(start + (last + i * spacing) * axis)
    return np.asarray(pts, dtype=float)


def projected_extent(points, u, v):
    a = np.asarray(points) @ normalize(u)
    b = np.asarray(points) @ normalize(v)
    return max(float(np.ptp(a)), float(np.ptp(b)))


# ---------------------------------------------------------------------
# SEGNO 3D
# ---------------------------------------------------------------------

def compute_sign_libigl(mesh, query_points):
    V = np.asarray(mesh.points, dtype=np.float64)
    fraw = np.asarray(mesh.faces)
    F4 = fraw.reshape(-1, 4)
    if not np.all(F4[:, 0] == 3):
        raise ValueError("Mesh must be triangular")
    F = F4[:, 1:4].astype(np.int32)
    Q = np.asarray(query_points, dtype=np.float64)
    w = igl.fast_winding_number(V=V, F=F, Q=Q)
    return np.where(np.abs(w) > 0.5, -1.0, +1.0)


# ---------------------------------------------------------------------
# CONTOUR = MESH ∩ PIANO
# ---------------------------------------------------------------------

def slice_surface_with_plane(surface, origin, normal):
    c = surface.slice(normal=normalize(normal), origin=np.asarray(origin, float), generate_triangles=False)
    if c is None:
        return None
    c = c.clean()
    if c.n_points == 0 or c.n_cells == 0 or np.asarray(c.lines).size == 0:
        return None
    return c


def plane_coords(points, origin, u, v):
    local = np.asarray(points, float) - np.asarray(origin, float)[None, :]
    return np.column_stack([local @ normalize(u), local @ normalize(v)])


def contour_segments_2d(contour, origin, u, v):
    if contour is None:
        return np.empty((0, 2)), np.empty((0, 2))
    p2 = plane_coords(contour.points, origin, u, v)
    lines = np.asarray(contour.lines)
    starts, ends = [], []
    k = 0
    while k < len(lines):
        n = int(lines[k])
        ids = lines[k + 1:k + 1 + n]
        for j in range(max(0, n - 1)):
            starts.append(p2[ids[j]])
            ends.append(p2[ids[j + 1]])
        k += n + 1
    if not starts:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.asarray(starts), np.asarray(ends)


def point_to_segment_distance_2d(query_2d, seg_a, seg_b, chunk=3000):
    q = np.asarray(query_2d, float)
    a = np.asarray(seg_a, float)
    b = np.asarray(seg_b, float)
    if len(a) == 0:
        return np.full(len(q), np.nan)
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    ok = ab2 > 1e-20
    a, ab, ab2 = a[ok], ab[ok], ab2[ok]
    if len(a) == 0:
        return np.full(len(q), np.nan)
    out = np.empty(len(q), dtype=float)
    for s in range(0, len(q), chunk):
        e = min(s + chunk, len(q))
        qq = q[s:e]
        ap = qq[:, None, :] - a[None, :, :]
        t = np.sum(ap * ab[None, :, :], axis=2) / ab2[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d2 = np.sum((qq[:, None, :] - closest) ** 2, axis=2)
        out[s:e] = np.sqrt(np.min(d2, axis=1))
    return out


def contour_unsigned_distance(query_points, contour, origin, u, v):
    q2 = plane_coords(query_points, origin, u, v)
    a, b = contour_segments_2d(contour, origin, u, v)
    return point_to_segment_distance_2d(q2, a, b)


def signed_contour_sdf(query_points, surface, contour, origin, u, v):
    n = len(query_points)
    if contour is None:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=np.float32)
    d = contour_unsigned_distance(query_points, contour, origin, u, v)
    valid = np.isfinite(d)
    sign = compute_sign_libigl(surface, query_points)
    sdf = np.zeros(n, dtype=float)
    sdf[valid] = d[valid] * sign[valid]
    return sdf, valid.astype(np.float32)


# ---------------------------------------------------------------------
# GRID 2D E SELEZIONE
# ---------------------------------------------------------------------

def generate_planar_grid(center, u, v, side, spacing, random_offset, rng):
    half = side / 2.0
    if random_offset:
        ou = rng.uniform(-0.5 * spacing, 0.5 * spacing)
        ov = rng.uniform(-0.5 * spacing, 0.5 * spacing)
    else:
        ou = ov = 0.0
    aa = np.arange(-half + ou, half + 0.5 * spacing, spacing)
    bb = np.arange(-half + ov, half + 0.5 * spacing, spacing)
    A, B = np.meshgrid(aa, bb, indexing="xy")
    local = np.column_stack([A.ravel(), B.ravel()])
    pts = (np.asarray(center)[None, :]
           + local[:, 0:1] * normalize(u)[None, :]
           + local[:, 1:2] * normalize(v)[None, :])
    return pts, local


def stratified_order(local_2d, bins, rng):
    x = np.asarray(local_2d, float)
    if len(x) == 0:
        return np.empty(0, dtype=int)
    bins = max(1, int(bins))
    mn, mx = x.min(0), x.max(0)
    span = np.maximum(mx - mn, 1e-12)
    ij = np.floor((x - mn) / span * bins).astype(int)
    ij = np.clip(ij, 0, bins - 1)
    cid = ij[:, 0] * bins + ij[:, 1]
    cells = np.unique(cid)
    rng.shuffle(cells)
    buckets = {}
    for c in cells:
        ids = np.flatnonzero(cid == c)
        rng.shuffle(ids)
        buckets[int(c)] = list(ids)
    order = []
    active = True
    while active:
        active = False
        for c in cells:
            b = buckets[int(c)]
            if b:
                order.append(b.pop())
                active = True
    return np.asarray(order, dtype=int)


def distributed_pick(candidate_indices, local_2d, n, bins, rng):
    ids = np.asarray(candidate_indices, dtype=int)
    if n <= 0 or len(ids) == 0:
        return np.empty(0, dtype=int)
    if len(ids) <= n:
        return ids.copy()
    o = stratified_order(local_2d[ids], bins, rng)
    return ids[o[:n]]


def select_training_nodes(valid_ids, local_2d, dists, n_requested, params, scale_mm, rng):
    valid_ids = np.asarray(valid_ids, dtype=int)
    if len(valid_ids) <= n_requested:
        return valid_ids.copy()

    n_surface = int(round(params.surface_sampling_fraction * n_requested))
    weights = np.asarray([params.epi_fraction, params.lv_fraction, params.rv_fraction], float)
    if weights.sum() <= 0:
        weights[:] = 1.0
    weights /= weights.sum()
    budgets = np.floor(n_surface * weights).astype(int)
    while budgets.sum() < n_surface:
        budgets[np.argmax(weights)] += 1

    chosen = []
    used = set()
    bands = [mm / scale_mm for mm in params.profile_bands_mm]

    for surf_i in range(3):
        need = int(budgets[surf_i])
        if need <= 0:
            continue
        d = dists[surf_i]
        candidates = np.empty(0, dtype=int)
        for band in bands:
            c = valid_ids[np.isfinite(d[valid_ids]) & (d[valid_ids] <= band)]
            c = np.asarray([i for i in c if int(i) not in used], dtype=int)
            if len(c):
                candidates = c
            if len(c) >= need:
                break
        picked = distributed_pick(candidates, local_2d, need, params.stratification_bins_2d, rng)
        for i in picked:
            chosen.append(int(i)); used.add(int(i))

    remaining_n = n_requested - len(chosen)
    if remaining_n > 0:
        rest = np.asarray([i for i in valid_ids if int(i) not in used], dtype=int)
        bg = distributed_pick(rest, local_2d, remaining_n, params.stratification_bins_2d, rng)
        chosen.extend([int(i) for i in bg])

    return np.asarray(chosen, dtype=int)


# ---------------------------------------------------------------------
# UN PIANO
# ---------------------------------------------------------------------

def generate_one_plane_samples(name, center, normal, u, v, side, n_requested,
                               grid_spacing_norm, contour_expansion_norm,
                               scale_mm, epi, lv, rv, params, seed):
    rng = np.random.default_rng(seed)

    c_epi = slice_surface_with_plane(epi, center, normal)
    c_lv  = slice_surface_with_plane(lv,  center, normal)
    c_rv  = slice_surface_with_plane(rv,  center, normal)

    #if c_epi is None:
    #    raise RuntimeError(f"{name}: epicardial contour is empty")

    if c_epi is None:
        print(
            f"WARNING {name}: epicardial contour is empty; "
            "SDF will be 0 and mask 0 for that surface."
        )

    grid, local = generate_planar_grid(center, u, v, side, grid_spacing_norm,
                                       params.random_grid_offset, rng)
    if len(grid) == 0:
        raise RuntimeError(f"{name}: grid is empty")

    d_epi = contour_unsigned_distance(grid, c_epi, center, u, v)
    d_lv = contour_unsigned_distance(grid, c_lv, center, u, v) if c_lv is not None else np.full(len(grid), np.nan)
    d_rv = contour_unsigned_distance(grid, c_rv, center, u, v) if c_rv is not None else np.full(len(grid), np.nan)

    if c_epi is None:
        # Nessun contour epicardico su questo piano:
        # teniamo comunque i punti del piano come campioni,
        # ma più avanti sdf_epi = 0 e mask_epi = 0.
        valid_ids = np.arange(
            len(grid),
            dtype=int,
        )

    else:
        inside_epi = compute_sign_libigl(
            epi,
            grid,
        ) < 0

        near_epi = (
            np.isfinite(d_epi)
            & (d_epi <= contour_expansion_norm)
        )

        valid_ids = np.flatnonzero(
            inside_epi | near_epi
        )

    if len(valid_ids) == 0:
        raise RuntimeError(
            f"{name}: no valid grid nodes"
        )

    chosen = select_training_nodes(valid_ids, local, (d_epi, d_lv, d_rv),
                                   n_requested, params, scale_mm, rng)
    if len(chosen) == 0:
        raise RuntimeError(f"{name}: no selected nodes")

    pts = grid[chosen]
    sdf_epi, m_epi = signed_contour_sdf(pts, epi, c_epi, center, u, v)
    sdf_lv,  m_lv  = signed_contour_sdf(pts, lv,  c_lv,  center, u, v)
    sdf_rv,  m_rv  = signed_contour_sdf(pts, rv,  c_rv,  center, u, v)

    samples = np.column_stack([
        pts, sdf_epi, sdf_lv, sdf_rv, m_epi, m_lv, m_rv
    ]).astype(np.float32)

    stats = {
        "plane_name": name,
        "n_grid": len(grid),
        "n_valid": len(valid_ids),
        "n_selected": len(chosen),
        "has_epi_contour": c_epi is not None,
        "has_lv_contour": c_lv is not None,
        "has_rv_contour": c_rv is not None,
    }

    debug = {
        "selected_points": pts,
        "contour_epi": c_epi,
        "contour_lv": c_lv,
        "contour_rv": c_rv,
        "center": np.asarray(center),
        "normal": np.asarray(normal),
        "u": np.asarray(u),
        "v": np.asarray(v),
    }
    return samples, stats, debug


# ---------------------------------------------------------------------
# FUNZIONE COMPATIBILE CON IL TUO BATCH
# ---------------------------------------------------------------------

def generate_single_patient_grid_dataset(patient, all_processed_dir, csv_path,
                                         output_dir, params=GridMRIParams()):
    all_processed_dir = Path(all_processed_dir)
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdir = all_processed_dir / patient
    epi_path = pdir / "epicardium-processed.vtp"
    lv_path  = pdir / "lv_endo-processed.vtp"
    rv_path  = pdir / "rv_endo-processed.vtp"
    for p in [epi_path, lv_path, rv_path]:
        if not p.is_file():
            raise FileNotFoundError(p)

    epi = prepare_surface(pv.read(epi_path), "epicardium")
    lv  = prepare_surface(pv.read(lv_path),  "lv_endo")
    rv  = prepare_surface(pv.read(rv_path),  "rv_endo")

    df = pd.read_csv(csv_path, sep=";")
    c_area, apex, t_area = read_patient_three_points(df, patient)
    e1, e2, e3 = build_three_axes(c_area, apex, t_area)

    if "scale-tooriginalrange" not in epi.field_data:
        raise KeyError("Missing epicardium field_data['scale-tooriginalrange']")
    scale_um = float(np.asarray(epi.field_data["scale-tooriginalrange"]).ravel()[0])
    scale_mm = scale_um / 1000.0
    if scale_mm <= 0:
        raise ValueError(f"Invalid scale_to_original_mm: {scale_mm}")

    spacing_sa = params.square_spacing_mm / scale_mm
    spacing_grid = params.grid_spacing_mm / scale_mm
    expansion = params.contour_expansion_mm / scale_mm
    shift_lax = params.plane_23_shift_mm / scale_mm

    side_sa   = params.square_margin_factor * projected_extent(epi.points, e2, e3)
    side_lax1 = params.square_margin_factor * projected_extent(epi.points, e1, e3)
    side_lax2 = params.square_margin_factor * projected_extent(epi.points, e1, e2)

    sa_centers = make_parallel_plane_points(
        c_area, apex, e1, spacing_sa,
        params.n_before_mitral, params.n_after_apex
    )

    # Se nel tuo vecchio codice il segno dello shift era opposto,
    # cambia + in - qui e SOLO qui.
    c_long = c_area + shift_lax * e1

    specs = []
    for i, c in enumerate(sa_centers):
        specs.append({
            "name": f"SA_{i:02d}", "type": "short_axis",
            "center": c, "normal": e1, "u": e2, "v": e3,
            "side": side_sa, "n": params.n_points_per_short_axis_plane,
        })

    if params.n_points_per_long_axis_volume > 0:
        specs.append({
            "name": "LAX_1_normal_e2", "type": "long_axis_1",
            "center": c_long, "normal": e2, "u": e1, "v": e3,
            "side": side_lax1, "n": params.n_points_per_long_axis_volume,
        })
        specs.append({
            "name": "LAX_2_normal_e3", "type": "long_axis_2",
            "center": c_long, "normal": e3, "u": e1, "v": e2,
            "side": side_lax2, "n": params.n_points_per_long_axis_volume,
        })

    all_samples, stats, debug_planes = [], [], []
    for i, s in enumerate(specs):
        print(f"[{i+1}/{len(specs)}] {s['name']}")
        arr, st, dbg = generate_one_plane_samples(
            s["name"], s["center"], s["normal"], s["u"], s["v"],
            s["side"], s["n"], spacing_grid, expansion, scale_mm,
            epi, lv, rv, params, params.random_seed + i
        )
        st["plane_index"] = i
        st["plane_type"] = s["type"]
        all_samples.append(arr)
        stats.append(st)
        debug_planes.append(dbg)

    samples = np.vstack(all_samples).astype(np.float32)
    if samples.shape[1] != 9:
        raise RuntimeError(f"Unexpected output shape: {samples.shape}")
    if not np.all(np.isfinite(samples[:, :6])):
        raise RuntimeError("Non-finite xyz/SDF values found")

    if params.save_npy:
        out = output_dir / f"{patient}_three_axis_mri_grid_samples.npy"
        np.save(out, samples)
        print("Saved:", out)

    if params.save_csv:
        out_csv = output_dir / f"{patient}_three_axis_mri_grid_samples.csv"
        pd.DataFrame(samples, columns=[
            "x","y","z","sdf_epi","sdf_lv","sdf_rv",
            "mask_epi","mask_lv","mask_rv"
        ]).to_csv(out_csv, index=False)

    if params.plot_debug:
        pl = pv.Plotter()
        pl.add_mesh(epi, opacity=0.12, color="lightgray")
        for d in debug_planes:
            pl.add_points(d["selected_points"], point_size=3)
            for key, color in [("contour_epi","black"), ("contour_lv","red"), ("contour_rv","blue")]:
                c = d[key]
                if c is not None:
                    pl.add_mesh(c, color=color, line_width=3)
        pl.show()

    debug = {
        "plane_specs": specs,
        "plane_debug": debug_planes,
        "axes": {"e1": e1, "e2": e2, "e3": e3},
        "landmarks": {"C_area": c_area, "A_maxD": apex, "T_area": t_area},
        "scale_to_original_mm": scale_mm,
    }
    return samples, pd.DataFrame(stats), debug


if __name__ == "__main__":
    PATIENT = "AF001"
    ALL_PROCESSED_DIR = Path("/home/rizzardi/Schreibtisch/AF001_aligned_processed")
    LANDMARKS_CSV = Path("/home/rizzardi/Schreibtisch/MRI_model/mitral_apex_tricuspid_locations.csv")
    OUTPUT_DIR = Path("/home/rizzardi/Schreibtisch/MRI_model/generated_npy_three_axis_grid_contourSDF")

    PARAMS = GridMRIParams(
        square_spacing_mm=6.0,
        short_axis_slab_width_mm=0.75,
        long_axis_volume_width_mm=0.75,
        n_before_mitral=3,
        n_after_apex=3,
        square_margin_factor=1.5,
        n_points_per_short_axis_plane=500,
        n_points_per_long_axis_volume=2000,
        grid_spacing_mm=1.0,
        contour_expansion_mm=25.0,
        profile_bands_mm=(2.0,4.0,6.0,8.0,12.0),
        surface_sampling_fraction=0.80,
        epi_fraction=1/3,
        lv_fraction=1/3,
        rv_fraction=1/3,
        stratification_bins_2d=20,
        stratification_bins_3d=10,
        plane_23_shift_mm=25.0,
        random_grid_offset=True,
        random_seed=42,
        save_npy=True,
        save_csv=False,
        plot_debug=True,
    )

    generate_single_patient_grid_dataset(
        patient=PATIENT,
        all_processed_dir=ALL_PROCESSED_DIR,
        csv_path=LANDMARKS_CSV,
        output_dir=OUTPUT_DIR,
        params=PARAMS,
    )