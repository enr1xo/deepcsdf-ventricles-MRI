#!/usr/bin/env python3
"""
visualize_SDF_curves_on_LAX_plane.py
====================================

Per un paziente:
- seleziona un piano SA (default: il 9° piano -> indice 8)
- usa entrambi i piani LAX (LAX1 e LAX2)
- per ciascuna retta di intersezione SA x LAX campiona N punti
- calcola, sugli stessi punti:
      SDF rispetto al contour SA
      SDF rispetto al contour del LAX corrispondente
- costruisce DUE curve 3D per ogni LAX:
      curva_SA  = q + sdf_SA  * g
      curva_LAX = q + sdf_LAX * g
  dove:
      q = punto sulla retta di intersezione
      g = direzione nel piano LAX, ortogonale alla retta di intersezione

Quindi le curve SDF stanno nel piano LAX, come richiesto.

La scena 3D mostra:
- le superfici
- il contour SA della superficie scelta
- i due contour LAX della superficie scelta
- le due rette di intersezione
- per ciascun LAX, due curve:
      una per SDF rispetto al contour SA
      una per SDF rispetto al contour LAX
"""

from __future__ import annotations

import argparse
from pathlib import Path

import igl
import numpy as np
import pandas as pd
import pyvista as pv


# ============================================================
# BASE GEOMETRICA
# ============================================================

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


def read_point(row, cols, name):
    if not all(c in row.index for c in cols):
        raise ValueError(f"Missing columns for {name}: {cols}")
    p = np.asarray([row[cols[0]], row[cols[1]], row[cols[2]]], dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} has non-finite coordinates: {p}")
    return p


def read_landmarks(csv_path, patient):
    df = pd.read_csv(csv_path, sep=";")
    pc = find_patient_column(df)
    rows = df[df[pc].astype(str) == str(patient)]
    if rows.empty:
        raise ValueError(f"Patient {patient} not found in CSV")
    if len(rows) != 1:
        raise ValueError(f"Patient {patient} appears {len(rows)} times in CSV")

    row = rows.iloc[0]
    c_area = read_point(row, ["C_area_x", "C_area_y", "C_area_z"], "C_area")
    a_maxd = read_point(row, ["A_maxD_x", "A_maxD_y", "A_maxD_z"], "A_maxD")
    t_area = read_point(row, ["T_area_x", "T_area_y", "T_area_z"], "T_area")
    return c_area, a_maxd, t_area


def build_three_axes(c_area, a_maxd, t_area):
    e1 = normalize(a_maxd - c_area, "e1")
    raw_e2 = normalize(t_area - c_area, "raw_e2")
    e2 = normalize(raw_e2 - np.dot(raw_e2, e1) * e1, "e2")
    e3 = normalize(np.cross(e1, e2), "e3")
    e2 = normalize(np.cross(e3, e1), "e2 right-handed")
    return e1, e2, e3


def prepare_surface(mesh, name):
    surf = mesh.extract_surface(algorithm=None).triangulate().clean()
    if surf.n_points == 0 or surf.n_cells == 0:
        raise ValueError(f"{name} is empty")
    if not surf.is_all_triangles:
        raise ValueError(f"{name} is not triangular")
    return surf


def make_parallel_plane_points(start_point, apex_point, axis, spacing, n_before_start=3, n_after_apex=3):
    start_point = np.asarray(start_point, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    axis = normalize(axis, "axis")

    axial_distance = float(np.dot(apex_point - start_point, axis))
    if axial_distance <= 0 or spacing <= 0:
        raise ValueError("Invalid geometry or spacing")

    n_full_steps = int(np.floor(axial_distance / spacing))
    points = []

    for i in range(n_before_start, 0, -1):
        points.append(start_point - i * spacing * axis)

    for i in range(n_full_steps + 1):
        points.append(start_point + i * spacing * axis)

    last_regular_distance = n_full_steps * spacing

    for i in range(1, n_after_apex + 1):
        points.append(start_point + (last_regular_distance + i * spacing) * axis)

    return np.asarray(points, dtype=float)


# ============================================================
# CONTOUR E SDF
# ============================================================

def compute_sign_libigl(mesh, query_points):
    V = np.asarray(mesh.points, dtype=np.float64)
    F4 = np.asarray(mesh.faces).reshape(-1, 4)
    if not np.all(F4[:, 0] == 3):
        raise ValueError("Mesh must be triangular")
    F = F4[:, 1:4].astype(np.int32)
    Q = np.asarray(query_points, dtype=np.float64)
    w = igl.fast_winding_number(V=V, F=F, Q=Q)
    return np.where(np.abs(w) > 0.5, -1.0, +1.0)


def slice_surface_with_plane(surface, origin, normal):
    contour = surface.slice(
        normal=normalize(normal),
        origin=np.asarray(origin, dtype=float),
        generate_triangles=False,
    )
    if contour is None:
        return None
    contour = contour.clean()
    if contour.n_points == 0 or contour.n_cells == 0 or np.asarray(contour.lines).size == 0:
        return None
    return contour


def plane_coords(points, origin, u, v):
    local = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)[None, :]
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
        ids = lines[k + 1 : k + 1 + n]
        for j in range(max(0, n - 1)):
            starts.append(p2[ids[j]])
            ends.append(p2[ids[j + 1]])
        k += n + 1

    if not starts:
        return np.empty((0, 2)), np.empty((0, 2))

    return np.asarray(starts), np.asarray(ends)


def point_to_segment_distance_2d(query_2d, seg_a, seg_b, chunk=3000):
    q = np.asarray(query_2d, dtype=float)
    a = np.asarray(seg_a, dtype=float)
    b = np.asarray(seg_b, dtype=float)

    if len(a) == 0:
        return np.full(len(q), np.nan)

    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)

    ok = ab2 > 1e-20
    a = a[ok]
    ab = ab[ok]
    ab2 = ab2[ok]

    if len(a) == 0:
        return np.full(len(q), np.nan)

    out = np.empty(len(q), dtype=float)

    for start in range(0, len(q), chunk):
        stop = min(start + chunk, len(q))
        qq = q[start:stop]
        ap = qq[:, None, :] - a[None, :, :]
        t = np.sum(ap * ab[None, :, :], axis=2) / ab2[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d2 = np.sum((qq[:, None, :] - closest) ** 2, axis=2)
        out[start:stop] = np.sqrt(np.min(d2, axis=1))

    return out


def contour_unsigned_distance(query_points, contour, origin, u, v):
    q2 = plane_coords(query_points, origin, u, v)
    a, b = contour_segments_2d(contour, origin, u, v)
    return point_to_segment_distance_2d(q2, a, b)


def signed_contour_sdf(query_points, surface, contour, origin, u, v):
    n = len(query_points)

    if contour is None:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=np.float32)

    distance = contour_unsigned_distance(query_points, contour, origin, u, v)
    valid = np.isfinite(distance)
    sign = compute_sign_libigl(surface, query_points)

    sdf = np.zeros(n, dtype=float)
    sdf[valid] = distance[valid] * sign[valid]

    return sdf, valid.astype(np.float32)


# ============================================================
# PIANI E INTERSEZIONI
# ============================================================

def build_plane_specs(c_area, a_maxd, t_area, scale_mm, square_spacing_mm, n_before_mitral, n_after_apex, plane_23_shift_mm):
    e1, e2, e3 = build_three_axes(c_area, a_maxd, t_area)
    spacing_norm = square_spacing_mm / scale_mm
    shift_lax_norm = plane_23_shift_mm / scale_mm

    sa_centers = make_parallel_plane_points(
        start_point=c_area,
        apex_point=a_maxd,
        axis=e1,
        spacing=spacing_norm,
        n_before_start=n_before_mitral,
        n_after_apex=n_after_apex,
    )

    c_long = c_area + shift_lax_norm * e1

    sa_specs = []
    for i, center in enumerate(sa_centers):
        sa_specs.append({
            "name": f"SA_{i:02d}",
            "center": center,
            "normal": e1,
            "u": e2,
            "v": e3,
        })

    lax1 = {
        "name": "LAX1",
        "center": c_long,
        "normal": e2,
        "u": e1,
        "v": e3,
    }

    lax2 = {
        "name": "LAX2",
        "center": c_long,
        "normal": e3,
        "u": e1,
        "v": e2,
    }

    return sa_specs, lax1, lax2


def plane_plane_intersection(origin1, normal1, origin2, normal2):
    o1 = np.asarray(origin1, dtype=float)
    o2 = np.asarray(origin2, dtype=float)
    n1 = normalize(normal1, "plane 1 normal")
    n2 = normalize(normal2, "plane 2 normal")

    direction = np.cross(n1, n2)
    norm_dir = np.linalg.norm(direction)

    if norm_dir < 1e-10:
        raise ValueError("Planes are parallel or nearly parallel")

    direction /= norm_dir

    A = np.vstack([n1, n2, direction])
    b = np.array([np.dot(n1, o1), np.dot(n2, o2), 0.0], dtype=float)

    point = np.linalg.solve(A, b)
    return point, direction


def line_sampling_interval_from_mesh(line_point, line_direction, mesh_points, margin_norm):
    rel = np.asarray(mesh_points, dtype=float) - line_point[None, :]
    t = rel @ line_direction
    return float(np.min(t) - margin_norm), float(np.max(t) + margin_norm)


def sample_line_with_n_points(line_point, line_direction, t_min, t_max, n_points):
    t = np.linspace(t_min, t_max, int(n_points))
    points = line_point[None, :] + t[:, None] * line_direction[None, :]
    return t, points


def plane_patch(center, u, v, size_u, size_v):
    u = normalize(u)
    v = normalize(v)
    c = np.asarray(center, dtype=float)
    p0 = c - size_u * u - size_v * v
    p1 = c + size_u * u - size_v * v
    p2 = c - size_u * u + size_v * v
    p3 = c + size_u * u + size_v * v
    return pv.Quadrilateral([p0, p1, p3, p2])


def polyline_from_points(points):
    return pv.lines_from_points(np.asarray(points, dtype=float), close=False)


# ============================================================
# COSTRUZIONE CURVE SDF NEL PIANO LAX
# ============================================================

def build_sdf_graph_curve(base_points, sdf_norm, lax_normal, line_direction, scale_visual=1.0):
    """
    La curva resta nel piano LAX.

    base_points: punti q sulla retta SA x LAX
    sdf_norm: SDF in unità normalizzate
    lax_normal: normale del piano LAX
    line_direction: direzione della retta di intersezione

    graph_dir = cross(lax_normal, line_direction)
    così graph_dir è:
      - nel piano LAX
      - ortogonale alla retta di intersezione
    """
    n = normalize(lax_normal, "lax normal")
    d = normalize(line_direction, "line direction")
    g = normalize(np.cross(n, d), "graph direction in LAX plane")

    curve_points = np.asarray(base_points, dtype=float) + (scale_visual * sdf_norm)[:, None] * g[None, :]
    return curve_points, g


def analyze_pair(surface, epi_surface, sa_spec, lax_spec, scale_mm, n_line_points, line_margin_mm, sdf_visual_scale=1.0):
    contour_sa = slice_surface_with_plane(surface, sa_spec["center"], sa_spec["normal"])
    contour_lax = slice_surface_with_plane(surface, lax_spec["center"], lax_spec["normal"])

    line_point, line_direction = plane_plane_intersection(
        sa_spec["center"], sa_spec["normal"],
        lax_spec["center"], lax_spec["normal"],
    )

    t_min, t_max = line_sampling_interval_from_mesh(
        line_point=line_point,
        line_direction=line_direction,
        mesh_points=epi_surface.points,
        margin_norm=line_margin_mm / scale_mm,
    )

    t_norm, q = sample_line_with_n_points(
        line_point=line_point,
        line_direction=line_direction,
        t_min=t_min,
        t_max=t_max,
        n_points=n_line_points,
    )

    sdf_sa_norm, mask_sa = signed_contour_sdf(
        query_points=q,
        surface=surface,
        contour=contour_sa,
        origin=sa_spec["center"],
        u=sa_spec["u"],
        v=sa_spec["v"],
    )

    sdf_lax_norm, mask_lax = signed_contour_sdf(
        query_points=q,
        surface=surface,
        contour=contour_lax,
        origin=lax_spec["center"],
        u=lax_spec["u"],
        v=lax_spec["v"],
    )

    curve_sa_pts, graph_dir = build_sdf_graph_curve(
        base_points=q,
        sdf_norm=sdf_sa_norm,
        lax_normal=lax_spec["normal"],
        line_direction=line_direction,
        scale_visual=sdf_visual_scale,
    )

    curve_lax_pts, _ = build_sdf_graph_curve(
        base_points=q,
        sdf_norm=sdf_lax_norm,
        lax_normal=lax_spec["normal"],
        line_direction=line_direction,
        scale_visual=sdf_visual_scale,
    )

    table = pd.DataFrame({
        "t_mm": t_norm * scale_mm,
        "x_base": q[:, 0],
        "y_base": q[:, 1],
        "z_base": q[:, 2],
        "mask_sa": mask_sa.astype(int),
        "mask_lax": mask_lax.astype(int),
        "sdf_sa_mm": sdf_sa_norm * scale_mm,
        "sdf_lax_mm": sdf_lax_norm * scale_mm,
        "delta_mm": (sdf_lax_norm - sdf_sa_norm) * scale_mm,
        "curve_sa_x": curve_sa_pts[:, 0],
        "curve_sa_y": curve_sa_pts[:, 1],
        "curve_sa_z": curve_sa_pts[:, 2],
        "curve_lax_x": curve_lax_pts[:, 0],
        "curve_lax_y": curve_lax_pts[:, 1],
        "curve_lax_z": curve_lax_pts[:, 2],
    })

    return {
        "q": q,
        "t_mm": t_norm * scale_mm,
        "line_direction": line_direction,
        "graph_direction": graph_dir,
        "contour_sa": contour_sa,
        "contour_lax": contour_lax,
        "sdf_sa_norm": sdf_sa_norm,
        "sdf_lax_norm": sdf_lax_norm,
        "sdf_sa_mm": sdf_sa_norm * scale_mm,
        "sdf_lax_mm": sdf_lax_norm * scale_mm,
        "curve_sa_pts": curve_sa_pts,
        "curve_lax_pts": curve_lax_pts,
        "base_line_poly": polyline_from_points(q),
        "curve_sa_poly": polyline_from_points(curve_sa_pts),
        "curve_lax_poly": polyline_from_points(curve_lax_pts),
        "table": table,
    }


# ============================================================
# VISUALIZZAZIONE
# ============================================================

def visualize(
    patient,
    all_processed_dir,
    landmarks_csv,
    output_dir,
    organ="epicardium",
    sa_index=8,
    n_line_points=120,
    square_spacing_mm=6.0,
    n_before_mitral=3,
    n_after_apex=3,
    plane_23_shift_mm=25.0,
    line_margin_mm=10.0,
    sdf_visual_scale=1.0,
    show_other_surfaces=True,
):
    patient_dir = Path(all_processed_dir) / patient

    epi = prepare_surface(pv.read(patient_dir / "epicardium-processed.vtp"), "epicardium")
    lv = prepare_surface(pv.read(patient_dir / "lv_endo-processed.vtp"), "lv_endo")
    rv = prepare_surface(pv.read(patient_dir / "rv_endo-processed.vtp"), "rv_endo")

    surfaces_all = {
        "epicardium": epi,
        "lv_endo": lv,
        "rv_endo": rv,
    }

    if organ not in surfaces_all:
        raise ValueError(f"Unsupported organ '{organ}'")

    target_surface = surfaces_all[organ]

    c_area, a_maxd, t_area = read_landmarks(landmarks_csv, patient)

    if "scale-tooriginalrange" not in epi.field_data:
        raise KeyError("Missing epicardium field_data['scale-tooriginalrange']")
    scale_um = float(np.asarray(epi.field_data["scale-tooriginalrange"]).ravel()[0])
    scale_mm = scale_um / 1000.0

    sa_specs, lax1, lax2 = build_plane_specs(
        c_area=c_area,
        a_maxd=a_maxd,
        t_area=t_area,
        scale_mm=scale_mm,
        square_spacing_mm=square_spacing_mm,
        n_before_mitral=n_before_mitral,
        n_after_apex=n_after_apex,
        plane_23_shift_mm=plane_23_shift_mm,
    )

    if sa_index < 0 or sa_index >= len(sa_specs):
        raise IndexError(f"sa_index out of range [0, {len(sa_specs)-1}]")

    sa_spec = sa_specs[sa_index]

    pair1 = analyze_pair(
        surface=target_surface,
        epi_surface=epi,
        sa_spec=sa_spec,
        lax_spec=lax1,
        scale_mm=scale_mm,
        n_line_points=n_line_points,
        line_margin_mm=line_margin_mm,
        sdf_visual_scale=sdf_visual_scale,
    )

    pair2 = analyze_pair(
        surface=target_surface,
        epi_surface=epi,
        sa_spec=sa_spec,
        lax_spec=lax2,
        scale_mm=scale_mm,
        n_line_points=n_line_points,
        line_margin_mm=line_margin_mm,
        sdf_visual_scale=sdf_visual_scale,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv1 = output_dir / f"{patient}_{organ}_{sa_spec['name']}_LAX1_sdf_curves.csv"
    csv2 = output_dir / f"{patient}_{organ}_{sa_spec['name']}_LAX2_sdf_curves.csv"
    pair1["table"].to_csv(csv1, index=False)
    pair2["table"].to_csv(csv2, index=False)

    # patches dei piani
    bbox = np.asarray(epi.bounds, dtype=float)
    dx = bbox[1] - bbox[0]
    dy = bbox[3] - bbox[2]
    dz = bbox[5] - bbox[4]
    size = 0.6 * max(dx, dy, dz)

    sa_patch = plane_patch(sa_spec["center"], sa_spec["u"], sa_spec["v"], size, size)
    lax1_patch = plane_patch(lax1["center"], lax1["u"], lax1["v"], size, size)
    lax2_patch = plane_patch(lax2["center"], lax2["u"], lax2["v"], size, size)

    pl = pv.Plotter(window_size=(1600, 1200))

    # superfici
    if show_other_surfaces:
        for name, surf in surfaces_all.items():
            if name == organ:
                pl.add_mesh(surf, color="lightgray", opacity=0.16)
            else:
                pl.add_mesh(surf, color="silver", opacity=0.05)
    else:
        pl.add_mesh(target_surface, color="lightgray", opacity=0.16)

    # piani
    pl.add_mesh(sa_patch, color="red", opacity=0.10)
    pl.add_mesh(lax1_patch, color="dodgerblue", opacity=0.10)
    pl.add_mesh(lax2_patch, color="limegreen", opacity=0.10)

    # contour della superficie scelta
    if pair1["contour_sa"] is not None:
        pl.add_mesh(pair1["contour_sa"], color="red", line_width=5)
    if pair1["contour_lax"] is not None:
        pl.add_mesh(pair1["contour_lax"], color="dodgerblue", line_width=5)
    if pair2["contour_lax"] is not None:
        pl.add_mesh(pair2["contour_lax"], color="limegreen", line_width=5)

    # rette di intersezione
    pl.add_mesh(pair1["base_line_poly"], color="black", line_width=2)
    pl.add_mesh(pair2["base_line_poly"], color="dimgray", line_width=2)

    # curve SDF LAX1
    pl.add_mesh(pair1["curve_sa_poly"], color="crimson", line_width=6)
    pl.add_mesh(pair1["curve_lax_poly"], color="navy", line_width=6)

    # curve SDF LAX2
    pl.add_mesh(pair2["curve_sa_poly"], color="orange", line_width=6)
    pl.add_mesh(pair2["curve_lax_poly"], color="purple", line_width=6)

    # punti campionati sulla retta
    pl.add_points(pair1["q"], color="black", point_size=8, render_points_as_spheres=True)
    pl.add_points(pair2["q"], color="dimgray", point_size=8, render_points_as_spheres=True)

    # centri dei piani
    pl.add_points(np.asarray([sa_spec["center"]]), color="red", point_size=14, render_points_as_spheres=True)
    pl.add_points(np.asarray([lax1["center"]]), color="dodgerblue", point_size=14, render_points_as_spheres=True)
    pl.add_points(np.asarray([lax2["center"]]), color="limegreen", point_size=14, render_points_as_spheres=True)

    legend_entries = [
        ["SA plane / contour", "red"],
        ["LAX1 plane / contour", "dodgerblue"],
        ["LAX2 plane / contour", "limegreen"],
        ["Intersection SAxLAX1", "black"],
        ["Intersection SAxLAX2", "dimgray"],
        ["Curve SA on LAX1", "crimson"],
        ["Curve LAX1 on LAX1", "navy"],
        ["Curve SA on LAX2", "orange"],
        ["Curve LAX2 on LAX2", "purple"],
    ]
    pl.add_legend(legend_entries, bcolor="white")

    pl.add_axes()
    pl.add_text(
        f"{patient} | {organ}\n"
        f"Selected SA: {sa_spec['name']} (index {sa_index})\n"
        f"N points/line: {n_line_points} | visual scale: {sdf_visual_scale}",
        font_size=11,
    )

    print("\n" + "=" * 80)
    print("3D SDF CURVE VISUALIZATION READY")
    print("=" * 80)
    print("Patient:", patient)
    print("Organ:", organ)
    print("Selected SA plane:", sa_spec["name"], f"(index {sa_index})")
    print("N points per line:", n_line_points)
    print("Scale:", scale_mm, "mm / normalized unit")
    print("Visual scale multiplier:", sdf_visual_scale)
    print("Saved CSV:")
    print(" ", csv1)
    print(" ", csv2)
    print("=" * 80)

    pl.show()

    return {
        "csv_lax1": csv1,
        "csv_lax2": csv2,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--patient", type=str, default="AF001")
    parser.add_argument("--all_processed_dir", type=Path, required=True)
    parser.add_argument("--landmarks_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("visualize_sdf_curves_on_lax_plane"))

    parser.add_argument("--organ", type=str, default="epicardium", choices=["epicardium", "lv_endo", "rv_endo"])
    parser.add_argument("--sa_index", type=int, default=8, help="0-based index. 8 means the 9th SA plane.")
    parser.add_argument("--n_line_points", type=int, default=120)

    parser.add_argument("--square_spacing_mm", type=float, default=6.0)
    parser.add_argument("--n_before_mitral", type=int, default=3)
    parser.add_argument("--n_after_apex", type=int, default=3)
    parser.add_argument("--plane_23_shift_mm", type=float, default=25.0)
    parser.add_argument("--line_margin_mm", type=float, default=10.0)

    parser.add_argument("--sdf_visual_scale", type=float, default=1.0,
                        help="Moltiplicatore visivo delle curve SDF. 1.0 = nessuna esagerazione.")
    parser.add_argument("--hide_other_surfaces", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    visualize(
        patient=args.patient,
        all_processed_dir=args.all_processed_dir,
        landmarks_csv=args.landmarks_csv,
        output_dir=args.output_dir,
        organ=args.organ,
        sa_index=args.sa_index,
        n_line_points=args.n_line_points,
        square_spacing_mm=args.square_spacing_mm,
        n_before_mitral=args.n_before_mitral,
        n_after_apex=args.n_after_apex,
        plane_23_shift_mm=args.plane_23_shift_mm,
        line_margin_mm=args.line_margin_mm,
        sdf_visual_scale=args.sdf_visual_scale,
        show_other_surfaces=not args.hide_other_surfaces,
    )


if __name__ == "__main__":
    main()
