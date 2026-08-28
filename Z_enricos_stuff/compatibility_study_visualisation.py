#!/usr/bin/env python3
"""
visualize_SA_LAX_one_plane_SDF_3d.py

Per un paziente:
- prende un piano SA scelto (default: 9° piano, indice 8)
- prende LAX1 e LAX2
- campiona N punti sulle rette SAxLAX1 e SAxLAX2
- calcola la SDF dei punti rispetto al contour SA e al contour LAX
- mostra tutto in PyVista in 4 pannelli interattivi

Output:
- due CSV con i punti campionati e le SDF
- finestra interattiva PyVista
"""

from __future__ import annotations

import argparse
from pathlib import Path

import igl
import numpy as np
import pandas as pd
import pyvista as pv


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
        raise ValueError("Invalid spacing or geometry")
    n_full_steps = int(np.floor(axial_distance / spacing))
    out = []
    for i in range(n_before_start, 0, -1):
        out.append(start_point - i * spacing * axis)
    for i in range(n_full_steps + 1):
        out.append(start_point + i * spacing * axis)
    last_regular_distance = n_full_steps * spacing
    for i in range(1, n_after_apex + 1):
        out.append(start_point + (last_regular_distance + i * spacing) * axis)
    return np.asarray(out, dtype=float)


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
    lax1 = {"name": "LAX1", "center": c_long, "normal": e2, "u": e1, "v": e3}
    lax2 = {"name": "LAX2", "center": c_long, "normal": e3, "u": e1, "v": e2}
    return sa_specs, lax1, lax2


def analyze_one_pair(surface, sa_spec, lax_spec, extent_mesh_points, scale_mm, n_line_points, line_margin_mm):
    contour_sa = slice_surface_with_plane(surface, sa_spec["center"], sa_spec["normal"])
    contour_lax = slice_surface_with_plane(surface, lax_spec["center"], lax_spec["normal"])

    line_point, line_direction = plane_plane_intersection(
        sa_spec["center"], sa_spec["normal"],
        lax_spec["center"], lax_spec["normal"],
    )

    t_min, t_max = line_sampling_interval_from_mesh(
        line_point=line_point,
        line_direction=line_direction,
        mesh_points=extent_mesh_points,
        margin_norm=line_margin_mm / scale_mm,
    )

    t_norm, q = sample_line_with_n_points(line_point, line_direction, t_min, t_max, n_line_points)

    sdf_sa, mask_sa = signed_contour_sdf(
        query_points=q,
        surface=surface,
        contour=contour_sa,
        origin=sa_spec["center"],
        u=sa_spec["u"],
        v=sa_spec["v"],
    )
    sdf_lax, mask_lax = signed_contour_sdf(
        query_points=q,
        surface=surface,
        contour=contour_lax,
        origin=lax_spec["center"],
        u=lax_spec["u"],
        v=lax_spec["v"],
    )

    df = pd.DataFrame({
        "t_mm": t_norm * scale_mm,
        "x": q[:, 0],
        "y": q[:, 1],
        "z": q[:, 2],
        "mask_sa": mask_sa.astype(int),
        "mask_lax": mask_lax.astype(int),
        "valid_both": ((mask_sa > 0.5) & (mask_lax > 0.5)).astype(int),
        "sdf_from_sa_mm": sdf_sa * scale_mm,
        "sdf_from_lax_mm": sdf_lax * scale_mm,
        "delta_mm": (sdf_lax - sdf_sa) * scale_mm,
    })

    return {
        "contour_sa": contour_sa,
        "contour_lax": contour_lax,
        "q": q,
        "t_mm": t_norm * scale_mm,
        "sdf_sa_mm": sdf_sa * scale_mm,
        "sdf_lax_mm": sdf_lax * scale_mm,
        "mask_sa": mask_sa,
        "mask_lax": mask_lax,
        "table": df,
        "line_poly": pv.Line(q[0], q[-1], resolution=n_line_points - 1),
    }


def add_common_geometry(plotter, surfaces_all, organ_name, pair_data, sa_spec, lax_spec):
    for name, surf in surfaces_all.items():
        if name == organ_name:
            plotter.add_mesh(surf, opacity=0.15, color="lightgray")
        else:
            plotter.add_mesh(surf, opacity=0.05, color="silver")

    if pair_data["contour_sa"] is not None:
        plotter.add_mesh(pair_data["contour_sa"], color="red", line_width=4)
    if pair_data["contour_lax"] is not None:
        plotter.add_mesh(pair_data["contour_lax"], color="blue", line_width=4)

    plotter.add_mesh(pair_data["line_poly"], color="black", line_width=2)
    plotter.add_points(np.asarray([sa_spec["center"]]), color="red", point_size=12, render_points_as_spheres=True)
    plotter.add_points(np.asarray([lax_spec["center"]]), color="blue", point_size=12, render_points_as_spheres=True)
    plotter.add_axes()


def build_point_cloud(points, scalar_name, values):
    poly = pv.PolyData(np.asarray(points, dtype=float))
    poly[scalar_name] = np.asarray(values, dtype=float)
    return poly


def visualize_patient(
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
    clim=None,
):
    patient_dir = Path(all_processed_dir) / patient

    epi = prepare_surface(pv.read(patient_dir / "epicardium-processed.vtp"), "epicardium")
    lv = prepare_surface(pv.read(patient_dir / "lv_endo-processed.vtp"), "lv_endo")
    rv = prepare_surface(pv.read(patient_dir / "rv_endo-processed.vtp"), "rv_endo")

    surfaces_all = {"epicardium": epi, "lv_endo": lv, "rv_endo": rv}
    if organ not in surfaces_all:
        raise ValueError(f"Unsupported organ: {organ}")

    target_surface = surfaces_all[organ]

    c_area, a_maxd, t_area = read_landmarks(landmarks_csv, patient)

    if "scale-tooriginalrange" not in epi.field_data:
        raise KeyError("Missing field_data['scale-tooriginalrange']")
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

    pair1 = analyze_one_pair(target_surface, sa_spec, lax1, epi.points, scale_mm, n_line_points, line_margin_mm)
    pair2 = analyze_one_pair(target_surface, sa_spec, lax2, epi.points, scale_mm, n_line_points, line_margin_mm)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv1 = output_dir / f"{patient}_{organ}_{sa_spec['name']}_LAX1_points.csv"
    csv2 = output_dir / f"{patient}_{organ}_{sa_spec['name']}_LAX2_points.csv"
    pair1["table"].to_csv(csv1, index=False)
    pair2["table"].to_csv(csv2, index=False)

    all_vals = np.concatenate([pair1["sdf_sa_mm"], pair1["sdf_lax_mm"], pair2["sdf_sa_mm"], pair2["sdf_lax_mm"]])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        auto_clim = (-1.0, 1.0)
    else:
        vmax = max(abs(float(np.min(finite_vals))), abs(float(np.max(finite_vals))))
        auto_clim = (-vmax, vmax)
    if clim is None:
        clim = auto_clim

    plotter = pv.Plotter(shape=(2, 2), window_size=(1700, 1200))

    plotter.subplot(0, 0)
    add_common_geometry(plotter, surfaces_all, organ, pair1, sa_spec, lax1)
    cloud = build_point_cloud(pair1["q"], "sdf_sa_mm", pair1["sdf_sa_mm"])
    plotter.add_mesh(cloud, scalars="sdf_sa_mm", render_points_as_spheres=True, point_size=12, clim=clim, scalar_bar_args={"title": "SDF from SA [mm]"})
    plotter.add_text(f"{patient} | {organ}\n{sa_spec['name']} x LAX1\nColor = SDF from SA", font_size=10)

    plotter.subplot(0, 1)
    add_common_geometry(plotter, surfaces_all, organ, pair1, sa_spec, lax1)
    cloud = build_point_cloud(pair1["q"], "sdf_lax_mm", pair1["sdf_lax_mm"])
    plotter.add_mesh(cloud, scalars="sdf_lax_mm", render_points_as_spheres=True, point_size=12, clim=clim, scalar_bar_args={"title": "SDF from LAX1 [mm]"})
    plotter.add_text(f"{patient} | {organ}\n{sa_spec['name']} x LAX1\nColor = SDF from LAX1", font_size=10)

    plotter.subplot(1, 0)
    add_common_geometry(plotter, surfaces_all, organ, pair2, sa_spec, lax2)
    cloud = build_point_cloud(pair2["q"], "sdf_sa_mm", pair2["sdf_sa_mm"])
    plotter.add_mesh(cloud, scalars="sdf_sa_mm", render_points_as_spheres=True, point_size=12, clim=clim, scalar_bar_args={"title": "SDF from SA [mm]"})
    plotter.add_text(f"{patient} | {organ}\n{sa_spec['name']} x LAX2\nColor = SDF from SA", font_size=10)

    plotter.subplot(1, 1)
    add_common_geometry(plotter, surfaces_all, organ, pair2, sa_spec, lax2)
    cloud = build_point_cloud(pair2["q"], "sdf_lax_mm", pair2["sdf_lax_mm"])
    plotter.add_mesh(cloud, scalars="sdf_lax_mm", render_points_as_spheres=True, point_size=12, clim=clim, scalar_bar_args={"title": "SDF from LAX2 [mm]"})
    plotter.add_text(f"{patient} | {organ}\n{sa_spec['name']} x LAX2\nColor = SDF from LAX2", font_size=10)

    print("\n" + "=" * 80)
    print("3D VISUALIZATION READY")
    print("=" * 80)
    print("Patient:", patient)
    print("Organ:", organ)
    print("Selected SA plane:", sa_spec["name"], f"(index {sa_index})")
    print("N points per line:", n_line_points)
    print("Scale:", scale_mm, "mm / normalized unit")
    print("CSV saved:")
    print(" ", csv1)
    print(" ", csv2)
    print("Color range [mm]:", clim)
    print("=" * 80)

    plotter.link_views()
    plotter.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, default="AF001")
    parser.add_argument("--all_processed_dir", required=True, type=Path)
    parser.add_argument("--landmarks_csv", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("visualize_sa_lax_one_plane"))
    parser.add_argument("--organ", type=str, default="epicardium", choices=["epicardium", "lv_endo", "rv_endo"])
    parser.add_argument("--sa_index", type=int, default=8, help="0-based. sa_index=8 means the 9th SA plane.")
    parser.add_argument("--n_line_points", type=int, default=120)
    parser.add_argument("--square_spacing_mm", type=float, default=6.0)
    parser.add_argument("--n_before_mitral", type=int, default=3)
    parser.add_argument("--n_after_apex", type=int, default=3)
    parser.add_argument("--plane_23_shift_mm", type=float, default=25.0)
    parser.add_argument("--line_margin_mm", type=float, default=10.0)
    parser.add_argument("--clim_min", type=float, default=None)
    parser.add_argument("--clim_max", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if (args.clim_min is None) ^ (args.clim_max is None):
        raise ValueError("Provide both --clim_min and --clim_max, or none.")

    clim = None
    if args.clim_min is not None and args.clim_max is not None:
        clim = (float(args.clim_min), float(args.clim_max))

    visualize_patient(
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
        clim=clim,
    )


if __name__ == "__main__":
    main()
