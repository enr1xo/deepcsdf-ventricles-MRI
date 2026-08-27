#!/usr/bin/env python3
"""
analyze_exact_SA_LAX_intersections.py
=====================================

Test "definitivo" di compatibilità SA-LAX usando GLI STESSI IDENTICI PUNTI q.

Per ogni coppia:
    Short Axis SA_i
    Long Axis LAX_j

1) costruisce i due piani esattamente come nel sampler;
2) calcola la loro linea di intersezione;
3) campiona punti q lungo quella linea;
4) calcola, nello STESSO punto q:
       SDF_SA(q)  = distanza 2D dal contour mesh∩SA_i
       SDF_LAX(q) = distanza 2D dal contour mesh∩LAX_j
5) usa lo stesso metodo del sampler contour-based:
       - contour = mesh.slice(...)
       - proiezione nel piano (u,v)
       - distanza minima 2D punto-segmento
       - segno da fast winding number sulla mesh 3D
6) confronta EPI/LV/RV:
       delta signed
       |delta|
       sign mismatch

Se una superficie non interseca uno dei due piani:
    SDF = 0
    mask = 0
e quella coppia non viene usata per il confronto di quella superficie.

Output:
    - CSV completo di tutti i punti q
    - CSV di riepilogo per coppia SA/LAX e superficie
    - grafici per le coppie con maggiore incompatibilità
"""

from __future__ import annotations

import argparse
from pathlib import Path

import igl
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt


ORGANS = ["epicardium", "lv_endo", "rv_endo"]


# ============================================================
# GEOMETRIA DI BASE
# ============================================================

def normalize(v, name="vector"):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError(f"{name} has zero norm")
    return v / n


def find_patient_column(df):
    for c in [
        "patient", "Patient", "PATIENT",
        "patient_id", "PatientID", "id", "ID"
    ]:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find patient column. Columns: {list(df.columns)}"
    )


def read_point(row, cols, name):
    if not all(c in row.index for c in cols):
        raise ValueError(f"Missing columns for {name}: {cols}")

    p = np.asarray(
        [row[cols[0]], row[cols[1]], row[cols[2]]],
        dtype=float,
    )

    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} has non-finite coordinates: {p}")

    return p


def read_patient_landmarks(csv_path, patient):
    df = pd.read_csv(csv_path, sep=";")
    pc = find_patient_column(df)

    rows = df[df[pc].astype(str) == str(patient)]

    if rows.empty:
        raise ValueError(f"Patient {patient} not found in CSV")

    if len(rows) != 1:
        raise ValueError(
            f"Patient {patient} appears {len(rows)} times in CSV"
        )

    row = rows.iloc[0]

    c_area = read_point(
        row,
        ["C_area_x", "C_area_y", "C_area_z"],
        "C_area",
    )

    apex = read_point(
        row,
        ["A_maxD_x", "A_maxD_y", "A_maxD_z"],
        "A_maxD",
    )

    t_area = read_point(
        row,
        ["T_area_x", "T_area_y", "T_area_z"],
        "T_area",
    )

    return c_area, apex, t_area


def build_three_axes(c_area, apex, t_area):
    e1 = normalize(apex - c_area, "e1")

    raw_e2 = normalize(
        t_area - c_area,
        "raw e2",
    )

    e2 = normalize(
        raw_e2 - np.dot(raw_e2, e1) * e1,
        "e2",
    )

    e3 = normalize(
        np.cross(e1, e2),
        "e3",
    )

    e2 = normalize(
        np.cross(e3, e1),
        "e2 right handed",
    )

    return e1, e2, e3


def prepare_surface(mesh, name):
    surface = (
        mesh
        .extract_surface(algorithm=None)
        .triangulate()
        .clean()
    )

    if surface.n_points == 0 or surface.n_cells == 0:
        raise ValueError(f"{name} is empty")

    if not surface.is_all_triangles:
        raise ValueError(f"{name} is not triangular")

    return surface


def make_parallel_plane_points(
    start,
    apex,
    axis,
    spacing,
    n_before,
    n_after,
):
    axis = normalize(axis, "SA translation axis")

    length = float(
        np.dot(apex - start, axis)
    )

    if length <= 0 or spacing <= 0:
        raise ValueError(
            "Invalid apex-axis geometry or spacing"
        )

    n_full = int(
        np.floor(length / spacing)
    )

    points = []

    for i in range(n_before, 0, -1):
        points.append(
            start - i * spacing * axis
        )

    for i in range(n_full + 1):
        points.append(
            start + i * spacing * axis
        )

    last = n_full * spacing

    for i in range(1, n_after + 1):
        points.append(
            start + (last + i * spacing) * axis
        )

    return np.asarray(points, dtype=float)


# ============================================================
# STESSO METODO DI DISTANZA DEL SAMPLER CONTOUR-BASED
# ============================================================

def compute_sign_libigl(mesh, query_points):
    V = np.asarray(mesh.points, dtype=np.float64)

    F4 = np.asarray(mesh.faces).reshape(-1, 4)

    if not np.all(F4[:, 0] == 3):
        raise ValueError("Mesh must be triangular")

    F = F4[:, 1:4].astype(np.int32)

    Q = np.asarray(
        query_points,
        dtype=np.float64,
    )

    w = igl.fast_winding_number(
        V=V,
        F=F,
        Q=Q,
    )

    return np.where(
        np.abs(w) > 0.5,
        -1.0,
        +1.0,
    )


def slice_surface_with_plane(
    surface,
    origin,
    normal,
):
    contour = surface.slice(
        normal=normalize(normal),
        origin=np.asarray(origin, dtype=float),
        generate_triangles=False,
    )

    if contour is None:
        return None

    contour = contour.clean()

    if (
        contour.n_points == 0
        or contour.n_cells == 0
        or np.asarray(contour.lines).size == 0
    ):
        return None

    return contour


def plane_coords(
    points,
    origin,
    u,
    v,
):
    local = (
        np.asarray(points, dtype=float)
        - np.asarray(origin, dtype=float)[None, :]
    )

    return np.column_stack(
        [
            local @ normalize(u),
            local @ normalize(v),
        ]
    )


def contour_segments_2d(
    contour,
    origin,
    u,
    v,
):
    if contour is None:
        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
        )

    p2 = plane_coords(
        contour.points,
        origin,
        u,
        v,
    )

    lines = np.asarray(
        contour.lines
    )

    starts = []
    ends = []

    k = 0

    while k < len(lines):
        n = int(lines[k])

        ids = lines[
            k + 1:
            k + 1 + n
        ]

        for j in range(max(0, n - 1)):
            starts.append(
                p2[ids[j]]
            )
            ends.append(
                p2[ids[j + 1]]
            )

        k += n + 1

    if not starts:
        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
        )

    return (
        np.asarray(starts),
        np.asarray(ends),
    )


def point_to_segment_distance_2d(
    query_2d,
    seg_a,
    seg_b,
    chunk=3000,
):
    q = np.asarray(
        query_2d,
        dtype=float,
    )

    a = np.asarray(
        seg_a,
        dtype=float,
    )

    b = np.asarray(
        seg_b,
        dtype=float,
    )

    if len(a) == 0:
        return np.full(
            len(q),
            np.nan,
        )

    ab = b - a

    ab2 = np.sum(
        ab * ab,
        axis=1,
    )

    ok = ab2 > 1e-20

    a = a[ok]
    ab = ab[ok]
    ab2 = ab2[ok]

    if len(a) == 0:
        return np.full(
            len(q),
            np.nan,
        )

    out = np.empty(
        len(q),
        dtype=float,
    )

    for start in range(
        0,
        len(q),
        chunk,
    ):
        stop = min(
            start + chunk,
            len(q),
        )

        qq = q[start:stop]

        ap = (
            qq[:, None, :]
            - a[None, :, :]
        )

        t = np.sum(
            ap * ab[None, :, :],
            axis=2,
        ) / ab2[None, :]

        t = np.clip(
            t,
            0.0,
            1.0,
        )

        closest = (
            a[None, :, :]
            + t[:, :, None]
            * ab[None, :, :]
        )

        d2 = np.sum(
            (
                qq[:, None, :]
                - closest
            ) ** 2,
            axis=2,
        )

        out[start:stop] = np.sqrt(
            np.min(
                d2,
                axis=1,
            )
        )

    return out


def contour_unsigned_distance(
    query_points,
    contour,
    origin,
    u,
    v,
):
    q2 = plane_coords(
        query_points,
        origin,
        u,
        v,
    )

    a, b = contour_segments_2d(
        contour,
        origin,
        u,
        v,
    )

    return point_to_segment_distance_2d(
        q2,
        a,
        b,
    )


def signed_contour_sdf(
    query_points,
    surface,
    contour,
    origin,
    u,
    v,
):
    n = len(query_points)

    if contour is None:
        return (
            np.zeros(n, dtype=float),
            np.zeros(n, dtype=np.float32),
        )

    distance = contour_unsigned_distance(
        query_points,
        contour,
        origin,
        u,
        v,
    )

    valid = np.isfinite(distance)

    sign = compute_sign_libigl(
        surface,
        query_points,
    )

    sdf = np.zeros(
        n,
        dtype=float,
    )

    sdf[valid] = (
        distance[valid]
        * sign[valid]
    )

    return (
        sdf,
        valid.astype(np.float32),
    )


# ============================================================
# INTERSEZIONE TRA DUE PIANI
# ============================================================

def plane_plane_intersection(
    origin1,
    normal1,
    origin2,
    normal2,
):
    """
    Restituisce:
        point_on_line
        line_direction

    per due piani non paralleli:
        n1·x = n1·o1
        n2·x = n2·o2
    """
    o1 = np.asarray(
        origin1,
        dtype=float,
    )

    o2 = np.asarray(
        origin2,
        dtype=float,
    )

    n1 = normalize(
        normal1,
        "plane 1 normal",
    )

    n2 = normalize(
        normal2,
        "plane 2 normal",
    )

    direction = np.cross(
        n1,
        n2,
    )

    norm_dir = np.linalg.norm(
        direction
    )

    if norm_dir < 1e-10:
        raise ValueError(
            "Planes are parallel or nearly parallel"
        )

    direction /= norm_dir

    # Troviamo il punto della linea più vicino all'origine:
    # solve:
    #   n1.x = d1
    #   n2.x = d2
    #   direction.x = 0
    A = np.vstack(
        [
            n1,
            n2,
            direction,
        ]
    )

    b = np.array(
        [
            np.dot(n1, o1),
            np.dot(n2, o2),
            0.0,
        ],
        dtype=float,
    )

    point = np.linalg.solve(
        A,
        b,
    )

    return point, direction


def line_sampling_interval_from_mesh(
    line_point,
    line_direction,
    mesh_points,
    margin_norm,
):
    """
    Proietta tutti i vertici epicardici sulla linea e usa
    l'intervallo min/max + margine.
    """
    rel = (
        np.asarray(mesh_points, dtype=float)
        - line_point[None, :]
    )

    t = rel @ line_direction

    return (
        float(np.min(t) - margin_norm),
        float(np.max(t) + margin_norm),
    )


def sample_line(
    line_point,
    line_direction,
    t_min,
    t_max,
    spacing_norm,
):
    if spacing_norm <= 0:
        raise ValueError(
            "Line sampling spacing must be positive"
        )

    t = np.arange(
        t_min,
        t_max + 0.5 * spacing_norm,
        spacing_norm,
    )

    points = (
        line_point[None, :]
        + t[:, None]
        * line_direction[None, :]
    )

    return t, points


# ============================================================
# ANALISI
# ============================================================

def build_plane_specs(
    c_area,
    apex,
    e1,
    e2,
    e3,
    square_spacing_norm,
    n_before_mitral,
    n_after_apex,
    shift_lax_norm,
):
    sa_centers = make_parallel_plane_points(
        c_area,
        apex,
        e1,
        square_spacing_norm,
        n_before_mitral,
        n_after_apex,
    )

    c_long = (
        c_area
        + shift_lax_norm * e1
    )

    sa_specs = []

    for i, center in enumerate(
        sa_centers
    ):
        sa_specs.append(
            {
                "name": f"SA_{i:02d}",
                "origin": center,
                "normal": e1,
                "u": e2,
                "v": e3,
            }
        )

    lax_specs = [
        {
            "name": "LAX1",
            "origin": c_long,
            "normal": e2,
            "u": e1,
            "v": e3,
        },
        {
            "name": "LAX2",
            "origin": c_long,
            "normal": e3,
            "u": e1,
            "v": e2,
        },
    ]

    return sa_specs, lax_specs


def analyze_patient(
    patient,
    all_processed_dir,
    landmarks_csv,
    output_dir,
    square_spacing_mm=6.0,
    n_before_mitral=3,
    n_after_apex=3,
    plane_23_shift_mm=25.0,
    line_spacing_mm=0.5,
    line_margin_mm=10.0,
    top_k_plots=8,
    show=False,
):
    patient_dir = (
        Path(all_processed_dir)
        / patient
    )

    epi = prepare_surface(
        pv.read(
            patient_dir
            / "epicardium-processed.vtp"
        ),
        "epicardium",
    )

    lv = prepare_surface(
        pv.read(
            patient_dir
            / "lv_endo-processed.vtp"
        ),
        "lv_endo",
    )

    rv = prepare_surface(
        pv.read(
            patient_dir
            / "rv_endo-processed.vtp"
        ),
        "rv_endo",
    )

    surfaces = {
        "epicardium": epi,
        "lv_endo": lv,
        "rv_endo": rv,
    }

    c_area, apex, t_area = read_patient_landmarks(
        landmarks_csv,
        patient,
    )

    e1, e2, e3 = build_three_axes(
        c_area,
        apex,
        t_area,
    )

    if "scale-tooriginalrange" not in epi.field_data:
        raise KeyError(
            "Missing epicardium field_data['scale-tooriginalrange']"
        )

    scale_um = float(
        np.asarray(
            epi.field_data[
                "scale-tooriginalrange"
            ]
        ).ravel()[0]
    )

    scale_mm = (
        scale_um
        / 1000.0
    )

    square_spacing_norm = (
        square_spacing_mm
        / scale_mm
    )

    shift_lax_norm = (
        plane_23_shift_mm
        / scale_mm
    )

    line_spacing_norm = (
        line_spacing_mm
        / scale_mm
    )

    line_margin_norm = (
        line_margin_mm
        / scale_mm
    )

    sa_specs, lax_specs = build_plane_specs(
        c_area=c_area,
        apex=apex,
        e1=e1,
        e2=e2,
        e3=e3,
        square_spacing_norm=square_spacing_norm,
        n_before_mitral=n_before_mitral,
        n_after_apex=n_after_apex,
        shift_lax_norm=shift_lax_norm,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    pair_cache = {}

    print("\n" + "=" * 80)
    print("EXACT SA-LAX INTERSECTION ANALYSIS")
    print("=" * 80)
    print("Patient:", patient)
    print("scale_mm:", scale_mm)
    print("SA planes:", len(sa_specs))
    print("LAX planes:", len(lax_specs))
    print("line spacing:", line_spacing_mm, "mm")
    print("=" * 80)

    for sa_idx, sa in enumerate(
        sa_specs
    ):
        for lax in lax_specs:
            print(
                f"{sa['name']} x {lax['name']}"
            )

            line_point, line_direction = (
                plane_plane_intersection(
                    sa["origin"],
                    sa["normal"],
                    lax["origin"],
                    lax["normal"],
                )
            )

            t_min, t_max = (
                line_sampling_interval_from_mesh(
                    line_point=line_point,
                    line_direction=line_direction,
                    mesh_points=epi.points,
                    margin_norm=line_margin_norm,
                )
            )

            t, q = sample_line(
                line_point=line_point,
                line_direction=line_direction,
                t_min=t_min,
                t_max=t_max,
                spacing_norm=line_spacing_norm,
            )

            pair_key = (
                sa["name"],
                lax["name"],
            )

            pair_cache[pair_key] = {
                "t_mm": t * scale_mm,
                "q": q,
                "organ_data": {},
            }

            for organ in ORGANS:
                surface = surfaces[
                    organ
                ]

                contour_sa = slice_surface_with_plane(
                    surface,
                    sa["origin"],
                    sa["normal"],
                )

                contour_lax = slice_surface_with_plane(
                    surface,
                    lax["origin"],
                    lax["normal"],
                )

                sdf_sa, mask_sa = signed_contour_sdf(
                    query_points=q,
                    surface=surface,
                    contour=contour_sa,
                    origin=sa["origin"],
                    u=sa["u"],
                    v=sa["v"],
                )

                sdf_lax, mask_lax = signed_contour_sdf(
                    query_points=q,
                    surface=surface,
                    contour=contour_lax,
                    origin=lax["origin"],
                    u=lax["u"],
                    v=lax["v"],
                )

                valid = (
                    (mask_sa > 0.5)
                    & (mask_lax > 0.5)
                )

                delta_signed = (
                    sdf_lax
                    - sdf_sa
                )

                delta_abs = np.abs(
                    delta_signed
                )

                sign_mismatch = np.full(
                    len(q),
                    np.nan,
                    dtype=float,
                )

                sign_defined = (
                    valid
                    & (np.abs(sdf_sa) > 1e-12)
                    & (np.abs(sdf_lax) > 1e-12)
                )

                sign_mismatch[
                    sign_defined
                ] = (
                    np.sign(
                        sdf_sa[sign_defined]
                    )
                    !=
                    np.sign(
                        sdf_lax[sign_defined]
                    )
                ).astype(float)

                pair_cache[pair_key]["organ_data"][
                    organ
                ] = {
                    "sdf_sa_mm": sdf_sa * scale_mm,
                    "sdf_lax_mm": sdf_lax * scale_mm,
                    "valid": valid,
                }

                for k in range(
                    len(q)
                ):
                    rows.append(
                        {
                            "patient": patient,
                            "sa_plane": sa["name"],
                            "lax_plane": lax["name"],
                            "organ": organ,
                            "sample_index_on_line": k,
                            "t_mm": t[k] * scale_mm,
                            "x": q[k, 0],
                            "y": q[k, 1],
                            "z": q[k, 2],
                            "mask_sa": int(mask_sa[k] > 0.5),
                            "mask_lax": int(mask_lax[k] > 0.5),
                            "valid_both": int(valid[k]),
                            "sdf_sa_mm": (
                                sdf_sa[k] * scale_mm
                                if mask_sa[k] > 0.5
                                else np.nan
                            ),
                            "sdf_lax_mm": (
                                sdf_lax[k] * scale_mm
                                if mask_lax[k] > 0.5
                                else np.nan
                            ),
                            "delta_signed_mm": (
                                delta_signed[k] * scale_mm
                                if valid[k]
                                else np.nan
                            ),
                            "delta_abs_mm": (
                                delta_abs[k] * scale_mm
                                if valid[k]
                                else np.nan
                            ),
                            "sign_mismatch": (
                                sign_mismatch[k]
                            ),
                        }
                    )

    full_df = pd.DataFrame(
        rows
    )

    full_csv = (
        output_dir
        / f"{patient}_exact_SA_LAX_intersections.csv"
    )

    full_df.to_csv(
        full_csv,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = []

    grouped = full_df.groupby(
        [
            "sa_plane",
            "lax_plane",
            "organ",
        ],
        sort=True,
    )

    for (
        sa_name,
        lax_name,
        organ,
    ), g in grouped:
        valid = g[
            g["valid_both"] > 0
        ]

        if len(valid) == 0:
            summary_rows.append(
                {
                    "sa_plane": sa_name,
                    "lax_plane": lax_name,
                    "organ": organ,
                    "n_valid": 0,
                }
            )
            continue

        delta = valid[
            "delta_abs_mm"
        ].to_numpy()

        mismatch = valid[
            "sign_mismatch"
        ].dropna().to_numpy()

        summary_rows.append(
            {
                "sa_plane": sa_name,
                "lax_plane": lax_name,
                "organ": organ,
                "n_valid": len(valid),
                "mean_abs_delta_mm": float(
                    np.mean(delta)
                ),
                "median_abs_delta_mm": float(
                    np.median(delta)
                ),
                "p95_abs_delta_mm": float(
                    np.percentile(
                        delta,
                        95,
                    )
                ),
                "max_abs_delta_mm": float(
                    np.max(delta)
                ),
                "sign_mismatch_fraction": (
                    float(
                        np.mean(mismatch)
                    )
                    if len(mismatch)
                    else np.nan
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        output_dir
        / f"{patient}_exact_SA_LAX_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    print("\nTOP INCOMPATIBLE PAIRS")
    print("-" * 80)

    ranked = (
        summary_df[
            summary_df["n_valid"] > 0
        ]
        .sort_values(
            "p95_abs_delta_mm",
            ascending=False,
        )
    )

    if len(ranked):
        print(
            ranked[
                [
                    "sa_plane",
                    "lax_plane",
                    "organ",
                    "n_valid",
                    "median_abs_delta_mm",
                    "p95_abs_delta_mm",
                    "max_abs_delta_mm",
                    "sign_mismatch_fraction",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # PLOT TOP-K
    # --------------------------------------------------------

    plot_dir = (
        output_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for _, row in ranked.head(
        top_k_plots
    ).iterrows():
        key = (
            row["sa_plane"],
            row["lax_plane"],
        )

        organ = row["organ"]

        cached = pair_cache[key]

        t_mm = cached["t_mm"]

        od = cached[
            "organ_data"
        ][organ]

        valid = od["valid"]

        fig, ax = plt.subplots(
            figsize=(10, 5),
        )

        ax.plot(
            t_mm[valid],
            od["sdf_sa_mm"][valid],
            label="SDF from SA contour",
            linewidth=2,
        )

        ax.plot(
            t_mm[valid],
            od["sdf_lax_mm"][valid],
            label="SDF from LAX contour",
            linewidth=2,
        )

        ax.axhline(
            0.0,
            linewidth=1,
        )

        ax.set_xlabel(
            "position along exact SA-LAX intersection line [mm]"
        )

        ax.set_ylabel(
            "SDF [mm]"
        )

        ax.set_title(
            f"{patient} | "
            f"{row['sa_plane']} x {row['lax_plane']} | "
            f"{organ}\n"
            f"median |Δ|={row['median_abs_delta_mm']:.2f} mm, "
            f"P95 |Δ|={row['p95_abs_delta_mm']:.2f} mm"
        )

        ax.grid(
            True,
            alpha=0.2,
        )

        ax.legend()

        fig.tight_layout()

        plot_path = (
            plot_dir
            / (
                f"{patient}_"
                f"{row['sa_plane']}_"
                f"{row['lax_plane']}_"
                f"{organ}.png"
            )
        )

        fig.savefig(
            plot_path,
            dpi=200,
            bbox_inches="tight",
        )

        if show:
            plt.show()
        else:
            plt.close(fig)

    print("\nSaved:")
    print("Full CSV:", full_csv)
    print("Summary CSV:", summary_csv)
    print("Plots:", plot_dir)

    return (
        full_df,
        summary_df,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patient",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--all_processed_dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--landmarks_csv",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "exact_sa_lax_compatibility"
        ),
    )

    parser.add_argument(
        "--square_spacing_mm",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--n_before_mitral",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--n_after_apex",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--plane_23_shift_mm",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--line_spacing_mm",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--line_margin_mm",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--top_k_plots",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    analyze_patient(
        patient=args.patient,
        all_processed_dir=args.all_processed_dir,
        landmarks_csv=args.landmarks_csv,
        output_dir=args.output_dir,
        square_spacing_mm=args.square_spacing_mm,
        n_before_mitral=args.n_before_mitral,
        n_after_apex=args.n_after_apex,
        plane_23_shift_mm=args.plane_23_shift_mm,
        line_spacing_mm=args.line_spacing_mm,
        line_margin_mm=args.line_margin_mm,
        top_k_plots=args.top_k_plots,
        show=args.show,
    )


if __name__ == "__main__":
    main()
