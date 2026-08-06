#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv

from MRI_dataset_tools_3axis_new_distance import (
    ThreeAxisMRIParams,
    build_three_axes,
    build_slab_patches,
    make_square,
    prepare_surface,
    read_patient_three_points,
    sample_inside_epi_or_near_slab_surface_min_dist,
)

PATIENT = "AF006"
ALL_PROCESSED_DIR = Path("/home/rizzardi/Schreibtisch/AF001_aligned_processed")
CSV_PATH = Path("/home/rizzardi/Schreibtisch/MRI_model/mitral_apex_tricuspid_locations.csv")

PARAMS = ThreeAxisMRIParams(
    square_spacing_mm=6.0,
    slab_width_mm=0.1,
    n_before_mitral=3,
    n_after_apex=3,
    square_margin_factor=1.5,
    n_points_per_square=1000,
    min_dist_mm=1.0,
    contour_expansion_mm=25.0,
    batch_size=5000,
    plane_23_shift_mm=25.0,
    save_npy=False,
    save_csv=False,
    plot_debug=False,
)

SHOW_SAMPLED_POINTS = False
SHOW_PATCH_PROJECTION = True
PLANE_SIZE_FACTOR = 1.20


def get_scale_to_original_mm(mesh):
    values = np.asarray(mesh.field_data["scale-tooriginalrange"]).ravel()
    return float(values[0]) / 1000.0


def project_points_to_plane(points, center, normal):
    normal = normal / np.linalg.norm(normal)
    depth = np.dot(points - center, normal)
    projected = points - depth[:, None] * normal[None, :]
    return projected, depth


def estimate_square_side(epi, plane_specs, slab_half_width, margin):
    patches = build_slab_patches(epi, plane_specs, slab_half_width)
    ranges = []
    for spec, patch in zip(plane_specs, patches):
        if patch is None or patch.n_points == 0:
            continue
        local = patch.points - spec["center"]
        ranges.extend([
            float(np.ptp(np.dot(local, spec["u"]))),
            float(np.ptp(np.dot(local, spec["v"]))),
        ])
    ranges = [r for r in ranges if np.isfinite(r) and r > 0]
    if not ranges:
        raise RuntimeError("No long-axis epicardial slab patch found.")
    return margin * max(ranges), patches


def print_diagnostics(spec, patch, slab_half_width, scale_mm):
    print("\n" + "=" * 72)
    print(spec["type"])
    print("=" * 72)
    if patch is None:
        print("NO PATCH")
        return
    depth = np.dot(patch.points - spec["center"], spec["normal"])
    print("points:", patch.n_points)
    print("cells :", patch.n_cells)
    print("depth min/max [mm]:", depth.min() * scale_mm, depth.max() * scale_mm)
    print("expected [mm]:", -slab_half_width * scale_mm, slab_half_width * scale_mm)
    print("open edges:", patch.n_open_edges)


def add_debug(plotter, epi, spec, patch, square_side, slab_half_width,
              scale_mm, color, sampled_points):
    center, normal, u, v = spec["center"], spec["normal"], spec["u"], spec["v"]
    side = square_side * PLANE_SIZE_FACTOR

    central = make_square(center, u, v, side)
    lower = make_square(center - slab_half_width * normal, u, v, side)
    upper = make_square(center + slab_half_width * normal, u, v, side)

    plotter.add_mesh(central, color=color, opacity=0.15, show_edges=True,
                     label=f'{spec["type"]}: central plane')
    plotter.add_mesh(lower, color=color, opacity=0.06, show_edges=True,
                     label=f'{spec["type"]}: slab limits')
    plotter.add_mesh(upper, color=color, opacity=0.06, show_edges=True)

    if patch is None:
        plotter.add_point_labels([center], ["NO PATCH"], text_color=color)
        return

    patch_show = patch.copy()
    depth_mm = np.dot(patch_show.points - center, normal) * scale_mm
    patch_show.point_data["slab_depth_mm"] = depth_mm

    plotter.add_mesh(
        patch_show,
        scalars="slab_depth_mm",
        opacity=0.90,
        show_edges=True,
        scalar_bar_args={"title": f'{spec["type"]} depth [mm]'},
        label=f'{spec["type"]}: clipped epicardium',
    )

    edges = patch.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=True,
    )
    if edges.n_cells:
        plotter.add_mesh(edges, color=color, line_width=5,
                         label=f'{spec["type"]}: patch boundary')

    if SHOW_PATCH_PROJECTION:
        projected, _ = project_points_to_plane(patch.points, center, normal)
        plotter.add_mesh(
            pv.PolyData(projected),
            color=color,
            point_size=5,
            render_points_as_spheres=True,
            opacity=0.55,
            label=f'{spec["type"]}: projected patch',
        )

    if SHOW_SAMPLED_POINTS and sampled_points is not None:
        plotter.add_mesh(
            pv.PolyData(sampled_points),
            color="black",
            point_size=5,
            render_points_as_spheres=True,
            label=f'{spec["type"]}: samples',
        )


def main():
    epi_path = ALL_PROCESSED_DIR / PATIENT / "epicardium-processed.vtp"
    epi_raw = pv.read(epi_path)
    scale_mm = get_scale_to_original_mm(epi_raw)
    epi = prepare_surface(epi_raw, f"{PATIENT} epicardium")

    df = pd.read_csv(CSV_PATH, sep=";")
    c_area, a_maxd, t_area = read_patient_three_points(df, PATIENT)
    e1, e2, e3 = build_three_axes(c_area, a_maxd, t_area)

    c_long = c_area + (PARAMS.plane_23_shift_mm / scale_mm) * e1
    plane_specs = [
        {"type": "normal_e2", "center": c_long, "normal": e2, "u": e1, "v": e3},
        {"type": "normal_e3", "center": c_long, "normal": e3, "u": e1, "v": e2},
    ]

    slab_half_width = (PARAMS.slab_width_mm / 2.0) / scale_mm
    min_dist = PARAMS.min_dist_mm / scale_mm
    expansion = PARAMS.contour_expansion_mm / scale_mm

    square_side, patches = estimate_square_side(
        epi, plane_specs, slab_half_width, PARAMS.square_margin_factor
    )

    sampled = []
    for plane_id, (spec, patch) in enumerate(zip(plane_specs, patches)):
        print_diagnostics(spec, patch, slab_half_width, scale_mm)
        if SHOW_SAMPLED_POINTS:
            pts = sample_inside_epi_or_near_slab_surface_min_dist(
                square_center=spec["center"],
                u=spec["u"],
                v=spec["v"],
                side_length=square_side,
                epi_mesh=epi,
                epi_slab_patch=patch,
                n_points=PARAMS.n_points_per_square,
                min_dist=min_dist,
                surface_expansion=expansion,
                seed=42 + plane_id,
                max_trials=PARAMS.max_sampling_trials,
                batch_size=PARAMS.batch_size,
            )
        else:
            pts = None
        sampled.append(pts)

    plotter = pv.Plotter(shape=(1, 2), window_size=(1900, 900))
    colors = {"normal_e2": "orange", "normal_e3": "purple"}

    for j, (spec, patch, pts) in enumerate(zip(plane_specs, patches, sampled)):
        plotter.subplot(0, j)
        plotter.add_text(f'{PATIENT} - {spec["type"]}', font_size=16)
        plotter.add_mesh(epi, color="lightgray", opacity=0.08,
                         label="complete epicardium")
        add_debug(plotter, epi, spec, patch, square_side, slab_half_width,
                  scale_mm, colors[spec["type"]], pts)
        plotter.add_mesh(pv.Sphere(radius=0.025, center=c_area),
                         color="magenta", label="mitral centroid")
        plotter.add_mesh(pv.Sphere(radius=0.025, center=a_maxd),
                         color="black", label="apex")
        plotter.add_axes()
        plotter.show_bounds(grid="front", location="outer", all_edges=True)
        plotter.add_legend()

    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    main()
