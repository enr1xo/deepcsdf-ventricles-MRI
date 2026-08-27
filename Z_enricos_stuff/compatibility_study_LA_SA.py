#!/usr/bin/env python3
"""
Analyze compatibility between Short-Axis (SA) and Long-Axis (LAX) samples
stored in the same DeepSDF MRI-like NPY file.

Expected NPY columns:
[x, y, z, sdf_epi, sdf_lv, sdf_rv, mask_epi, mask_lv, mask_rv]

Default ordering assumption:
    all SA samples
    LAX1: penultimate 2000 samples
    LAX2: last 2000 samples

For each LAX point, the script finds the nearest SA point in 3D and compares
the SDF targets for EPI/LV/RV only when both masks are valid.

Outputs:
- CSV with all LAX->nearest-SA pairs
- CSV summary for several spatial thresholds
- distance-vs-deltaSDF plots
- SA-SDF-vs-LAX-SDF plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

try:
    import pyvista as pv
except ImportError:
    pv = None


ORGANS = ["epicardium", "lv_endo", "rv_endo"]


def read_scale_mm(scale_mm=None, epi_mesh=None):
    if scale_mm is not None:
        if scale_mm <= 0:
            raise ValueError("--scale_mm must be positive")
        return float(scale_mm), True

    if epi_mesh is not None:
        if pv is None:
            raise ImportError("PyVista is required when using --epi_mesh")
        mesh = pv.read(epi_mesh)
        if "scale-tooriginalrange" not in mesh.field_data:
            raise KeyError("Missing field_data['scale-tooriginalrange']")
        scale_um = float(np.asarray(mesh.field_data["scale-tooriginalrange"]).ravel()[0])
        return scale_um / 1000.0, True

    return 1.0, False


def split_blocks(data, n_lax_per_plane):
    n_lax_total = 2 * n_lax_per_plane
    if data.ndim != 2 or data.shape[1] < 9:
        raise ValueError(f"Expected N x >=9 array, got {data.shape}")
    if data.shape[0] <= n_lax_total:
        raise ValueError("Not enough rows for requested LAX blocks")

    sa = data[:-n_lax_total]
    lax1 = data[-n_lax_total:-n_lax_per_plane]
    lax2 = data[-n_lax_per_plane:]
    return sa, lax1, lax2


def pair_lax_to_sa(sa, lax, lax_name, scale_mm):
    xyz_sa = sa[:, :3]
    xyz_lax = lax[:, :3]

    sdf_sa = sa[:, 3:6]
    sdf_lax = lax[:, 3:6]

    mask_sa = sa[:, 6:9]
    mask_lax = lax[:, 6:9]

    tree = cKDTree(xyz_sa)
    dist_norm, idx_sa = tree.query(xyz_lax, k=1, workers=-1)

    nearest_xyz_sa = xyz_sa[idx_sa]
    nearest_sdf_sa = sdf_sa[idx_sa]
    nearest_mask_sa = mask_sa[idx_sa]

    out = {
        "lax_plane": [lax_name] * len(lax),
        "lax_index": np.arange(len(lax)),
        "nearest_sa_index": idx_sa,
        "distance_xyz_norm": dist_norm,
        "distance_xyz_mm": dist_norm * scale_mm,
        "lax_x": xyz_lax[:, 0],
        "lax_y": xyz_lax[:, 1],
        "lax_z": xyz_lax[:, 2],
        "sa_x": nearest_xyz_sa[:, 0],
        "sa_y": nearest_xyz_sa[:, 1],
        "sa_z": nearest_xyz_sa[:, 2],
    }

    eps = 1e-12

    for j, organ in enumerate(ORGANS):
        valid = (mask_lax[:, j] > 0.5) & (nearest_mask_sa[:, j] > 0.5)

        s_lax = sdf_lax[:, j]
        s_sa = nearest_sdf_sa[:, j]

        delta_signed = s_lax - s_sa
        delta_abs = np.abs(delta_signed)

        ratio = np.full(len(lax), np.nan, dtype=float)
        ratio[valid] = delta_abs[valid] / np.maximum(dist_norm[valid], eps)

        mismatch = np.full(len(lax), np.nan, dtype=float)
        sign_defined = (np.abs(s_lax) > eps) & (np.abs(s_sa) > eps)
        sel = valid & sign_defined
        mismatch[sel] = (np.sign(s_lax[sel]) != np.sign(s_sa[sel])).astype(float)

        out[f"{organ}_valid"] = valid.astype(int)
        out[f"{organ}_sdf_lax_norm"] = np.where(valid, s_lax, np.nan)
        out[f"{organ}_sdf_sa_norm"] = np.where(valid, s_sa, np.nan)
        out[f"{organ}_sdf_lax_mm"] = np.where(valid, s_lax * scale_mm, np.nan)
        out[f"{organ}_sdf_sa_mm"] = np.where(valid, s_sa * scale_mm, np.nan)
        out[f"{organ}_delta_signed_norm"] = np.where(valid, delta_signed, np.nan)
        out[f"{organ}_delta_abs_norm"] = np.where(valid, delta_abs, np.nan)
        out[f"{organ}_delta_signed_mm"] = np.where(valid, delta_signed * scale_mm, np.nan)
        out[f"{organ}_delta_abs_mm"] = np.where(valid, delta_abs * scale_mm, np.nan)
        out[f"{organ}_ratio_delta_over_xyz"] = ratio
        out[f"{organ}_sign_mismatch"] = mismatch

    return pd.DataFrame(out)


def summarize(df, thresholds, use_mm):
    rows = []
    dist_col = "distance_xyz_mm" if use_mm else "distance_xyz_norm"

    for lax_name in ["LAX1", "LAX2"]:
        dplane = df[df["lax_plane"] == lax_name]

        for organ in ORGANS:
            for threshold in thresholds:
                valid = (
                    (dplane[f"{organ}_valid"] > 0)
                    & (dplane[dist_col] <= threshold)
                )

                dsub = dplane.loc[valid]

                delta_col = f"{organ}_delta_abs_mm" if use_mm else f"{organ}_delta_abs_norm"
                xyz = dsub[dist_col].to_numpy()
                delta = dsub[delta_col].to_numpy()
                ratio = dsub[f"{organ}_ratio_delta_over_xyz"].to_numpy()
                mismatch = dsub[f"{organ}_sign_mismatch"].dropna().to_numpy()

                if len(dsub) == 0:
                    rows.append({
                        "lax_plane": lax_name,
                        "organ": organ,
                        "distance_threshold": threshold,
                        "n_pairs": 0,
                    })
                    continue

                rows.append({
                    "lax_plane": lax_name,
                    "organ": organ,
                    "distance_threshold": threshold,
                    "n_pairs": len(dsub),
                    "xyz_mean": np.mean(xyz),
                    "xyz_median": np.median(xyz),
                    "xyz_p95": np.percentile(xyz, 95),
                    "delta_sdf_mean": np.mean(delta),
                    "delta_sdf_median": np.median(delta),
                    "delta_sdf_p95": np.percentile(delta, 95),
                    "delta_sdf_max": np.max(delta),
                    "ratio_mean": np.mean(ratio),
                    "ratio_median": np.median(ratio),
                    "ratio_p95": np.percentile(ratio, 95),
                    "fraction_ratio_gt_1": np.mean(ratio > 1.0),
                    "sign_mismatch_fraction": np.mean(mismatch) if len(mismatch) else np.nan,
                })

    return pd.DataFrame(rows)


def plot_distance_vs_delta(df, lax_name, out_path, use_mm, show=False):
    dplane = df[df["lax_plane"] == lax_name]
    dist_col = "distance_xyz_mm" if use_mm else "distance_xyz_norm"
    unit = "mm" if use_mm else "normalized units"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    global_max = 0.0

    for j, organ in enumerate(ORGANS):
        valid = dplane[f"{organ}_valid"] > 0
        x = dplane.loc[valid, dist_col].to_numpy()
        delta_col = f"{organ}_delta_abs_mm" if use_mm else f"{organ}_delta_abs_norm"
        y = dplane.loc[valid, delta_col].to_numpy()

        if len(x):
            global_max = max(global_max, float(np.max(x)), float(np.max(y)))

        axes[j].scatter(x, y, s=10, alpha=0.35)
        axes[j].set_title(organ)
        axes[j].set_xlabel(f"nearest SA distance [{unit}]")
        axes[j].set_ylabel(f"|SDF_LAX - SDF_SA| [{unit}]")
        axes[j].grid(True, alpha=0.2)

    if global_max <= 0:
        global_max = 1.0

    for ax in axes:
        ax.plot([0, global_max], [0, global_max], "--", linewidth=1.5, label="y = x")
        ax.set_xlim(0, global_max)
        ax.set_ylim(0, global_max)
        ax.legend()

    fig.suptitle(f"{lax_name}: SA-LAX target compatibility")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_sdf_vs_sdf(df, lax_name, out_path, use_mm, max_distance, show=False):
    dplane = df[df["lax_plane"] == lax_name].copy()
    dist_col = "distance_xyz_mm" if use_mm else "distance_xyz_norm"
    dplane = dplane[dplane[dist_col] <= max_distance]

    suffix = "mm" if use_mm else "norm"
    unit = "mm" if use_mm else "normalized units"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    global_absmax = 0.0

    for j, organ in enumerate(ORGANS):
        valid = dplane[f"{organ}_valid"] > 0
        x = dplane.loc[valid, f"{organ}_sdf_sa_{suffix}"].to_numpy()
        y = dplane.loc[valid, f"{organ}_sdf_lax_{suffix}"].to_numpy()

        if len(x):
            global_absmax = max(
                global_absmax,
                float(np.max(np.abs(x))),
                float(np.max(np.abs(y))),
            )

        axes[j].scatter(x, y, s=10, alpha=0.35)
        axes[j].axhline(0.0, linewidth=0.8)
        axes[j].axvline(0.0, linewidth=0.8)
        axes[j].set_title(organ)
        axes[j].set_xlabel(f"SDF SA [{unit}]")
        axes[j].set_ylabel(f"SDF LAX [{unit}]")
        axes[j].grid(True, alpha=0.2)

    if global_absmax <= 0:
        global_absmax = 1.0

    for ax in axes:
        ax.plot(
            [-global_absmax, global_absmax],
            [-global_absmax, global_absmax],
            "--",
            linewidth=1.5,
            label="SA = LAX",
        )
        ax.set_xlim(-global_absmax, global_absmax)
        ax.set_ylim(-global_absmax, global_absmax)
        ax.legend()

    fig.suptitle(
        f"{lax_name}: SDF SA vs LAX for pairs within {max_distance:g} {unit}"
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True, type=Path)
    p.add_argument("--n_lax_per_plane", type=int, default=1000)
    p.add_argument("--scale_mm", type=float, default=None)
    p.add_argument("--epi_mesh", type=Path, default=None)
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0],
    )
    p.add_argument("--plot_pair_threshold", type=float, default=1.0)
    p.add_argument("--output_dir", type=Path, default=Path("sa_lax_compatibility"))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    data = np.load(args.npy)

    scale_mm, use_mm = read_scale_mm(
        scale_mm=args.scale_mm,
        epi_mesh=args.epi_mesh,
    )

    sa, lax1, lax2 = split_blocks(
        data,
        n_lax_per_plane=args.n_lax_per_plane,
    )

    print("\nINPUT")
    print("NPY:", args.npy)
    print("Full shape:", data.shape)
    print("SA:", sa.shape)
    print("LAX1:", lax1.shape)
    print("LAX2:", lax2.shape)

    if use_mm:
        print("Scale:", scale_mm, "mm / normalized unit")
    else:
        print("No scale supplied: using normalized units.")

    df1 = pair_lax_to_sa(sa, lax1, "LAX1", scale_mm)
    df2 = pair_lax_to_sa(sa, lax2, "LAX2", scale_mm)

    pairs = pd.concat([df1, df2], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = args.output_dir / f"{args.npy.stem}_SA_LAX_pairs.csv"
    pairs.to_csv(pairs_path, index=False)

    summary = summarize(
        pairs,
        thresholds=args.thresholds,
        use_mm=use_mm,
    )

    summary_path = args.output_dir / f"{args.npy.stem}_SA_LAX_summary.csv"
    summary.to_csv(summary_path, index=False)

    unit = "mm" if use_mm else "normalized units"

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for _, row in summary.iterrows():
        if row["n_pairs"] == 0:
            continue
        print(
            f"{row['lax_plane']:4s} | "
            f"{row['organ']:11s} | "
            f"d <= {row['distance_threshold']:g} {unit} | "
            f"N={int(row['n_pairs']):4d} | "
            f"median ΔSDF={row['delta_sdf_median']:.4f} | "
            f"P95 ΔSDF={row['delta_sdf_p95']:.4f} | "
            f"median ratio={row['ratio_median']:.2f} | "
            f"ratio>1={100*row['fraction_ratio_gt_1']:.1f}% | "
            f"sign mismatch={100*row['sign_mismatch_fraction']:.1f}%"
        )

    for lax_name in ["LAX1", "LAX2"]:
        plot_distance_vs_delta(
            pairs,
            lax_name,
            args.output_dir / f"{args.npy.stem}_{lax_name}_distance_vs_deltaSDF.png",
            use_mm,
            args.show,
        )

        plot_sdf_vs_sdf(
            pairs,
            lax_name,
            args.output_dir / f"{args.npy.stem}_{lax_name}_SDF_SA_vs_LAX.png",
            use_mm,
            args.plot_pair_threshold,
            args.show,
        )

    print("\nSaved:")
    print("Pairs CSV:", pairs_path)
    print("Summary CSV:", summary_path)
    print("Plots dir:", args.output_dir)


if __name__ == "__main__":
    main()
