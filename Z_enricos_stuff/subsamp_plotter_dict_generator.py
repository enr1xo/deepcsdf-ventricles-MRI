"""
questo codice prende in input un directory. 
in questa directory ci sono 3 cartelle, in ognuna delle quali vi sono due file csv.
un file csv riporta la chamfer distance rispetto a tre superfici per ogni paziente,
il secondo la stesas cosa ma la distance è la haussdorff.

questo codice andrà applicato a 2 directory, con la stessa struttura di cartelle.

voglio in outpur un dizioanrio del tipo:

nome_folder_di_cuipasso_la_directory = {
                                        nome_sottocartella_1: {
                                                                chamfer: {
                                                                            epi: (mean, std),
                                                                            lv: (mean, std),
                                                                            rv: (mean, std)
                                                                            }
                                                                haussdorff: {
                                                                            epi: (mean, std),
                                                                            lv: (mean, std),
                                                                            rv: (mean, std)
                                                                            }
                                                                }
                                        nome_sottocartella_2: {
                                                                chamfer: {
                                                                            epi: (mean, std),
                                                                            lv: (mean, std),
                                                                            rv: (mean, std)
                                                                            }
                                                                haussdorff: {
                                                                            epi: (mean, std),
                                                                            lv: (mean, std),
                                                                            rv: (mean, std)
                                                                            }
                                                                }  
                                        nome_sottocartella_3: {
                                                                chamfer: {
                                                                            epi: (mean, std),
                                                                            lv: (mean, std),
                                                                            rv: (mean, std)
                                                                            }
                                                                haussdorff: {
                                                                            epi: (mean, std),
                                                                            lv: (mean, std),
                                                                            rv: (mean, std)
                                                                            }
                                                                }                                 
                                        }
"""

#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from pprint import pprint


# ============================================================
# HELPERS
# ============================================================
ORGAN_NAME_MAP = {
    "epicardium": "epi",
    "lv_endo": "lv",
    "rv_endo": "rv",
}


def compute_mean_std(csv_file: Path):
    """
    Expected CSV columns:
        patient, organ, metric, value
    """

    df = pd.read_csv(csv_file)

    results = {}

    for organ_name, organ_short in ORGAN_NAME_MAP.items():

        organ_df = df[df["organ"] == organ_name]

        values = organ_df["value"].values.astype(float)

        results[organ_short] = (
            float(np.mean(values)),
            float(np.std(values))
        )

    return results


def find_metric_file(folder: Path, metric_name: str):
    """
    Finds csv containing metric_name in filename.
    """

    matches = list(folder.glob(f"*{metric_name}*.csv"))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No CSV found for metric '{metric_name}' in {folder}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple CSVs found for metric '{metric_name}' in {folder}"
        )

    return matches[0]


# ============================================================
# MAIN FUNCTION
# ============================================================
def summarize_results(root_dir: Path):

    root_dir = Path(root_dir)

    output = {}

    root_name = root_dir.name

    output[root_name] = {}

    # iterate over subfolders
    for subfolder in sorted(root_dir.iterdir()):

        if not subfolder.is_dir():
            continue

        print(f"\nProcessing: {subfolder.name}")

        chamfer_csv = find_metric_file(subfolder, "chamfer")
        hauss_csv = find_metric_file(subfolder, "haussdorff")

        chamfer_stats = compute_mean_std(chamfer_csv)
        hauss_stats = compute_mean_std(hauss_csv)

        output[root_name][subfolder.name] = {
            "chamfer": chamfer_stats,
            "haussdorff": hauss_stats,
        }

    return output


# ============================================================
# EXAMPLE
# ============================================================
if __name__ == "__main__":

    DIR1 = Path("/home/rizzardi/Schreibtisch/sdf_subsampling_test/sdf_magnitude_filter/5k_D7_W256_L128/results/case1_3surf_from2")
    DIR2 = Path("/home/rizzardi/Schreibtisch/sdf_subsampling_test/sdf_magnitude_filter/5k_D7_W256_L128/results/case2_sdf_filter")

    DIR_ORIGINAL = Path("/home/rizzardi/Schreibtisch/combinations/5k_architecture_exploration_extended/combs_data/5k_D7_W256_L128/results/metrics")


    # results_1 = summarize_results(DIR1)
    # results_2 = summarize_results(DIR2)

    # results_orig = summarize_results(DIR_ORIGINAL)

    # pprint(results_1)
    # pprint(results_2)
    # pprint(results_orig)

    case3_001_dir = Path("/home/rizzardi/Schreibtisch/sdf_subsampling_test/sdf_magnitude_filter/5k_D7_W256_L128/results/case3_sdf_filter_+_3surf_from_2/thresh_o001")
    case3_002_dir = Path("/home/rizzardi/Schreibtisch/sdf_subsampling_test/sdf_magnitude_filter/5k_D7_W256_L128/results/case3_sdf_filter_+_3surf_from_2/thresh_o002")
    case3_0005_dir = Path("/home/rizzardi/Schreibtisch/sdf_subsampling_test/sdf_magnitude_filter/5k_D7_W256_L128/results/case3_sdf_filter_+_3surf_from_2/thresh_o0005")

    # res_001 = summarize_results(case3_001_dir)
    # res_002 = summarize_results(case3_002_dir)
    # res_0005 = summarize_results(case3_0005_dir)

    # pprint(res_001)
    # pprint(res_002)
    # pprint(res_0005)

    dir_1_eik_on = Path("/home/rizzardi/Schreibtisch/sdf_subsampling_test/sdf_magnitude_filter/5k_D7_W256_L128/results/case1_3surf_from2/eikonal_on/")

    res_1_eik_on = summarize_results(dir_1_eik_on)

    pprint(res_1_eik_on)


    dir_orig_eik_on = Path("/home/rizzardi/Schreibtisch/eikonal_study/5k_D7_W256_L128/results/metrics/architecture_exploration_eik_ON/")
    res_orig_eik_on = summarize_results(dir_orig_eik_on)

    pprint(res_orig_eik_on)