# from pathlib import Path
# from MRI_dataset_tools import generate_patient_mri_dataset, MRIDatasetParams

# params = MRIDatasetParams(
#     n_points_per_square=1000,
#     plot_debug=True,
#     save_npy=True,
#     save_csv=False,
# )

# samples, stats, debug = generate_patient_mri_dataset(
#     patient="LEU_NORM_4590",
#     all_processed_dir=Path("/home/rizzardi/Schreibtisch/AF001_aligned_processed"),
#     csv_path=Path("/home/rizzardi/Schreibtisch/MRI_model/mitral_Carea_apex_MaxD.csv"),
#     output_dir=Path("/home/rizzardi/Schreibtisch/MRI_model/generated_npy"),
#     params=params,
# )

# print(stats)

# import numpy as np

# data = np.load("/home/rizzardi/Schreibtisch/MRI_model/generated_npy/AF001_mri_samples.npy")

# print(data.shape)
# print(data.dtype)
# print(data[998:1002])


import json
from pathlib import Path

json_path = Path("/home/rizzardi/Schreibtisch/MRI_model/MRI_model/test/data_fnames_test.json")
json_path = Path("/home/rizzardi/Schreibtisch/MRI_model/MRI_model/train/data_fnames_train.json")

with open(json_path, "r") as f:
    data = json.load(f)

def rename(obj):
    if isinstance(obj, str):
        return obj.replace(
            "_MRI_like_coords_and_sdf.npy",
            "_mri_samples.npy",
        )

    elif isinstance(obj, list):
        return [rename(x) for x in obj]

    elif isinstance(obj, dict):
        return {k: rename(v) for k, v in obj.items()}

    return obj

data = rename(data)

with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

print("Done.")