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

import numpy as np

data = np.load("/home/rizzardi/Schreibtisch/MRI_model/generated_npy/AF001_mri_samples.npy")

print(data.shape)
print(data.dtype)
print(data[998:1002])
