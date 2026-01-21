from pathlib import Path
import json
import random

from config import (
    PATIENTS_NPY_DATA_DIR,
    TRAIN_DATA_DIR,
    TEST_DATA_DIR
)

def create_train_test_split(
        split = 0.8,
        save_train_filename = "data_fnames_train.json",
        save_test_filename = "data_fnames_test.json",
        num_train = None
    ):
    """
        Splits split% the files found in the directory into train, the rest in test.
        Creates json file with npy files paths.
    """

    patient_names = [file.name for file in PATIENTS_NPY_DATA_DIR.iterdir()]

    if any( not pf.endswith(".npy")  for pf in patient_names):
        raise ImportError("Some file in patient files directory are not .npy but they should all be.")

    num_train = int( split * len(patient_names) )

    random.seed(42)

    train_names = random.sample(patient_names, num_train)
    test_names = [n for n in patient_names if n not in train_names]

    data = []
    for file in train_names:
        pf = PATIENTS_NPY_DATA_DIR / file
        if Path(pf).stat().st_size > 0:  
            data.append(str(file))

    with open(TRAIN_DATA_DIR / save_train_filename, "w") as f:
        json.dump(data, f)

    data = []
    for file in test_names:
        pf = PATIENTS_NPY_DATA_DIR / file
        if Path(pf).stat().st_size > 0:  
            data.append(str(file))

    with open(TEST_DATA_DIR / save_test_filename, "w") as f:
        json.dump(data, f)

    return


if __name__ == "__main__":

    pass

    create_train_test_split(
        save_train_filename= "data_fnames_train.json",
        save_test_filename="data_fnames_test.json"
    )

    # patient_names = ["LEU_NORM_0032"]

    # patient_files = [file.name for file in PATIENTS_NPY_DATA_DIR.iterdir() if any( name in file.name for name in patient_names)]
    
    # random.seed(42)

    # # keep = 20
    # # patient_files = random.sample(patient_files, len(keep))

    # data = []
    # # j = 0
    # for file in patient_files:
    #     # if file not in ["LEU_NORM_0032, AF001, AF069"]:
    #         fname = PATIENTS_NPY_DATA_DIR.name + "/" + file
    #         # build names relative to below DATA_DIR directory only so I can export them elsewhere, and not have absolute paths
    #         # then build the absolute path with DATA_DIR / fname
    #         pf = DATA_DIR / fname
    #         if Path(pf).stat().st_size > 0:  
    #             data.append(str(fname))
    #             # j += 1
    #     # if j == keep:
    #     #     break
    
    # # save_fname = f"data_fnames_{len(data)}_patients.json"
    # print(len(data))

    # save_fname = f"data_fnames_all_patients.json"
    # with open(TEST_DATA_DIR / save_fname, "w") as f:
    #     json.dump(data, f)
       