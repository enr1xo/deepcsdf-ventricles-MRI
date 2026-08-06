import json
from pathlib import Path


TRAIN_JSON = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/MRI_model/3axis/train/data_fnames_train.json"
)

TEST_JSON = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/MRI_model/3axis/test/data_fnames_test.json"
)


def update_filename(name: str) -> str:
    """
    Modifica qui la regola dal vecchio nome al nuovo nome.
    """

    return name.replace(
        "_mri_samples.npy",
        "_three_axis_mri_samples.npy",
    )


def update_json_file(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = [update_filename(name) for name in data]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=4)

    print(f"Updated: {json_path}")
    print("First entries:")
    for name in updated[:10]:
        print("  ", name)


update_json_file(TRAIN_JSON)
update_json_file(TEST_JSON)