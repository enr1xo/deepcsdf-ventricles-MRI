from pathlib import Path
from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[0]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# PATIENT_MESHES_DIR = Path("/home/navarri/AtriaProject/DATASETS/AtrialGeometries")

PATIENT_MESHES_DIR = Path("/home/davidenava_linux/DATASETS/AtrialGeometries")

ATRIA_TAGS_METADATA = {

    "CUSTOM_LABELS": {
        "99": "RAA_BB", "97": "RA_FO", "96": "RA_INSU_BB", "95": "RA_BB",
        "94": "PM", "93": "CT_endo", "92": "CS_endo", "91": "RA_sept_endo",
        "90": "TV_endo", "89": "INT_cav_bundle_endo", "88": "CT_epi", "87": "CS_epi",
        "86": "SA", "85": "RA_sept_epi", "84": "TV_epi", "83": "INT_cav_bundle_epi",
        "82": "LA_FO", "81": "LA_INSU_BB", "80": "LAA_BB", "79": "RIPV_endo",
        "78": "RSPV_endo", "77": "LIPV_endo", "76": "LSPV_endo", "75": "MV_endo",
        "74": "RIPV_epi", "73": "RSPV_epi", "72": "LIPV_epi", "71": "LSPV_epi",
        "70": "MV_epi", "48": "RAA_epi", "47": "RAA_endo", "46": "IVC_epi",
        "45": "IVC_endo", "44": "SVC_epi", "43": "SVC_endo", "42": "RA_wall_epi",
        "41": "RA_wall_endo", "38": "LAA_epi", "37": "LAA_endo", "32": "LA_wall_epi",
        "31": "LA_wall_endo"
    },

    "RA_TAGS": [99, 97, 95, 96, 94, 93, 92, 91, 90, 88, 87, 86, 85, 84, 48, 47, 46, 45, 44, 43, 42, 41],

    "RA_ENDO_TAGS": [41, 43, 45, 47, 86, 90, 91, 92, 93, 94, 97],

    "RA_EPI_TAGS": [42, 44, 46, 48, 84, 85, 86, 87, 88, 95, 99],

    "LA_TAGS": [82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 38, 37, 32, 31],

    "LA_ENDO_TAGS": [31, 37, 75, 76, 77, 78, 79, 82],

    "LA_EPI_TAGS": [32, 38, 70, 71, 72, 73, 74, 80, 82],

    "SHARED_TAGS": [89, 83],

    "vein_names": ["IPV", "SPV", "IVC", "SVC"],
    
    "valve_names": ["MV", "TV"]
}

DATA_DIR = PROJ_ROOT / "data"

# PATIENTS_COORDS_AND_SDFS_DIR = Path("/home/navarri/AtriaProject/DATASETS/AtriaPointsAndSDF")

# PATIENTS_NPY_DATA_DIR =  PATIENTS_COORDS_AND_SDFS_DIR / "single_patients_100000pts_npy"

PATIENTS_COORDS_AND_SDFS_DIR = Path("/home/davidenava_linux/DATASETS/AtriaPointsAndSDFs")

PATIENTS_NPY_DATA_DIR =  PATIENTS_COORDS_AND_SDFS_DIR / "single_patients_50000pts_npy"

SPECS_FILES_DIR = PROJ_ROOT / "specs_files"

EXPERIMENTS_DIR = PROJ_ROOT / "experiments"

TRAIN_DATA_DIR = DATA_DIR / "train"

TEST_DATA_DIR = DATA_DIR / "test"

MODELS_DIR = PROJ_ROOT / "models"

RESULTS_DIR = PROJ_ROOT / "results"

IMAGES_DIR = RESULTS_DIR / "images"

RECONSTRUCTED_MESHES_DIR = RESULTS_DIR / "reconstructed"

LATENTS_DIR = RESULTS_DIR / "fitted_latents"