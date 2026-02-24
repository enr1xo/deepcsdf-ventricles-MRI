# in questo file vogliamo generar un csv riassiuntivo di tutti i pazienti che includiamo nel dataset per la rete.
# Struttura:
# 1 - name: nome del paziente
# 2 - cohort: coorte di origine
# 3 - gender: male/female/unkown
# 4 - volume_lv: voluem del left ventricle
# 5 - volume_rv: volume del right ventricle
# 6 - mass_lv: massa del left ventircle
# 7 - mass_rv: massa del right ventricle
# 8 - wall_thickness_avg_lv: wall thickness media del left ventricle
# 9 - wall_thickness_avg_rv: wall thickness media del right ventricle
# 10 - wall_thickness_max_lv: wall thickness massima del left ventricle
# 11 - wall_thickness_max_rv: wall thickness massima del right ventricle
# 12 - spt_wall_ratio: septal-to-posterior wall ratio del left ventricle

import pyvista as pv
from pathlib import Path
import pandas as pd

def extract_name():
    return None

def extract_cohort(processed_patients_path):
    result = {}
    cohort_names = {"AF": "AF", 
                    "yrm": "SickValve", 
                    "norm": "Leuven_norm", 
                    "BBB": "Leuven_BBB", 
                    "VT": "VT",
                    "S": "ilearnHeart"
                    }

    df = pd.read_excel(processed_patients_path, sheet_name="Included")

    first_col = df.columns[0]

    patient_names = (df[first_col].dropna().astype(str).str.strip())

    for patient in patient_names:
        assigned = None

        for key, cohort_value in cohort_names.items():
            if key.lower() in patient.lower():
                assigned = cohort_value
                break
        
        result[patient] = assigned

    return result

def extract_gender(original_data_path, processed_patient_path):
    original_df = pd.read_excel(original_data_path, engine="odf")
    processed_df = pd.read_excel(processed_patient_path, sheet_name="Included")

    original_col_name_list = ["MUG-ID", "ERA ID"]
    original_col_gender_list = ["Sex", "sex"]

    col_processed = "Included Patients"

    id_col = None
    for name in original_col_name_list:
        if name in original_df.columns:
            id_col = name
            break
        
    if id_col is None:
        raise ValueError(f"nessuna delle colonne ID trovata tra: {original_col_name_list}")

    gender_col = None
    for name in original_col_gender_list:
        if name in original_df.columns:
            gender_col = name
            break
    
    if gender_col is None:
        raise ValueError(f"nessuna delle colonne gender trovata tra: {original_col_gender_list}")

    set_original = set(
        original_df[id_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    set_processed = set(processed_df[col_processed].dropna().astype(str).str.strip())

    common_patients = set_original.intersection(set_processed)

    df_gender = original_df[original_df[id_col].astype(str).str.strip().isin(common_patients)][[id_col, gender_col]].copy()
    
    df_gender[id_col] = df_gender[id_col].astype(str).str.strip()

    df_gender[gender_col] = df_gender[gender_col].replace({0: "M", 1: "F", "0": "M", "1": "F"})

    df_gender.loc[df_gender[gender_col].isna(), gender_col] = None

    df_gender[gender_col] = df_gender[gender_col].apply(
        lambda x: x.strip().upper() if isinstance(x, str) else x
    )

    df_gender.loc[df_gender[gender_col].isin(["", "NAN", "NONE", "nan", "None"]), gender_col] = None

    gender_dict = dict(zip(df_gender[id_col], df_gender[gender_col]))

    return gender_dict

def compute_chamber_volume(patient_dir, disease=None):

    results = {}

    if patient_dir.is_dir() and patient_dir.name.startswith(disease):

        epi_path = patient_dir / "epicardium-processed.vtp"
        lv_path  = patient_dir / "lv_endo-processed.vtp"
        rv_path  = patient_dir / "rv_endo-processed.vtp"

        if not (epi_path.exists() and lv_path.exists() and rv_path.exists()):
            print(f"Skipping {patient_dir.name} (missing files)")
        
        epi = pv.read(str(epi_path)).triangulate()
        lv  = pv.read(str(lv_path)).triangulate()
        rv  = pv.read(str(rv_path)).triangulate()

        scale_to_original = epi.field_data["scale-tooriginalrange"]
        
        s_3 = scale_to_original ** 3

        v_epi = epi.volume * s_3
        v_lv = lv.volume * s_3
        v_rv = rv.volume * s_3

        v_myo = v_epi - v_lv - v_rv

        v_lv /= 1e12
        v_rv /= 1e12
        v_myo /= 1e12

        results[patient_dir.name] = {   "v_myo": v_myo,
                                        "v_lv": v_lv,
                                        "v_rv": v_rv,
        }
    
    return results

original_path = Path("/mnt/c/Users/e.rizzardi/OneDrive/Desktop/biv_deepsdf/originial_cohort_data/AF_cases.ods")

# "C:\Users\e.rizzardi\OneDrive\Desktop\biv_deepsdf\biv_deepsdf\AF_cohort_report.xlsx"
# "C:\Users\e.rizzardi\OneDrive\Desktop\biv_deepsdf\originial_cohort_data\AF_cases.ods"
processed_patients_path = Path("/mnt/c/Users/e.rizzardi/OneDrive/Desktop/biv_deepsdf/biv_deepsdf/yrm_cohort_report.xlsx")

res = extract_gender(original_path, processed_patients_path)
# print(res)

cohort = extract_cohort(processed_patients_path)
print(cohort)
print(cohort.items())