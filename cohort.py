# in questo file vogliamo generare un csv riassiuntivo di tutti i pazienti che includiamo nel dataset per la rete.
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
import re
import pandas as pd

def _resolve_data_file(path):
    """
    Se path è file -> ritorna quello.
    Se path è directory -> cerca dentro csv/ods/xlsx.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Path non trovato: {p}")

    if p.is_file():
        return p

    # Cerca file dati dentro la cartella
    extensions = (".csv", ".ods", ".xlsx", ".xls")
    candidates = []
    for ext in extensions:
        candidates += list(p.rglob(f"*{ext}"))

    if len(candidates) == 0:
        raise FileNotFoundError(f"Nessun file dati trovato dentro: {p}")

    return sorted(candidates)[0]  # prende il primo in ordine alfabetico


def _read_table_auto(file_path):
    """
    Legge automaticamente csv / ods / xlsx
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(file_path)

    elif ext == ".ods":
        return pd.read_excel(file_path, engine="odf")

    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)

    else:
        raise ValueError(f"Formato non supportato: {ext}")


def _norm_patient_key_for_match(pid: str, processed_patient_path: str) -> str:
    s = str(pid).strip()
    up_path = str(processed_patient_path).upper()

    if "LEU_BBB" in up_path:
        # "LEU_BBB_21001" -> "21001"
        m = re.search(r"(\d+)", s)
        return m.group(1) if m else s

    if "YRM" in up_path:
        # "yrm0342_v1" / "yrm0619_v1_2" -> "YRM0342"
        m = re.search(r"(YRM\d+)", s, flags=re.IGNORECASE)
        return m.group(1).upper() if m else s.upper()

    # default
    return s.upper()

def _norm_original_id(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    # "21001.0" -> "21001"
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s.upper()

def _parse_gender_value(x):
    """
    Ritorna 'M' o 'F' o None.
    Gestisce: 0/1, '0;;;', '1;;;', 'M', 'F', 'male', 'female'.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    # LEU_BBB: "1;;;" / "0;;;" -> prendi primo 0/1
    m01 = re.search(r"\b([01])\b", s)
    if m01:
        return "F" if m01.group(1) == "1" else "M"

    up = s.upper()
    if up in {"F", "FEMALE", "DONNA", "WOMAN"}:
        return "F"
    if up in {"M", "MALE", "UOMO", "MAN"}:
        return "M"

    return None

def norm_processed_id(pid, processed_path):
    s = str(pid).strip()
    up_path = str(processed_path).upper()

    if "LEU_BBB" in up_path:
        m = re.search(r"(\d+)", s)
        return m.group(1) if m else s

    if "YRM" in up_path:
        m = re.search(r"(YRM\d+)", s, flags=re.IGNORECASE)
        return m.group(1).upper() if m else s.upper()

    return s.upper()
    
def extract_name():
    return None

def extract_cohort(processed_patients_path):
    """
    takes as path the path pointing to the .xlsx file of the included/excluded patients of the cohort 
    """

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
    """
    anche qua puntiamo agli excel con i dati originali e quelli dei processati
    """

    def _is_norm_path(p: str):
        return "NORM" in str(p).upper()
    
    original_df = _read_table_auto(original_data_path)
    processed_df = pd.read_excel(processed_patient_path, sheet_name="Included")

    col_processed = "Included Patients"


    # ---------- Caso NORM (gender dal nome) ----------
    if "NORM" in str(original_data_path).upper() and \
       "NORM" in str(processed_patient_path).upper():

        included = (
            processed_df[col_processed]
            .dropna()
            .astype(str)
            .str.strip()
        )

        return {
            pid: ("F" if "F" in pid.upper() else "M")
            for pid in included
        }


    # ==========================================================
    # TROVA COLONNE ID E GENDER
    # ==========================================================

    colmap = {str(c).strip().lower(): c for c in original_df.columns}

    possible_id_cols = ["mug-id", "era id", "id", "folder_name"]
    possible_gender_cols = ["sex", "female", "gender"]

    id_col = next((colmap[c] for c in possible_id_cols if c in colmap), None)
    if id_col is None:
        raise ValueError(f"Nessuna colonna ID trovata in {original_data_path}")

    gender_col = next((colmap[c] for c in possible_gender_cols if c in colmap), None)

    # fallback: colonna che contiene 'female'
    if gender_col is None:
        for c in original_df.columns:
            if "female" in str(c).lower():
                gender_col = c
                break

    if gender_col is None:
        raise ValueError(
            f"Nessuna colonna gender trovata in {original_data_path}. "
            f"Colonne disponibili: {list(original_df.columns)}"
        )


    # ==========================================================
    # MERGE ROBUSTO
    # ==========================================================

    included = (
        processed_df[col_processed]
        .dropna()
        .astype(str)
        .str.strip()
    )

    orig = original_df[[id_col, gender_col]].copy()
    orig[id_col] = orig[id_col].apply(_norm_original_id)
    orig[gender_col] = orig[gender_col].apply(_parse_gender_value)

    inc_df = pd.DataFrame({"Included_ID": included})
    inc_df["match_id"] = inc_df["Included_ID"].apply(
        lambda v: norm_processed_id(v, processed_patient_path)
    )

    orig["match_id"] = orig[id_col].apply(
        lambda v: norm_processed_id(v, processed_patient_path)
    )

    merged = inc_df.merge(
        orig[["match_id", gender_col]],
        on="match_id",
        how="left"
    )

    gender_dict = dict(zip(merged["Included_ID"], merged[gender_col]))

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


def write_csv(cohort_pairs, output_path):
    """
    cohort_pairs: list of tuples (original_patients_path, processed_patients_path)
      es: [(orig_AF, proc_AF), (orig_yrm, proc_yrm), ...]
    output_path: file o cartella (se cartella -> patients_features.csv)
    """

    complete = {}  # patient_id -> {"Patient_ID":..., "Cohort":..., "Gender":...}

    for original_path, processed_path in cohort_pairs:
        cohort_dict = extract_cohort(processed_path)
        gender_dict = extract_gender(original_path, processed_path)

        # ===== DEBUG =====
        print("\n---- DEBUG COHORT:", processed_path)
        print("cohort keys sample:", list(cohort_dict.keys())[:5])
        print("gender keys sample:", list(gender_dict.keys())[:5])

        missing = set(cohort_dict) - set(gender_dict)
        print("missing genders:", len(missing), "example:", list(missing)[:10])
        # =================

        patients = set(cohort_dict) | set(gender_dict)

        for p in patients:
            cohort_val = cohort_dict.get(p)
            gender_val = gender_dict.get(p)

            if p not in complete:
                complete[p] = {
                    "Patient_ID": p,
                    "Cohort": cohort_val,
                    "Gender": gender_val
                }
            else:
                # Non deve mai succedere: paziente già visto = in un'altra coorte
                prev = complete[p]["Cohort"]
                new = cohort_val
                raise ValueError(
                    f"Paziente '{p}' appare in più coorti. "
                    f"Cohort precedente: {prev}, nuova: {new}. "
                    f"Controlla i file processed/original."
                )

    df = pd.DataFrame(list(complete.values())).sort_values("Patient_ID")

    output_path = Path(output_path)
    if output_path.is_dir() or output_path.suffix == "":
        output_path = output_path / "patients_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"CSV scritto in: {output_path}")
    return df

ods_file_for_LEU_NORM = False
if ods_file_for_LEU_NORM:

    def generate_gender_ods(input_folder_path, output_file_path):
        """
        Cerca le sottocartelle dentro input_folder_path
        e genera un file .ods con:
            Folder_Name | Gender
        """

        input_path = Path(input_folder_path)

        if not input_path.exists():
            raise ValueError(f"La cartella {input_folder_path} non esiste")

        if not input_path.is_dir():
            raise ValueError(f"{input_folder_path} non è una cartella")

        rows = []

        for subfolder in sorted(input_path.iterdir()):
            if subfolder.is_dir():
                folder_name = subfolder.name
                gender = "F" if "F" in folder_name.upper() else "M"

                rows.append({
                    "Folder_Name": folder_name,
                    "Gender": gender
                })

        df = pd.DataFrame(rows)

        output_file_path = Path(output_file_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Salvataggio in formato ODS
        df.to_excel(output_file_path, engine="odf", index=False)

        print(f"File ODS creato in: {output_file_path}")
    
    L_norm_path = "/home/rizzardi/Schreibtisch/SDF_original_patients/leu_normal_cases/leu_normal_cases"
    output_path = "/home/rizzardi/Schreibtisch/original_patitents_data/patient_metadata_leuNORM.ods"

    # generate_gender_ods(L_norm_path, output_path)

csv_to_ods_LEU_BBB = False
if csv_to_ods_LEU_BBB:
    def csv_to_ods(csv_path, ods_path=None):
        csv_path = Path(csv_path)
        if ods_path is None:
            ods_path = csv_path.with_suffix(".ods")
        else:
            ods_path = Path(ods_path)

        # leggi csv con separatore intelligente
        df_comma = pd.read_csv(csv_path)
        df_semi = pd.read_csv(csv_path, sep=";")
        df = df_semi if df_semi.shape[1] > df_comma.shape[1] else df_comma

        ods_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(ods_path, engine="odf", index=False)

        print(f"Convertito: {csv_path} -> {ods_path}")
        print("Colonne:", df.columns.tolist())
        return ods_path


    csv_to_ods(
    "/home/rizzardi/Schreibtisch/original_patitents_data/patient_metadata_leuBB.csv",
    "/home/rizzardi/Schreibtisch/original_patitents_data/patient_metadata_leuBB.ods"
)

test = False
if test:
    original_path = Path("/mnt/c/Users/e.rizzardi/OneDrive/Desktop/biv_deepsdf/originial_cohort_data/AF_cases.ods")

    # "C:\Users\e.rizzardi\OneDrive\Desktop\biv_deepsdf\biv_deepsdf\AF_cohort_report.xlsx"
    # "C:\Users\e.rizzardi\OneDrive\Desktop\biv_deepsdf\originial_cohort_data\AF_cases.ods"
    processed_patients_path = Path("/mnt/c/Users/e.rizzardi/OneDrive/Desktop/biv_deepsdf/biv_deepsdf/yrm_cohort_report.xlsx")

    res = extract_gender(original_path, processed_patients_path)
    # print(res)

    cohort = extract_cohort(processed_patients_path)
    print(cohort)
    print(cohort.items())

new_test = True
if new_test:
    # cohort_list = ["AF_cases", "2017_ilearnHeart", "ERA-CVD", "VT_cases", "patient_metadata_leuBB"]

    # for c in cohort:
    #     original_path = f"/home/rizzardi/Schreibtisch/original_patitents_data/{c}.ods"

    #     processed_path = "/home/rizzardi/Schreibtisch/processed_patients_cohort_report/AF_cohort_report.xlsx"

    #     output_path = "/home/rizzardi/Schreibtisch/patients_features.csv"

    #     write_csv(original_path, processed_path, output_path)
    

    orig_path_list = ["/home/rizzardi/Schreibtisch/original_patitents_data/AF_cases.ods",
                      "/home/rizzardi/Schreibtisch/original_patitents_data/2017_ilearnHeart.ods",
                      "/home/rizzardi/Schreibtisch/original_patitents_data/ERA-CVD.ods",
                      "/home/rizzardi/Schreibtisch/original_patitents_data/VT_cases.ods",
                      "/home/rizzardi/Schreibtisch/original_patitents_data/patient_metadata_leuBB.ods",
                      "/home/rizzardi/Schreibtisch/original_patitents_data/patient_metadata_leuNORM.ods"]
    
    proc_path_list = ["/home/rizzardi/Schreibtisch/processed_patients_cohort_report/AF_cohort_report.xlsx",
                      "/home/rizzardi/Schreibtisch/processed_patients_cohort_report/S_cohort_report.xlsx",
                      "/home/rizzardi/Schreibtisch/processed_patients_cohort_report/yrm_cohort_report.xlsx",
                      "/home/rizzardi/Schreibtisch/processed_patients_cohort_report/VT_cohort_report.xlsx",
                      "/home/rizzardi/Schreibtisch/processed_patients_cohort_report/LEU_BBB_cohort_report.xlsx",
                      "/home/rizzardi/Schreibtisch/processed_patients_cohort_report/LEU_NORM_cohort_report.xlsx"]
    
    if len(orig_path_list) != len(proc_path_list):
        raise ValueError("le liste dei dati originali e dei processati hanno lunghezze diverse.")
    
    pairs = list(zip(orig_path_list, proc_path_list))

    write_csv(pairs, "/home/rizzardi/Schreibtisch")

    # for i in orig_path_list:
    #     if i.endswith("ods"):
    #         df = pd.read_excel(i, engine="odf")
    #         print(i)
    #         print(df.columns)
    #         print("\n")

    

    