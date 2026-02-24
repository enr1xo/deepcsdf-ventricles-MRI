import pandas as pd

# ————————————————
# Carica i due file Excel
# ————————————————
original_patients_data = "/mnt/c/Users/e.rizzardi/OneDrive/Desktop/biv_deepsdf/originial_cohort_data/AF_cases.ods"
processed_patients = "/mnt/c/Users/e.rizzardi/OneDrive/Desktop/biv_deepsdf/biv_deepsdf/AF_cohort_report.xlsx"

original_df = pd.read_excel(original_patients_data, engine="odf")
processed_df = pd.read_excel(processed_patients, sheet_name="Included")

# -----------------------
# NOMI COLONNE
# -----------------------
col_original = "MUG-ID"
col_sex = "Sex"
col_processed = "Included Patients"

# ————————————————
# Trova i pazienti presenti in entrambi i file
# ————————————————
set_original = set(original_df[col_original].dropna().astype(str))
set_processed = set(processed_df[col_processed].dropna().astype(str))

print(set_original)
print(set_processed)

pazienti_comuni = set_original.intersection(set_processed)

print(f"Pazienti presenti in entrambi i file ({len(pazienti_comuni)})")
print(f"pazienti processati: {len(set_processed)}")

# for p in pazienti_comuni:
#     print(p)

# estrai ID + sex solo per i comuni
df_genere = original_df[original_df[col_original].astype(str).isin(pazienti_comuni)][[col_original, col_sex]].copy()

# normalizza sex
df_genere[col_sex] = df_genere[col_sex].astype(str).str.strip().str.upper()

# maschera: sesso mancante o non valido
mask_missing = (
    df_genere[col_sex].isna()
    | (df_genere[col_sex] == "")
    | (~df_genere[col_sex].isin(["M", "F"]))
)

no_gender_patients = df_genere.loc[mask_missing, col_original]

print("\nGenere distribution:")
print(df_genere[col_sex].value_counts(dropna=False))

if len(no_gender_patients) > 0:
    print("\nThere are patients with unknown gender!")
    print(no_gender_patients.tolist())
else:
    print("\nAll common patients have a known gender (M/F).")

# # ————————————————
# # (Opzionale) Salva in Excel
# # ————————————————
# df_genere.to_excel("pazienti_comuni_con_genere.xlsx", index=False)