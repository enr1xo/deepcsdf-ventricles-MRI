"""
For each cohort, it creates an excel file in which patients are distinguished 
depending on their inclusion in the potential training dataset.

deseases:
    - AF : atrial fibrillation cohort
    - yrm : sick valve cohort
"""

from pathlib import Path
import pandas as pd


desease_list = ["AF", "yrm", "S", "VT"] 

for disease in desease_list:

    print(f"\n Creatin execel file for {disease}.\n")
    
    original_patients_path = Path(f"/mnt/c/Users/e.rizzardi/OneDrive/Desktop/SDF_patients/{disease}_patients")

    if disease == "yrm":
        original_patients_path = Path(f"/mnt/c/Users/e.rizzardi/OneDrive/Desktop/SDF_patients/SickValve_patients")
    
    elif disease == "S":
        original_patients_path = Path(f"/mnt/c/Users/e.rizzardi/OneDrive/Desktop/SDF_patients/2017_ilearnHeart")

    elif disease == "VT":
        original_patients_path = Path(f"/mnt/c/Users/e.rizzardi/OneDrive/Desktop/SDF_patients/{disease}_cases")

    processed_patients_path = Path("/mnt/c/Users/e.rizzardi/OneDrive/Desktop/biv_deepsdf/biv_deepsdf/AF-and-sicvalves")

    output_base_path = Path("/mnt/c/Users/e.rizzardi/OneDrive/Desktop/biv_deepsdf/biv_deepsdf")


    # the original cohort, andiamo a cercare in original_aptients_apth tutti quelli che iniziano con desease
    original_coort = {
        p.name for p in original_patients_path.iterdir() 
        if p.is_dir() and p.name.startswith(disease)
    }
    
    # andiamo a cercare in processed patients tutti quelli che hanno nome "disease"
    
    processed_coort = {
        p.name for p in processed_patients_path.iterdir()
        if p.is_dir() and p.name.startswith(disease)
    }

    # print(original_coort)
    # print(processed_coort)
    # break

    excluded = original_coort - processed_coort
    included = processed_coort & original_coort
    
    print("len included:", len(included))
    print("len excluded:", len(excluded))
    print("len original_coort:", len(original_coort))

    sanity_check = processed_coort | excluded
    print("sanity_cehck:", len(sanity_check))

    # break
    if sanity_check != original_coort:
        raise ValueError("there is no match between the original coort and the sum between excluded and included!")

    print(f"Original cohort: {len(original_coort)}")
    print(f"Included: {len(included)}")
    print(f"Exncluded: {len(excluded)}")

    # creating excel
    df_included = pd.DataFrame({"Included Patients": sorted(included)})
    df_excluded = pd.DataFrame({"Excluded Patients": sorted(excluded)})

    output_file = output_base_path / f"{disease}_cohort_report.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_included.to_excel(writer, sheet_name="Included", index=False)
        df_excluded.to_excel(writer, sheet_name="Excluded", index=False)
    
    print(f"Excel file saved in: {output_file}")