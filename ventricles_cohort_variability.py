import os
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

old = False
if old:
    n = 4

    diseases_list = ["AF", "yrm", "VT", "LEU_BBB", "LEU_NORM"]
    disease = diseases_list[n]

    cohort_list = ["AF_processed", "SV_processed", "VT_cases_processed", "LEU_BBB_processed", "LEU_NORM_processed"]
    cohort = cohort_list[n]

    print("processing desease:", cohort)

    # root_AF = Path("/home/rizzardi/Schreibtisch/SDF_processed_patients/AF_processed/AF_processed")
    # root_SV = Path("/home/rizzardi/Schreibtisch/SDF_processed_patients/sicvalves_processed/sicvalves_processed")
    # root_VT = Path("/home/rizzardi/Schreibtisch/SDF_processed_patients/VT_cases_processed/VT_cases_processed")
    # root_Leu_BBB = Path("/home/rizzardi/Schreibtisch/SDF_processed_patients/leu_BBB_cases_processed/leu_BBB_cases_processed")
    root_Leu_NORM = Path("/home/rizzardi/Schreibtisch/SDF_processed_patients/leu_normal_cases_processed/leu_normal_cases_processed")

    root = root_Leu_NORM
    results = {}

        
    for patient_dir in sorted(root.iterdir()):
        # filtriamo quelli con AF o SV (yrm)
        if patient_dir.is_dir() and patient_dir.name.startswith(disease):

            epi_path = patient_dir / "epicardium-processed.vtp"
            lv_path  = patient_dir / "lv_endo-processed.vtp"
            rv_path  = patient_dir / "rv_endo-processed.vtp"

            if not (epi_path.exists() and lv_path.exists() and rv_path.exists()):
                print(f"Skipping {patient_dir.name} (missing files)")
                continue

            epi = pv.read(str(epi_path)).triangulate()
            lv  = pv.read(str(lv_path)).triangulate()
            rv  = pv.read(str(rv_path)).triangulate()

            scale_to_original = epi.field_data["scale-tooriginalrange"]
            scale_lv = lv.field_data["scale-tooriginalrange"]
            scale_rv = rv.field_data["scale-tooriginalrange"]

            # volumes
            s_3 = scale_to_original**3

            V_epi = epi.volume * s_3
            V_LV  = lv.volume * s_3
            V_RV  = rv.volume * s_3

            V_myo = V_epi - V_LV - V_RV

            # areas
            s_2 = scale_to_original ** 2

            A_epi = epi.area * s_2
            A_lv = lv.area * s_2
            A_rv = rv.area * s_2

            A_endo = A_lv + A_rv
            A_tot = A_epi + A_endo

            myo_avg_thickness = V_myo / (A_tot / 2)

            results[patient_dir.name] = {
                "V_myo": V_myo,
                "V_LV": V_LV,
                "A_LV": A_lv,
                "V_RV": V_RV,
                "A_RV": A_rv,
                "myo_avg_thickness": myo_avg_thickness,
                "scale": scale_to_original
            }
        # break
    # stampa risultato
    # for k, v in results.items():
    #     print(k, v)

    # print("\nNumero pazienti trovati:", len(results))
    # af001_scale = results["AF001"]["scale"]

    # for patient_dir in sorted(root.iterdir()):
    #     if patient_dir.is_dir() and patient_dir.name.startswith("AF"):

    #         af001_epi_path = patient_dir / "epicardium-processed.vtp"
    #         af001_lv_path  = patient_dir / "lv_endo-processed.vtp"
    #         af001_rv_path  = patient_dir / "rv_endo-processed.vtp"
    #     break

    # af001_epi = pv.read(str(af001_epi_path)).triangulate()
    # af001_lv = pv.read(str(af001_lv_path)).triangulate()
    # af001_rv = pv.read(str(af001_rv_path)).triangulate()

    # print("pre scaling AF001 - epi")
    # print("min: ", af001_epi.points.min(), "max: ", af001_epi.points.max())
    # print("range coordinate: ", af001_epi.bounds)

    # af001_epi.points *= af001_scale

    # print("post scaling AF001 - epi")
    # print("min: ", af001_epi.points.min(), "max: ", af001_epi.points.max())
    # print("range coordinate: ", af001_epi.bounds)


    patients = sorted(results.keys())
    # scaled volumes in mm**3
    V_LV  = np.array([results[p]["V_LV"]  for p in patients], dtype=float)
    A_LV = np.array([results[p]["A_LV"]  for p in patients], dtype=float)

    V_RV  = np.array([results[p]["V_RV"]  for p in patients], dtype=float)
    A_RV = np.array([results[p]["A_RV"]  for p in patients], dtype=float)

    V_MYO = np.array([results[p]["V_myo"] for p in patients], dtype=float)

    avg_thick = np.array([results[p]["myo_avg_thickness"] for p in patients], dtype=float)

    # print(V_LV)
    # print("LV areas pre:", A_LV[0])
    # # A_LV[0] = A_LV[0] / 1e6
    # print("LV areas post:", A_LV[0])

    # print("max_VLV: ", V_LV.max())
    # print("min_VLV: ", V_LV.min())
    print("LV")
    idx_min_vlv = np.argmin(V_LV)
    idx_max_vlv = np.argmax(V_LV)

    print("min lv area patient", patients[idx_min_vlv])
    print("max lv area patient", patients[idx_max_vlv])

    print("\n")

    idx_min_alv = np.argmin(A_LV)
    idx_max_alv = np.argmax(A_LV)

    print("min lv area patient", patients[idx_min_alv])
    print("max lv area patient", patients[idx_max_alv])

    print("\n")

    print("RV")
    idx_min_vrv = np.argmin(V_RV)
    idx_max_vrv = np.argmax(V_RV)

    print("min lv area patient", patients[idx_min_vlv])
    print("max lv area patient", patients[idx_max_vlv])
    
    print("\n")

    idx_min_arv = np.argmin(A_RV)
    idx_max_arv = np.argmax(A_RV)

    print("min lv area patient", patients[idx_min_alv])
    print("max lv area patient", patients[idx_max_alv])

    V_LV = V_LV / 1e12
    V_RV = V_RV / 1e12
    V_MYO = V_MYO / 1e12

    A_LV = A_LV / 1e6
    A_RV = A_RV / 1e6 

    # print("post conversion in ml")
    # print("max_VLV: ", V_LV.max())
    # print("min_VLV: ", V_LV.min())

    avg_thick_mm = avg_thick * 1e-3
    ## PLOTTING STUFF

plotting_old = False
if plotting_old:
    # LV
    plt.figure()
    plt.hist(V_LV, bins="auto", rwidth=0.9)
    plt.title("Distribuzione volume LV")
    plt.xlabel("Volume LV (ml)")
    plt.ylabel("quantity")
    plt.show()

    # print("LV done")

    # RV
    plt.figure()
    plt.hist(V_RV, bins="auto", rwidth=0.9)
    plt.title("Distribuzione volume RV")
    plt.xlabel("Volume RV (ml)")
    plt.ylabel("quantity")
    plt.show()

    # # MYO
    # plt.figure()
    # plt.hist(V_MYO, bins="auto")
    # plt.title("Distribuzione volume MYO")
    # plt.xlabel("Volume Myo (ml")
    # plt.ylabel("quantity")
    # plt.show()

    # Thickness
    plt.figure()
    plt.hist(avg_thick_mm, bins="auto", rwidth=0.9)
    plt.title("Distribuzione spessore medio myocardio")
    plt.xlabel("Spessore medio (mm)")
    plt.ylabel("Quantity")
    plt.show()


plotting_new = False
if plotting_new:
    # save_dir =f"/mnt/c/Users/e.rizzardi/OneDrive/Desktop/biv_deepsdf/biv_deepsdf/cohort_distribution"
    
    # if disease == "yrm":
    #     save_dir = os.path.join(save_dir, "Sick_valves")
    # elif disease == "AF":
    #     save_dir = os.path.join(save_dir, "AF")
    
    # os.makedirs(save_dir, exist_ok=True)

    # print("saving plots in:", save_dir)

    # creatin bins
    x = np.asarray(V_LV, dtype=float).ravel()

    bin_width = 10  # ampiezza intervallo in ml

    min_edge = bin_width * np.floor(x.min() / bin_width)
    max_edge = bin_width * np.ceil(x.max() / bin_width)

    print("V_LV min/max:", x.min(), x.max())

    bins = np.arange(min_edge, max_edge + bin_width, bin_width)

    # LV
    # LV volume
    x = np.asarray(V_LV, dtype=float).ravel()
    x = x[np.isfinite(x)]

    plt.figure()
    plt.hist(x, bins=bins, edgecolor='black', rwidth=0.9)

    plt.xlabel("Volume LV (ml)")
    plt.ylabel("Numero pazienti")
    plt.title(f"{disease} - Distribuzione volumi LV (bin = {bin_width})")

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "LV_volume_distribution.png"), dpi=300)
    # plt.close()
    plt.show()

    # LV area
    x = np.asarray(A_LV, dtype=float).ravel()
    x = x[np.isfinite(x)]
    print("A_LV min/max:", x.min(), x.max())

    bin_width_area = 30  

    min_edge = bin_width_area * np.floor(x.min() / bin_width_area)
    max_edge = bin_width_area * np.ceil(x.max() / bin_width_area)

    bins_area = np.arange(min_edge, max_edge + bin_width_area, bin_width_area)


    plt.figure()
    plt.hist(x, bins=bins_area, edgecolor='black', rwidth=0.9)

    plt.xlabel("Area LV (mm^2)")
    plt.ylabel("Numero pazienti")
    plt.title(f"{disease} - Distribuzione area LV (bin = {bin_width_area})")

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "LV_area_distribution.png"), dpi=300)
    # plt.close()    
    plt.show()

    # RV
    # RV Volume
    x = np.asarray(V_RV, dtype=float).ravel()
    x = x[np.isfinite(x)]

    min_edge = bin_width * np.floor(x.min() / bin_width)
    max_edge = bin_width * np.ceil(x.max() / bin_width)

    print("V_RV min/max raw:", x.min(), x.max())

    bins = np.arange(min_edge, max_edge + bin_width, bin_width)

    plt.figure()
    plt.hist(x, bins=bins, edgecolor='black', rwidth=0.9)

    plt.xlabel("Volume RV (ml)")
    plt.ylabel("Numero pazienti")
    plt.title(f"{disease} - Distribuzione volumi RV (bin = {bin_width})")

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "RV_volume_distribution.png"), dpi=300)
    # plt.close()  
    plt.show()

    # RV Area
    x = np.asarray(A_RV, dtype=float).ravel()
    x = x[np.isfinite(x)]
    
    print("A_RV min/max:", x.min(), x.max())

    bin_width_area = 30  

    min_edge = bin_width_area * np.floor(x.min() / bin_width_area)
    # print("min edge rv: ", min_edge)
    max_edge = bin_width_area * np.ceil(x.max() / bin_width_area)
    # print("max edge rv: ", max_edge)

    bins_area = np.arange(min_edge, max_edge + bin_width_area, bin_width_area)

    plt.figure()
    plt.hist(x, bins=bins_area, edgecolor='black', rwidth=0.9)

    plt.xlabel("Area RV (mm^2)")
    plt.ylabel("Numero pazienti")
    plt.title(f"{disease} - Distribuzione aree RV (bin = {bin_width})")

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "RV_area_distribution.png"), dpi=300)
    # plt.close()  
    plt.show()

    # AVG thickness
    # x = np.asarray(avg_thick_mm, dtype=float).ravel()
    # x = x[np.isfinite(x)]

    # bin_width = 0.25  # ampiezza intervallo in mm

    # # creiamo i bin multipli di 0.5
    # min_edge = bin_width * np.floor(x.min() / bin_width)
    # max_edge = bin_width * np.ceil(x.max() / bin_width)

    # bins = np.arange(min_edge, max_edge + bin_width, bin_width)

    # plt.figure()
    # plt.hist(x, bins=bins, edgecolor='black', rwidth=0.9)

    # plt.xlabel("Average thickness (mm)")
    # plt.ylabel("Numero pazienti")
    # plt.title(f"Distribuzione average thickness (bin = {bin_width} mm)")

    # plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.show()



# dove stanno i dati: desktop/nome_file.vsc
# leggiamo il csv
# dividiamo sulla colonna cohort
# contiamo quanti tipi di genere abbiamo per ogni cohort
# plottiamo uno scatter per ogni cohort

path = Path("/home/rizzardi/Schreibtisch/patients_features.csv")

df = pd.read_csv(path)

summary = (df.groupby(["Cohort", "Gender"]).size().unstack(fill_value=0))

genders_counts = df["Gender"].value_counts()
n_male = genders_counts["M"]
n_female = genders_counts["F"]

df_missing = df[df["Gender"].isna()]
n_miss = len(df_missing["Gender"])

summary.plot(kind="bar")
plt.ylabel("Numero pazienti")
plt.title(f"Gender distribution\n tot_m {n_male}, tot_f {n_female}, missing {n_miss}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

