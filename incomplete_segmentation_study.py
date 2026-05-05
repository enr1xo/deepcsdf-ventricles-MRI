import pyvista as pv
from pathlib import Path
import math

import math
from pathlib import Path
import pyvista as pv

def visualize_epi_in_batches(folder_path, batch_size=24):
    folder = Path(folder_path)
    epi_files = sorted(folder.rglob("*epicardium-processed.vtp"))

    if not epi_files:
        print("Nessun file epicardium-processed trovato.")
        return

    print(f"Trovati {len(epi_files)} file di epi")

    for i in range(0, len(epi_files), batch_size):
        batch = epi_files[i:i + batch_size]
        n = len(batch)

        # griglia quasi quadrata basata su n (non batch_size)
        n_col = math.ceil(math.sqrt(n))
        n_row = math.ceil(n / n_col)

        print(f"\nVisualizzo file {i+1} - {i+n} (grid {n_row}x{n_col})")

        # PyVista: (rows, cols)
        plotter = pv.Plotter(shape=(n_row, n_col))

        for j, vtk_file in enumerate(batch):
            row = j // n_col
            col = j % n_col

            plotter.subplot(row, col)
            mesh = pv.read(vtk_file)

            patient_name = vtk_file.parent.name
            plotter.add_mesh(mesh, color="lightgray")
            plotter.add_text(f"{patient_name}\n{vtk_file.name}", font_size=9)

        plotter.link_views()
        plotter.show()

        input("Premi INVIO per vedere i prossimi...")


# -------- USO --------
# leu_bbb_path = "/home/rizzardi/Schreibtisch/SDF_processed_patients/leu_BBB_cases_processed/leu_BBB_cases_processed"
# leu_normal_path = "/home/rizzardi/Schreibtisch/SDF_processed_patients/leu_normal_cases_processed/leu_normal_cases_processed"
# af_path = "/home/rizzardi/Schreibtisch/SDF_processed_patients/AF_processed/AF_processed"
# sv_path = "/home/rizzardi/Schreibtisch/SDF_processed_patients/sicvalves_processed/sicvalves_processed"
# ilearnHeart_path = "/home/rizzardi/Schreibtisch/SDF_processed_patients/2017_ilearnHeart_prcessed/2017_ilearnHeart_prcessed"
#vt_path = "/home/rizzardi/Schreibtisch/SDF_processed_patients/VT_cases_processed/VT_cases_processed"

# new paths for leuven patients
leuNORM = "/home/rizzardi/Schreibtisch/leuvenNORMcomplete"
leuBBB = "/home/rizzardi/Schreibtisch/leuvenBBBcomplete"


visualize_epi_in_batches(leuBBB)
