import numpy as np
import pyvista as pv

NPY_PATH = "/mnt/c/Users/e.rizzardi/OneDrive/Desktop/AF_patients/single_patients_100000pts_npy/AF008_P1-epi_lv_rv_100000_coords_and_sdf.npy"
OUT_HTML = "/mnt/c/Users/e.rizzardi/OneDrive/Desktop/sdf_view_AF008_P1_epi.html"

SDF_COL = 3          # 3=epi, 4=lv, 5=rv
NEAR_EPS = 2      # mostra anche una “shell” vicino alla superficie

data = np.load(NPY_PATH)
coords = data[:, :3]
sdf = data[:, SDF_COL]

cloud = pv.PolyData(coords)
cloud["sdf"] = sdf

# shell vicino alla superficie (più leggibile)
mask = np.abs(sdf) < NEAR_EPS
near = pv.PolyData(coords[mask])
near["sdf"] = sdf[mask]

p = pv.Plotter()
p.add_text(f"SDF col={SDF_COL} | near |sdf|<{NEAR_EPS} (N={mask.sum()})", font_size=10)

# punti tutti: trasparenti
p.add_mesh(cloud, scalars="sdf", render_points_as_spheres=True, point_size=3, opacity=0.15)

# near-surface: evidenziati
p.add_mesh(near, scalars="sdf", render_points_as_spheres=True, point_size=6, opacity=1.0)

p.show_axes()
p.export_html(OUT_HTML)
print("HTML salvato in:", OUT_HTML)


    

