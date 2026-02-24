# controlliamo se effettivamente il vtk che abbiamo a disposizione sia voluemtrico

import pyvista as pv
from pathlib import Path

p = Path(r"C:\Users\e.rizzardi\OneDrive\Desktop\SDF_patients\AF_patients\AF001\vol_gen.vtk")
m = pv.read(p)

print(type(m))
print("n_cells:", m.n_cells, "n_point:", m.n_points)
print("cell arrays:", list(m.cell_data.keys()))
print("point arrays:", list(m.point_data.keys()))