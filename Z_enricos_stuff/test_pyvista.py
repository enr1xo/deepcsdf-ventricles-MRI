import pyvista as pv
from pyvistaqt import BackgroundPlotter

mesh = pv.Sphere()

plotter = BackgroundPlotter()
plotter.add_mesh(mesh, color="red")

input("Press Enter to close...")