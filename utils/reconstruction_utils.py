import numpy as np
from skimage import measure
import pyvista as pv
import pyacvd

def isosurface_from_sdf(x, y, z, sdf_pred, level, box_lim = 105):
    
    D = sdf_pred.reshape((len(x), len(y), len(z)))

    D = np.transpose(D, (1, 0, 2))

    # Run marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        D,
        level=level,
        spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0])
    )

    # Adjust vertices
    # my volume is in [-105,105] cube but marching cubes assumes a vertex is in (0,0,0), so I need to traslate it back to my real coordinates
    verts = verts - box_lim  

    # Convert faces for PyVista
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)

    # Create PyVista mesh
    mesh = pv.PolyData(verts, faces_pv)

    return mesh

def isosurface_from_query_sdf(query_points, sdf_values):

    from scipy.interpolate import griddata

    resolution = 128
    # query_points → (N, 3) points where SDF is known
    # sdf_values   → (N,)   known SDF values

    # create grid
    x = np.linspace(-105, 105, resolution)
    y = np.linspace(-105, 105, resolution)
    z = np.linspace(-105, 105, resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # interpolate
    sdf_grid = griddata(
        points=query_points,
        values=sdf_values,
        xi=grid_points,
        method='linear'
    )

    sdf_grid = sdf_grid.reshape((len(x), len(y), len(z)))

    sdf_grid = np.transpose(sdf_grid, (1, 0, 2))

    vertices, faces, normals, values = measure.marching_cubes(sdf_grid, level=0.0, spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0]))

    vertices = vertices - 105

    # Convert faces for PyVista
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)

    # Create PyVista mesh
    mesh = pv.PolyData(vertices, faces_pv)

    return mesh


if __name__ == "__main__":

    from pathlib import Path

    patient_files_dir = Path("/home/davidenava_linux/AtriaProject/deepcsdf_fork/deepcsdf/deepcsdf/deepcsdfatria/data/single_patients_npy")

    patient_meshes_dir = Path("/home/davidenava_linux/DATASETS/AtrialGeometriesData")

    patient_name = "AF069"

    patient_dir = patient_meshes_dir / patient_name

    mesh_file = next( (f for f in patient_dir.iterdir() if "epicardium_processed.vtp" in f.name ))

    mesh = pv.read(mesh_file)

    points_file = next( patient_files_dir.rglob(f"{patient_name}*"), None)

    data = np.load(points_file)

    points = data[:,:3] * 100
    sdf_values = data[:,3]

    # points = pv.PolyData(points)
    # points["sdf"] = sdf_values

    # plotter = pv.Plotter()
    # plotter.add_mesh(points, scalars = "sdf", cmap = "jet_r")
    # plotter.show()
    
    reconstructed = isosurface_from_query_sdf(points, sdf_values)
    reconstructed.plot()

    from scipy.interpolate import Rbf
    import matplotlib.pyplot as plt

    # x, y, z = points[:,0], points[:,1], points[:,2]
    # sdf = sdf_values

    # rbf = Rbf(x, y, z, sdf, function='multiquadric', smooth=0.0)
    x = np.linspace(-105, 105, 64)
    y = np.linspace(-105, 105, 64)
    z = np.linspace(-105, 105, 64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # sdf_grid = rbf(xx, yy, zz)
    # mid = sdf_grid.shape[0] // 2

    # plt.figure(figsize=(15,5))

    # plt.subplot(131)
    # plt.title("X slice")
    # plt.imshow(sdf_grid[mid,:,:], origin='lower')
    # plt.colorbar()

    # plt.subplot(132)
    # plt.title("Y slice")
    # plt.imshow(sdf_grid[:,mid,:], origin='lower')
    # plt.colorbar()

    # plt.subplot(133)
    # plt.title("Z slice")
    # plt.imshow(sdf_grid[:,:,mid], origin='lower')
    # plt.colorbar()

    # plt.show()
    from scipy.interpolate import RBFInterpolator

    rbf = RBFInterpolator(points, sdf_values, kernel='multiquadric', epsilon=50)
    sdf_grid = rbf(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
    sdf_grid = sdf_grid.reshape(xx.shape)
    mid = sdf_grid.shape[0] // 2

    plt.figure(figsize=(15,5))

    plt.subplot(131)
    plt.title("X slice")
    plt.imshow(sdf_grid[mid,:,:], origin='lower')
    plt.colorbar()

    plt.subplot(132)
    plt.title("Y slice")
    plt.imshow(sdf_grid[:,mid,:], origin='lower')
    plt.colorbar()

    plt.subplot(133)
    plt.title("Z slice")
    plt.imshow(sdf_grid[:,:,mid], origin='lower')
    plt.colorbar()

    plt.show()