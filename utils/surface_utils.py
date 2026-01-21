import pyvista as pv
import pymeshfix
import numpy as np
from loguru import logger
import trimesh
# import pyacvd
# from mesh_to_sdf import sample_sdf_near_surface
# import gc


# def remesh(mesh, n_points=30000):
#     clus = pyacvd.Clustering(mesh)
#     n_subdivs = round(np.log(n_points // mesh.n_points + 1)) + 1
#     clus.subdivide(n_subdivs)
#     clus.cluster(n_points)
#     return clus.create_mesh()

def check_watertight(mesh: pv.PolyData):
    """
        Checks if there are any boundary edges in the mesh.
    """
    
    boundary_edges = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            non_manifold_edges=False,
            manifold_edges=False,
    )

    return boundary_edges.n_cells == 0

def make_surface_watertight(surface_mesh: pv.PolyData, name = ""):

    vertices = surface_mesh.points

    faces = surface_mesh.faces.reshape((-1, 4))[:, 1:4]

    mf = pymeshfix.MeshFix(vertices, faces)

    mf.repair()
    
    vertices_fixed, faces_fixed = mf.v, mf.f

    faces_pv = np.hstack([np.full((faces_fixed.shape[0], 1), 3), faces_fixed]).astype(np.int64)

    faces_pv = faces_pv.ravel()

    surface_closed = pv.PolyData(vertices_fixed, faces_pv)

    if not check_watertight(surface_closed):
        logger.warning(f"Closed mesh {name} is not watertight (found boundary edges)")

    return surface_closed

def scale_to_unit_sphere(points, return_transf_params = False):

    centroid = np.mean( points, axis = 0 )
    points -= centroid

    distances = np.linalg.norm(points, axis=1)
    points /= np.max(distances)

    if return_transf_params:
        return points, centroid, np.max(distances)
    else:
        return points

def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(
            amount - points_available
        )
        return result
    else:
        return unit_sphere_points[:amount, :]

def make_trimesh_from_pv(mesh: pv.UnstructuredGrid | pv.PolyData | trimesh.Trimesh ):
    if not isinstance(mesh, trimesh.Trimesh):
        surface = mesh.extract_surface()
        faces = surface.faces.reshape((-1, 4))[:, 1:] 
        vertices = surface.points
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        return mesh

def make_pv_from_trimesh(mesh: trimesh.Trimesh):
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64)
    return pv.PolyData(mesh.vertices, faces)

def fix_non_manifold_mesh(mesh: trimesh.Trimesh, verbose = False):
    mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    mf.repair()
    verts, faces = mf.v, mf.f
    return trimesh.Trimesh(vertices=verts, faces=faces) 

def sample_surface_by_curvature(
        mesh_raw: trimesh.Trimesh,
        number_of_points = 3000,
        rho = 0.75,
        lamb = 0.2,
        rg = 0.01,
        rm = 0.01,
        verbose_out = False
    ):
    #TODO: the mesh exhibits a lot of curvature around the rims of the veins, correct that
    # # create a lot of points first, then subsample by curvature
    # spc = create_from_scans(
    #     mesh_raw,
    #     scan_count = 128,
    #     scan_resolution = 512,
    #     calculate_normals=False
    # )
    # points = spc.points --> something like 11 250 000 points

    query_points = mesh_raw.sample(count=int(1.5*number_of_points))   
    
    # # smooth a bit before so jagged triangles around patched holes do not get too much weight
    # mesh_pv = make_pv_from_trimesh(mesh_raw)
    # mesh_smooth = mesh_pv.smooth_taubin(n_iter=50, pass_band=0.1)
    # mesh_smooth = make_trimesh_from_pv(mesh_smooth.extract_surface())
    mesh_smooth = mesh_raw

    if verbose_out:
        logger.info(f"Computing gaussian and mean curvature on {len(query_points)} points ... ")

    gaussian_curvs = trimesh.curvature.discrete_gaussian_curvature_measure(mesh_smooth, query_points, radius=rg)

    mean_curvs = trimesh.curvature.discrete_mean_curvature_measure(mesh_smooth, query_points, radius=rm)

    if verbose_out:
        logger.info("Done")

    gaussian_w = np.abs(gaussian_curvs) ** rho 

    mean_w = np.abs(mean_curvs) ** rho

    norm_g = np.sum(gaussian_w)

    norm_m = np.sum(mean_w)

    w_lamb_rho = (1- lamb) * (gaussian_w / norm_g) + lamb * (mean_w / norm_m)

    # remove NaNs
    # print(f"Found {np.isnan(w_lamb_rho).sum()} NaNs in prob vector")
    w_lamb_rho = np.nan_to_num(w_lamb_rho, nan=0.0)
    w_lamb_rho = w_lamb_rho / np.sum(w_lamb_rho)

    idxs = np.random.choice( np.arange(len(query_points)), size=number_of_points, p=w_lamb_rho)

    samples = query_points[idxs]

    return samples

def sample_surface_for_deepsdf(
        surface_mesh: trimesh.Trimesh,
        number_of_points,
        sample_surface_method = "curvature",
        rho = 0.75,
        lamb = 0.2,
        use_deepsdf_convention = True,
        verbose_out = False
    ):
    """
        Create samples near a surface in DeepSDF style.
    """

    #TODO: always make sure it's scaled to unit sphere before sampling. 

    if not isinstance(surface_mesh, trimesh.Trimesh):
        surface_mesh = make_trimesh_from_pv(surface_mesh)

    query_points = []

    if use_deepsdf_convention:
        surface_sample_count = int(number_of_points * 47/50) // 2
    else:
        surface_sample_count = number_of_points // 2

    if sample_surface_method == "curvature":
        surface_points = sample_surface_by_curvature(
            surface_mesh,
            surface_sample_count,
            rho=rho,
            lamb=lamb,
            verbose_out=verbose_out
        )
    elif sample_surface_method == "uniform":
        surface_points = surface_mesh.sample(count=surface_sample_count)
    else:
        raise ValueError("Unknown sampling method requested, available 'curvature' or 'uniform'.")

    scale1 = 0.0025
    scale2 = scale1 / 10

    query_points.append(
        surface_points + np.random.normal(scale=scale1, size=(surface_sample_count, 3))
    )
    query_points.append(
        surface_points + np.random.normal(scale=scale2, size=(surface_sample_count, 3))
    )

    if use_deepsdf_convention:
        unit_sphere_sample_count = number_of_points - 2*surface_sample_count
        unit_sphere_points = sample_uniform_points_in_unit_sphere(unit_sphere_sample_count)
        query_points.append(unit_sphere_points)
    
    query_points = np.concatenate(query_points).astype(np.float32)

    return query_points

# def compute_sdf_at_points(surface: trimesh.Trimesh, pointcloud):
#     """
#         wrapper of sample_sdf_near_surface to avoid memory leaks when using it repeatedly,
#         since it build huge Scan objects containing millions of points and uses OpenGL renderers
#     """
#     surface = make_trimesh_from_pv(surface)

#     result = sample_sdf_near_surface(
#         mesh_raw = surface,
#         pointcloud=pointcloud
#     )

#     query_points, sdfs = result

#     del surface
#     del pointcloud

#     gc.collect()

#     return sdfs
