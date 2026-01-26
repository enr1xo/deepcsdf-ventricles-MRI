import pyvista as pv
import pymeshfix
import numpy as np
from loguru import logger
import trimesh
from scipy.spatial import KDTree
import pyacvd
import open3d as o3d
import igl
import gc


def remesh(mesh, n_points=50000):
    clus = pyacvd.Clustering(mesh)
    n_subdivs = round(np.log(n_points // mesh.n_points + 1)) + 1
    clus.subdivide(n_subdivs)
    clus.cluster(n_points)
    return clus.create_mesh()

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

def make_surface_watertight(surface_mesh: pv.PolyData):

    # --- Step 0: Remove duplicate points & clean non-manifold vertices
    clean_mesh = surface_mesh.clean(
        tolerance=0.0,     # merge points at exactly the same coordinates
        inplace=False
    )
    
    # --- Step 1: Recompute normals for consistent orientation
    # Ensures inside/outside orientation is consistent
    clean_mesh.compute_normals(
        cell_normals=True,
        point_normals=True,
        auto_orient_normals=True,  # automatically flips normals to be consistent
        split_vertices=False,      # keep shared vertices
        inplace=True
    )

    vertices = clean_mesh.points

    faces = clean_mesh.faces.reshape((-1, 4))[:, 1:4]

    mf = pymeshfix.MeshFix(vertices, faces)

    mf.repair()
    
    vertices_fixed, faces_fixed = mf.v, mf.f
    faces_pv = np.hstack([np.full((faces_fixed.shape[0], 1), 3), faces_fixed]).astype(np.int64)
    faces_pv = faces_pv.ravel()

    surface_closed = pv.PolyData(vertices_fixed, faces_pv)

    surface_closed = surface_closed.clean(
        tolerance=0.0,     # merge points at exactly the same coordinates
        inplace=True
    )

    if not check_watertight(surface_closed):
        logger.warning(f"Mesh is not watertight (found boundary edges)")

    # mark patches holes cells
    orig_face_sets = { frozenset(tuple(vertices[v]) for v in tri) for tri in faces } # to lookup
    closed_face_sets = [ frozenset(tuple(vertices_fixed[v]) for v in tri) for tri in faces_fixed ] # to iterate over

    mask = np.ones(len(faces_fixed), dtype=np.int8)
    for i, face_set in enumerate(closed_face_sets):
        if face_set in orig_face_sets:
            mask[i] = 0   # original face

    surface_closed.cell_data["isholepatch"] = mask # true (not 0) ==> not part of original mesh, so is part of a hole patch

    return surface_closed

def scale_to_unit_sphere(points, return_transf_params = False):

    centroid = np.mean( points, axis = 0 )

    scale = np.max( np.linalg.norm(points - centroid, axis=1) )

    if return_transf_params:
        return centroid, scale
    else:
        return ( points - centroid ) / scale

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

def sample_surface_by_curvature(
        mesh_raw: pv.PolyData,
        number_of_points = 3000,
        rho = 0.75,
        lamb = 0.1,
        rg = 0.01,
        rm = 0.01,
        verbose_out = False
    ):

    mesh_raw = make_trimesh_from_pv(mesh_raw)

    query_points, face_idxs = mesh_raw.sample(count=int(2*number_of_points), return_index=True)

    if verbose_out:
        logger.info(f"Computing gaussian and mean curvature for {len(query_points)} points ...")
                
    gaussian_curvs = trimesh.curvature.discrete_gaussian_curvature_measure(mesh_raw, query_points, radius=rg)

    mean_curvs = trimesh.curvature.discrete_mean_curvature_measure(mesh_raw, query_points, radius=rm)

    gaussian_w = np.abs(gaussian_curvs) ** rho 

    mean_w = np.abs(mean_curvs) ** rho

    norm_g = np.sum(gaussian_w)

    norm_m = np.sum(mean_w)

    w_lamb_rho = (1- lamb) * (gaussian_w / norm_g) + lamb * (mean_w / norm_m)

    w_lamb_rho = np.nan_to_num(w_lamb_rho, nan=0.0)
    w_lamb_rho = w_lamb_rho / np.sum(w_lamb_rho)

    idxs = np.random.choice( np.arange(len(query_points)), size=number_of_points, p=w_lamb_rho, replace=False)

    samples = query_points[idxs]

    return samples

def sample_surface_for_deepsdf(
        surface_mesh: pv.PolyData,
        number_of_points,
        sample_surface_method = "curvature",
        rho = 0.75,
        lamb = 0.2,
        use_deepsdf_convention = True,
        ratio = 47/50,
        verbose_out = False
    ):
    """
        Create samples near a surface (optionally) in DeepSDF style.
    """

    query_points = []

    if use_deepsdf_convention:
        surface_sample_count = int(number_of_points * ratio) // 2
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
        surface_points = make_trimesh_from_pv(surface_mesh).sample(count=surface_sample_count)
    else:
        raise ValueError("Unknown sampling method requested, available 'curvature' or 'uniform'.")

    # these scales make sense when points are in unit-sphere like range, make sure surface mesh is !!
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
        # move sphere center at the centroid of the points on the surface, so they are around the shape
        # evenly, even if it is la or ra endocardium and they happen to be in their real position, not translated to the origin
        centroid = np.mean( np.concatenate(query_points, axis = 0), axis = 0 )
        unit_sphere_points += centroid
        query_points.append(unit_sphere_points)
    
    query_points = np.concatenate(query_points).astype(np.float32)

    return query_points

def compute_signed_distance_o3d(mesh: pv.PolyData, query_points):

    vertices = mesh.points
    faces_raw = mesh.faces.reshape(-1, 4)
    faces = faces_raw[:, 1:4]
    
    # Build triangle mesh tensor
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.float32),
        o3d.core.Tensor(faces, dtype=o3d.core.int32),
    )

    # Scene for distance queries
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh)

    # Query points to tensor
    queries = o3d.core.Tensor(query_points, dtype=o3d.core.float32)

    # Unsigned distance
    unsigned = scene.compute_distance(queries).numpy()

    # Signed distance via winding number
    occupancy = scene.compute_occupancy(queries).numpy() # implemented via ray casting + parity counting.`
    # occupancy ∈ [0,1], >0.5 = inside
    inside = occupancy > 0.5

    sign = np.where(inside, -1.0, 1.0)

    return sign * unsigned

def compute_signed_distance_libigl(mesh: pv.PolyData, query_points):

    # check again meshes are watertight! --> maybe original are, but then scaling them down introduces small numerical error in vertices so that mesh doesnìt result watertight really anymore ...
    if not check_watertight(mesh):
        logger.error("Going to compute SDF on a mesh that doesn't result watertight: found boundary edges. This may be small numerical errors introduced by previously scaling the meshes.")

    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    elements = faces.astype(np.int32)

    # TODO: automatic inside-outisde orientation, instead of manually flipping sign if it's opposite ...

    sq_d, _, _ = igl.point_mesh_squared_distance(
        P = query_points,
        V = vertices,
        Ele = elements
    )

    w = igl.fast_winding_number(V = vertices, F = elements, Q = query_points.astype(np.float64))

    sdf = np.sqrt(sq_d) * np.sign(w - 0.5)

    return sdf * -1

if __name__ == "__main__":

    from pathlib import Path
    from tqdm import tqdm
    
    # PATIENT_MESHES_DIR = Path("/home/davidenava_linux/DATASETS/AtrialGeometriesData")

    # patient_names = [subdir.name for subdir in PATIENT_MESHES_DIR.iterdir()] 
    
    # search_for_mesh_files = ["_epicardium_surface", "_LA_endo_surface", "_RA_endo_surface"]
    # for i in tqdm(range(len(patient_names))):
    #     patient = patient_names[i]
    #     subdir = PATIENT_MESHES_DIR / patient
    #     for meshname in search_for_mesh_files:
    #         mesh_file = next( subdir.rglob(f"*{meshname}.vtp"), None)
    #         mesh = pv.read(mesh_file)
    #         # be sure every one is closed
    #         mesh_closed = make_surface_watertight(mesh)
    #         save_fname = patient + meshname + "_closed.vtp"
    #         mesh_closed.save(subdir / save_fname)


        
            

