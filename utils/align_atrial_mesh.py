import numpy as np
import trimesh
from mesh_to_sdf.surface_point_cloud import sample_from_mesh
from .surface_utils import make_trimesh_from_pv
from loguru import logger
if not hasattr(np, "infty"): # to not have conflicts with old versions of numpy
    np.infty = np.inf


def get_tagged_centroids(mesh, all_tags, wanted_tags):
    
    centroids = []

    for tag in wanted_tags:
        
        m = mesh.extract_cells( np.isin( all_tags, [tag]) )

        centroid = np.mean(m.points, axis=0)

        centroids.append(centroid)
    
    assert len(centroids) % 2 == 0, "Expected two tags (epi/endo) for each vein"

    centroids = np.array(centroids)

    # get a single point for every tagged element from the split epi/endo
    D = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    nearest_idx = np.argmin(D, axis=1)
    mask = np.arange(len(centroids)) < nearest_idx
    halved_centroids = (centroids + centroids[nearest_idx]) / 2
    halved_centroids = halved_centroids[mask]

    return halved_centroids

def rigid_transform(source, target):
    """
    Computes the best-fit rigid transformation that aligns points in source to points in target
    using the left-multiplication convention:

        target ≈ R @ source + t

    (N, 3) source points (to move)
    (N, 3) target points (reference)

    Returns:
        R (3x3), t (3,)
    such that:
        source_aligned = (R @ source.T + t[:, None]).T ≈ target
    """
    if source.shape != target.shape:
        raise AssertionError(f"Shape mismatch: A {source.shape}, B {target.shape}")

    # Compute centroids
    centroid_s = source.mean(axis=0)
    centroid_t = target.mean(axis=0)

    # Center the point clouds
    S = (source - centroid_s).T  # 3xN
    T = (target - centroid_t).T  # 3xN

    # Covariance matrix (note: BB @ AA.T for left-mult convention)
    H = T @ S.T

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Reflection fix
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Translation
    t = centroid_t - R @ centroid_s

    return R, t

def skew(v):
    """Return 3x3 skew-symmetric matrix of vector v (3,)."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

def build_icp_ptpl_system(source_pc_current, target_pc_matched, normals_matched):
    """
    Build point-to-plane linear system A * delta_xi = b using current transformed source points.
    source_pc_current: (N,3) = current transformed source points (R @ p + t)
    target_pc_matched: (N,3) matched target points q
    normals_matched: (N,3) normals at the matched targets (in world frame)
    Returns A (N,6), b (N,)
    """
    N = source_pc_current.shape[0]
    A = np.zeros((N, 6), dtype=float)
    b = np.zeros(N, dtype=float)

    for i in range(N):
        p_p = source_pc_current[i]
        q = target_pc_matched[i]
        n = normals_matched[i]

        A[i, :3] = - (n @ skew(p_p))   # rotation part (1x3)
        A[i, 3:] = n                   # translation part (1x3)
        b[i] = n @ (q - p_p)           # scalar residual

    return A, b

def icp_point_to_plane(
        source_pc,
        target_pc,
        target_normals,
        target_kdtree,
        max_iter=50,
        max_rot_deg=10,
        max_trasl=0.05,
        keep_metrics_log = False,
        verbose_out = True
    ):
    """
        Point-to-plane ICP, align source point cloud to target point cloud
        source_pc: (Ns,3) original source points
        target_pc: (Nt,3)
        target_normals: (Nt,3) (in world frame)
        target_kdtree: KD-tree built on target_pc
        Returns transformed source_pc_transformed, R, t, metrics_log
    """

    if keep_metrics_log:
        metrics_log = {
            "rmse": [],
            "mean_error": [],
            "rotation_mag": [],
            "translation_mag": []
        }
    else:
        metrics_log = None

    R = np.eye(3)
    t = np.zeros(3)

    source_pc_moved = (R @ source_pc.T).T + t  

    for it in range(max_iter):
        
        if verbose_out:
            logger.info(f"ICP point to plane iteration {it+1}/{max_iter}")

        # build current point correspondances, using available kdtree for speed
        dists, idxs = target_kdtree.query(source_pc_moved, k=1)
        idxs = np.asarray(idxs).ravel()
        dists = np.asarray(dists).ravel()

        tgt_matched = target_pc[idxs]
        nrm_matched = target_normals[idxs]

        # linearize nonlinear least square objective around current source point pose
        A, b = build_icp_ptpl_system(source_pc_moved, tgt_matched, nrm_matched)

        # Solve least squares A delta_xi = b
        # delta_xi = [delta_rot (3,), delta_trans (3,)] where delta_trans is expressed in world frame
        try:
            delta_xi, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception as e:
            logger.error("lstsq failed: %s", e)
            break

        delta_rot = delta_xi[:3]
        delta_trasl = delta_xi[3:]

        # Clamp updates
        rot_mag = np.linalg.norm(delta_rot)
        max_theta = np.deg2rad(max_rot_deg)
        if rot_mag > max_theta and rot_mag > 0:
            delta_rot = (delta_rot / rot_mag) * max_theta

        trans_mag = np.linalg.norm(delta_trasl)
        if trans_mag > max_trasl and trans_mag > 0:
            delta_trasl = (delta_trasl / trans_mag) * max_trasl

        # Convert rotation vector (axis-angle) to matrix (Rodrigues' formula)
        if np.linalg.norm(delta_rot) < 1e-12:
            R_delta = np.eye(3)
        else:
            theta = np.linalg.norm(delta_rot)
            k = delta_rot / theta
            K = skew(k)
            R_delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        # For point-to-plane we treat delta_trans as world-frame (since normals/residuals are in world frame)
        R = R_delta @ R
        t = t + delta_trasl

        # apply: always move from initial points to current composed transformation instead of carrying over the points
        # this guarantees I am doing it right otherwise I seem to mess up
        source_pc_moved = (R @ source_pc.T).T + t

        # Compute residuals 
        # residuals are n^T (q - current_pose)
        if keep_metrics_log:
            res = nrm_matched * (tgt_matched - source_pc_moved[:len(tgt_matched)])
            residuals = np.einsum('ij,ij->i', nrm_matched, tgt_matched - source_pc_moved[:len(tgt_matched)])
            rmse = np.sqrt(np.mean(residuals**2))
            mean_err = np.mean(np.abs(residuals))

            metrics_log["rmse"].append(rmse)
            metrics_log["mean_error"].append(mean_err)
            metrics_log["rotation_mag"].append(np.linalg.norm(delta_rot))
            metrics_log["translation_mag"].append(np.linalg.norm(delta_trasl))

        # Optional quick convergence check
        if np.linalg.norm(delta_rot) < 1e-6 and np.linalg.norm(delta_trasl) < 1e-6:
            if verbose_out:
                logger.info(f"Stopping at iteration {it+1}, update too small ( < 1e-6 ). RMSE = {rmse}.")
            break

    return source_pc_moved, R, t, metrics_log

def icp_point_to_point(
        source_pc,
        target_pc,
        target_kdtree,
        max_iter = 50,
        keep_metrics_log = False,
        verbose_out = False
    ):

    if keep_metrics_log:
        metrics_log = {
            "rmse": [],             
            "mean_error": [],       
            "rotation_angle": [],     
            "translation_mag": []   
        }
    else:
        metrics_log = None
    
    R_tot = np.eye(3)
    t_tot = np.zeros(3)

    source_pc_moved = source_pc.copy()

    for it in range(max_iter):
        
        if verbose_out:
            logger.info(f"ICP iteration {it+1}/{max_iter}")

        _, idxs = target_kdtree.query( source_pc_moved, k=1)

        idxs = np.asarray(idxs).flatten()

        target_pc_matched = target_pc[idxs]

        R,t = rigid_transform(source_pc_moved, target_pc_matched)

        source_pc_moved = (R @ source_pc_moved.T + t[:, None]).T

        R_tot = R @ R_tot
        t_tot = R @ t_tot + t

        residuals = np.linalg.norm(target_pc_matched - source_pc_moved, axis=1)
        rmse = np.sqrt(np.mean(residuals**2))
        
        if keep_metrics_log:
            metrics_log["rmse"].append( rmse )
            metrics_log["mean_error"].append( np.mean(residuals) )
            metrics_log["rotation_angle"].append( np.arccos((np.trace(R) - 1)/2) )
            metrics_log["translation_mag"].append( np.linalg.norm(t) )

        val = (np.trace(R) - 1) / 2
        val = np.clip(val, -1.0, 1.0)

        if np.arccos(val) < 1e-6 and np.linalg.norm(t) < 1e-6:
            # logger.info(f"Stopping at iteration {it+1}, update too small ( < 1e-6 ).")
            break

    return source_pc_moved, R_tot, t_tot, metrics_log

def icp(
    source_pc,
    target_pc,
    target_kdtree,
    target_normals = None,
    max_iter=50,
    max_rot_deg=10,
    max_trasl=0.05,
    keep_metrics_log = False,
    verbose_out = False):
    " Wrapper to either select ICP point to point or point to plane depending on wheter normals are passed as an argument or not. "

    if target_normals is not None:
        return icp_point_to_plane(
            source_pc,
            target_pc,
            target_normals,
            target_kdtree,
            max_iter,
            max_rot_deg,
            max_trasl,
            keep_metrics_log,
            verbose_out)
    else:
        return icp_point_to_point(
            source_pc,
            target_pc,
            target_kdtree,
            max_iter,
            keep_metrics_log,
            verbose_out
        )
    
def apply_icp_result(R,t,P, mult_convention = "left"):
    return (R @ P.T + t[:, None]).T if mult_convention == "left" else P @ R + t
    
def align_to_reference_mesh(
        source_mesh,
        target_mesh,
        num_source_samples = 10000,
        num_target_samples = 20000,
        method = "point",
        max_iter = 25,
        max_rot_deg = 10,
        max_trasl = 0.1,
        keep_metrics_log = False,
        verbose_out = False
        ):
    """
        Uses Iterative Closest Point algorithm to align the source shape to the target shape, by sampling uniform point clouds on their surfaces.

        Args:
            source_mesh: UnstructuredGrid, PolyData, or Trimesh object. The output will be of the same type
            target_mesh: UnstructuredGrid, PolyData, or Trimesh object
            method: str, if 'point' the classic point to point correspondance algorithm is used, if 'plane', the point to plane variant
        
        Returns:
            A copy of source_mesh with transformed points locations, rotation and translation of the transformation, and optionally some metrics.
    """

    #TODO: make returning R and t available instead of modified points only

    if method == "point":
        calculate_normals = False
    elif method == "plane":
        calculate_normals = True
    else:
        raise ValueError(f"Unknown ICP method requested. Can be 'point' or 'plane', got {method}.")

    source_surface = make_trimesh_from_pv(source_mesh)
    
    target_surface = make_trimesh_from_pv(target_mesh)

    source_spc = sample_from_mesh(
        mesh = source_surface,
        sample_point_count = num_source_samples,
        calculate_normals = False
    )

    target_spc = sample_from_mesh(
            mesh = target_surface,
            sample_point_count = num_target_samples,
            calculate_normals = calculate_normals
        )
    
    normals = target_spc.normals # None if not requested --> use icp point to point
    
    source_pc = source_spc.points

    target_pc = target_spc.points
    
    target_kdtree = target_spc.kd_tree

    # ICP
    _, R, t, metrics_log = icp(source_pc, target_pc, target_kdtree, normals, max_iter,
                                max_rot_deg, max_trasl, keep_metrics_log, verbose_out)

    source_mesh_moved = source_mesh.copy()

    if not isinstance(source_mesh, trimesh.Trimesh):
        source_mesh_moved.points = apply_icp_result(R, t, source_mesh.points)
    else:
        source_mesh_moved.vertices = apply_icp_result(R, t, source_mesh.vertices)


    if metrics_log is not None:
        return source_mesh_moved, R, t, metrics_log 
    else:
        return source_mesh_moved, R, t
        























"""

    #region align landmarks
    #TODO: try to align like volumetric pieces that are almost equal in all the meshes, like that ciambella-like thing between the atria
    # # # extract centroids of principal veins and valves :  !!! tags are splitted also between epi and endo !!!

    # landmarks_1 = get_tagged_centroids(
    #     atria_mesh_1.mesh,
    #     atria_mesh_1.elemTags, 
    #     atria_mesh_1.veins_tags + atria_mesh_1.valves_tags
    # )

    # landmarks_2 = get_tagged_centroids(
    #     atria_mesh_2.mesh,
    #     atria_mesh_2.elemTags, 
    #     atria_mesh_2.veins_tags + atria_mesh_2.valves_tags
    # )

    # R, t = rigid_transform(landmarks_1, landmarks_2)
    
    # plotter = pv.Plotter()
    # plotter.add_mesh( atria_mesh_1.mesh, color = 'black', opacity = 0.7, show_edges = False)
    # plotter.add_mesh( atria_mesh_2.mesh, color = 'red', opacity = 0.7, show_edges = False)
    # # plotter.add_points( landmarks_1, color = 'red')
    # # plotter.add_points( landmarks_2, color = 'red')
    # plotter.show()

    # atria_mesh_2.mesh.points = atria_mesh_2.mesh.points @ R + t
    # landmarks_2 = landmarks_2 @ R + t

    # plotter = pv.Plotter()
    # plotter.add_mesh( atria_mesh_1.mesh, color = 'white', opacity = 0.7, show_edges = False)
    # plotter.add_mesh( atria_mesh_2.mesh, color = 'yellow', opacity = 0.7, show_edges = False)
    # plotter.add_points( landmarks_1, color = 'red')
    # plotter.add_points( landmarks_2, color = 'red')
    # plotter.show()

    #region align FO: using only this actually isn't ideal for the overall shape
    # FO_1 = atria_mesh_1.mesh.extract_cells( np.isin(atria_mesh_1.elemTags, [82,97]) )
    
    # FO_2 = atria_mesh_2.mesh.extract_cells( np.isin(atria_mesh_2.elemTags, [82,97]) )

    # surface = atria_mesh_1.mesh.extract_surface()
    # faces = surface.faces.reshape((-1, 4))[:, 1:] 
    # vertices = surface.points
    # # vertices = scale_to_unit_sphere(vertices)
    # target_surface = trimesh.Trimesh(vertices=vertices, faces=faces)

    # surface = atria_mesh_2.mesh.extract_surface()
    # faces = surface.faces.reshape((-1, 4))[:, 1:] 
    # vertices = surface.points
    # # vertices = scale_to_unit_sphere(vertices)
    # source_surface = trimesh.Trimesh(vertices=vertices, faces=faces)

    # source_surface = AtrialGeometry.make_trimesh_from_pv(atria_mesh_2.mesh)
    
    # target_surface = AtrialGeometry.make_trimesh_from_pv(atria_mesh_1.mesh)

    # source_spc = sample_from_mesh(
    #     mesh = source_surface,
    #     sample_point_count = 10000,
    #     calculate_normals = False
    # )

    # target_spc = sample_from_mesh(
    #     mesh = target_surface,
    #     sample_point_count = 20000,
    #     calculate_normals = True
    # )

    # source_pc = source_spc.points

    # target_pc = target_spc.points
    
    # normals = target_spc.normals

    # target_kdtree = target_spc.kd_tree

    # # ICP
    # _, R, t, log_metrics = icp_point_to_point(source_pc, target_pc, target_kdtree)

    

    # # ICP point to plane
    # _, R_ptpl, t_ptpl, log_metrics_ptpl = icp_point_to_plane(source_pc, target_pc, normals, target_kdtree)



    # # Compute relative rotation
    # R_diff = R.T @ R_ptpl
    # trace_val = np.trace(R_diff)
    # cos_theta = (trace_val - 1) / 2.0
    # cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # theta = np.arccos(cos_theta)
    # print(f"Rotation difference between classic ICP and point to plane: {theta} degrees" )


    # source_icp = apply_icp_result(R, t, atria_mesh_2.mesh.points)
    
    # source_icp_ptpl = apply_icp_result(R_ptpl, t_ptpl, atria_mesh_2.mesh.points)

    # atria_mesh_2.mesh.points = source_icp
    # plotter = pv.Plotter()
    # plotter.add_mesh(atria_mesh_2.mesh, color = 'red', opacity = 0.8)
    # plotter.add_mesh(atria_mesh_1.mesh, color = 'blue', opacity = 0.8)
    # plotter.show_grid()
    # plotter.show()

    # atria_mesh_2.mesh.points = source_icp_ptpl
    # plotter = pv.Plotter()
    # plotter.add_mesh(atria_mesh_2.mesh, color = 'red', opacity = 0.8)
    # plotter.add_mesh(atria_mesh_1.mesh, color = 'blue', opacity = 0.8)
    # plotter.show_grid()
    # plotter.show()


    # for k,item in log_metrics.items():
    #     plt.figure()
    #     plt.plot( np.arange(len(item)), item)
    #     plt.title(k)
    # plt.show()

    
"""



