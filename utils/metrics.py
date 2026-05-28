from scipy.spatial import KDTree
import numpy as np
import pyvista as pv
from .surface_utils import remesh
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from vtk import vtkSampleFunction, vtkImplicitPolyDataDistance
from vtk.util.numpy_support import vtk_to_numpy

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def chamfer_distance_L2(points1, points2):
    if len(points1) == 0 or len(points2) == 0:
        return float("nan")
    tree = KDTree(points1)
    dists_1, _ = tree.query(points2)
    tree = KDTree(points2)
    dists_2, _ = tree.query(points1)
    chd = 0.5 * ( np.mean(dists_1)  + np.mean(dists_2) )

    return float(chd)

def varifold_inner(faces1, faces2, gamma = 1.0, block=2048, device = "cuda"):
    faces1 = faces1.to(device)
    faces2 = faces2.to(device)

    c1 = faces1[:, :3]
    n1 = faces1[:, 3:6]
    a1 = faces1[:, 6]

    c2 = faces2[:, :3]
    n2 = faces2[:, 3:6]
    a2 = faces2[:, 6]

    total = torch.zeros(1, device=device)

    c2_norm2 = (c2 ** 2).sum(dim=1)  # (N2,)
    n2T = n2.t()                     # (3, N2)

    for i in range(0, c1.shape[0], block):
        c1b = c1[i:i+block]
        n1b = n1[i:i+block]
        a1b = a1[i:i+block]

        # squared distances (B, N2)
        d2 = (
            (c1b ** 2).sum(dim=1)[:, None]
            + c2_norm2[None, :]
            - 2 * c1b @ c2.t()
        )

        K = torch.exp(-gamma * d2)

        # normal dot products (B, N2)
        Ndot = n1b @ n2T

        total += torch.sum(
            K * Ndot * a1b[:, None] * a2[None, :]
        )

    return total

def LDDMM_loss(mesh1: pv.PolyData, mesh2: pv.PolyData, compute_normals = True, remeshing = True, n_points = 50000, gamma = 1.0, device = "cuda"):

    # remeshing to have same resolution
    if remeshing:
        # logger.info("Remeshing")
        mesh1 = remesh(mesh1, n_points)
        mesh2 = remesh(mesh2, n_points)

    # logger.info("Extracting faces data")
    data = [None, None]

    for i,m in enumerate([mesh1, mesh2]):

        if compute_normals:
            m.compute_normals(
                cell_normals=True,
                point_normals=False,
                auto_orient_normals=True,
                split_vertices=False,
                inplace=True
            )

        faces = m.faces.reshape((-1, 4))[:, 1:4]

        cell_centers = m.cell_centers().points

        cell_normals = m.cell_normals

        v0 = m.points[faces[:, 0]]
        v1 = m.points[faces[:, 1]]
        v2 = m.points[faces[:, 2]]
        cell_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        data[i] = np.concatenate([cell_centers, cell_normals, cell_areas[:,None]], axis=1)

    faces1 = torch.from_numpy( data[0]).to(dtype=torch.float32)
    faces2 = torch.from_numpy( data[1] ).to(dtype=torch.float32 )

    # print("Number of faces mesh 1: ", faces1.shape )
    # print("Number of faces mesh 2: ", faces2.shape )

    # logger.info("Computing LDDMM loss")
    K11 = varifold_inner(faces1, faces1, gamma, device=device)
    K22 = varifold_inner(faces2, faces2, gamma, device=device)
    K12 = varifold_inner(faces1, faces2, gamma, device=device)

    dL = K11 + K22 - 2*K12

    return dL.cpu().detach().numpy()

def haussdorff(points1, points2):
    
    tree = KDTree(points1)
    dists_1, _ = tree.query(points2)
    tree = KDTree(points2)
    dists_2, _ = tree.query(points1)

    HD = max( max(dists_1), max(dists_2) )
    
    return HD

def chamfer_and_haussdorff(points1, points2):
    if len(points1) == 0 or len(points2) == 0:
        return float("nan")
    tree = KDTree(points1)
    dists_1, _ = tree.query(points2)
    tree = KDTree(points2)
    dists_2, _ = tree.query(points1)

    chd = np.mean(dists_1)  + np.mean(dists_2)
    hdd = max( max(dists_1), max(dists_2) )

    return {"chamfer" : chd, "haussdorff" : hdd}




def f1_score_function(points_pred, points_gt, tau):

    if len(points_pred) == 0 or len(points_gt) == 0:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1score": float("nan")
        }
    
    tree_gt = KDTree(points_gt)
    d_pred_to_gt, _ = tree_gt.query(points_pred)

    tree_pred = KDTree(points_pred)
    d_gt_to_pred, _ = tree_pred.query(points_gt)

    precision = np.mean(d_pred_to_gt < tau)
    recall = np.mean(d_gt_to_pred < tau)

    if precision + recall == 0:
        f1_score = 0.0
    
    else:
        f1_score = 2 * precision * recall/ ( precision + recall)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1score": float(f1_score)
    }

def sdf_gt_on_regular_grid(mesh_gt, resolution, box_lim):
    implicit = vtkImplicitPolyDataDistance()
    implicit.SetInput(mesh_gt)

    sampler = vtkSampleFunction()
    sampler.SetImplicitFunction(implicit)
    sampler.SetModelBounds(
        -box_lim, box_lim,
        -box_lim, box_lim,
        -box_lim, box_lim
    )
    sampler.SetSampleDimensions(resolution, resolution, resolution)
    sampler.ComputeNormalsOff()
    sampler.Update()

    sdf_vtk = vtk_to_numpy(
        sampler.GetOutput().GetPointData().GetScalars()
    )

    # VTK usa x come asse più veloce; riordino per matchare np.ravel C-style
    sdf_grid = sdf_vtk.reshape(
        (resolution, resolution, resolution),
        order="F"
    ).ravel(order="C")

    return sdf_grid

def compute_dice_score(sdf_pred, sdf_gt, level=0.0, eps=1e-8):

    pred_occ = sdf_pred <= level
    gt_occ = sdf_gt <= level

    intersection = np.logical_and(pred_occ, gt_occ).sum()

    pred_volume = pred_occ.sum()
    gt_volume = gt_occ.sum()

    dice = (2.0 * intersection + eps) / (pred_volume + gt_volume + eps)

    return dice


def compute_iou_score(sdf_pred, sdf_gt, level=0.0, eps=1e-8):

    pred_occ = sdf_pred <= level
    gt_occ = sdf_gt <= level

    intersection = np.logical_and(pred_occ, gt_occ).sum()
    union = np.logical_or(pred_occ, gt_occ).sum()

    iou = float((intersection + eps) / (union + eps))

    return iou


if __name__ == "__main__":

    pass


