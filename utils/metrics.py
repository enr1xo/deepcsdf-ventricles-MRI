from scipy.spatial import KDTree
import numpy as np
import pyvista as pv
from .surface_utils import remesh
from loguru import logger
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

def chamfer_distance_L2(points1, points2):
    if len(points1) == 0 or len(points2) == 0:
        return float("nan")
    tree = KDTree(points1)
    dists_1, _ = tree.query(points2)
    tree = KDTree(points2)
    dists_2, _ = tree.query(points1)
    return np.mean(dists_1)  + np.mean(dists_2)


def chamfer_distance_L2_squared(points1, points2):
    if len(points1) == 0 or len(points2) == 0:
        return float("nan")
    tree = KDTree(points1)
    dists_1, _ = tree.query(points2)
    tree = KDTree(points2)
    dists_2, _ = tree.query(points1)
    return np.mean(dists_1 ** 2)  + np.mean(dists_2 ** 2)


def varifold_inner(faces1, faces2, gamma = 1.0, block=2048):
    faces1 = faces1.to("cuda")
    faces2 = faces2.to("cuda")

    c1 = faces1[:, :3]
    n1 = faces1[:, 3:6]
    a1 = faces1[:, 6]

    c2 = faces2[:, :3]
    n2 = faces2[:, 3:6]
    a2 = faces2[:, 6]

    total = torch.zeros(1, device="cuda")

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


def LDDMM_loss(mesh1: pv.PolyData, mesh2: pv.PolyData, compute_normals = True, remeshing = True, n_points = 50000, gamma = 1.0):

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
    K11 = varifold_inner(faces1, faces1, gamma)
    K22 = varifold_inner(faces2, faces2, gamma)
    K12 = varifold_inner(faces1, faces2, gamma)

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


if __name__ == "__main__":

    pass


