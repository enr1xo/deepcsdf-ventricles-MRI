import subprocess, re
from scipy.spatial import KDTree
import numpy as np

def chamfer_distance(points1, points2):
    if len(points1) == 0 or len(points2) == 0:
        return float("nan")
    tree = KDTree(points1)
    dists_1, _ = tree.query(points2)
    tree = KDTree(points2)
    dists_2, _ = tree.query(points1)
    return np.sum(dists_1) / len(points2) + np.sum(dists_2) / len(points1)

def MC_EMD(pc1_file, pc2_file, **kwargs):
    """
        Computes Earth Mover's distance approximation for wanted point clouds. 
        Uses repreated subsamples of points from the given clouds for memory saving (Monte Carlo approx).
        Run as a subprocess to utilize JAX's cuda capabilities and jit.

        Args:
            pc1_file: path to .npy file containing points (N,3) for point cloud P
            pc2_file: path to .npy file containing points (M,3) for point cloud Q

        Computes an approximation of EMD(P,Q) using samples and sinkhorn algorithm.
    """

    epsilon = kwargs.get("epsilon", 0.001)
    max_trials = kwargs.get("max_trials", 150)
    num_samples_per_trial = kwargs.get("num_samples_per_trial", 1024)
    e_rel = kwargs.get("epsilon_rel", 0.005)

    # Build subprocess args list using variables: never use verbose!!
    python_exe = "/home/davidenava_linux/FiredrakeJAXProjects/venv-firedrake/bin/python"
    script_path = "/home/davidenava_linux/FiredrakeJAXProjects/utils/earth_mover.py"
    cmd = [
        python_exe, script_path,
        "--pointcloud_files", pc1_file, pc2_file,
        "--epsilon", str(epsilon),
        "--max_trials", str(max_trials),
        "--num_samples_per_trial", str(num_samples_per_trial),
        "--epsilon_rel", str(e_rel)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout

    # parse value from printed output
    match = re.search(r"Mean EMD:\s*([0-9.eE+-]+)", output)
    if match:
        mean_emd = float(match.group(1))

    ci_match = re.search(r"95% C\.I\. : \(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)", output)
    if ci_match:
        ci_lower = float(ci_match.group(1))
        ci_upper = float(ci_match.group(2))
        CI = (ci_lower, ci_upper)

    return mean_emd, CI


if __name__ == "__main__":

    pass

    # region EARTH MOVER'S
    # pc_1_file = "/home/davidenava_linux/AtriaProject/data/point_clouds/AF069_epicardium_vertices.npy"
    # pc_2_file = "/home/davidenava_linux/AtriaProject/data/point_clouds/LEU_NORM_F009_epicardium_vertices.npy"
    # epsilon = 0.001

    # mean_emd, CI = MC_EMD(pc_1_file, pc_2_file, epsilon = 0.001, max_trials = 200)

    # print(f"Mean EMD: {mean_emd}")
    # print(f"95% C.I. : {CI}")

    # region CHAMFER DISTANCE
    # pc_1_file = "/home/davidenava_linux/AtriaProject/data/point_clouds/AF069_epicardium_vertices.npy"
    # pc_2_file = "/home/davidenava_linux/AtriaProject/data/point_clouds/LEU_NORM_F009_epicardium_vertices.npy"
    # P, Q = np.load(pc_1_file), np.load(pc_2_file)
    # chd = chamfer_distance(P,Q)
    # print("Chamfer distance: ", chd)