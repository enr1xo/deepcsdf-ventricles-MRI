from pathlib import Path
import numpy as np
from loguru import logger
import pyvista as pv
import subprocess
import os
import gc
import json
from tqdm import tqdm
from utils.align_atrial_mesh import apply_icp_result, align_to_reference_mesh
from scipy.spatial import KDTree
import pymeshfix
from utils.surface_utils import (
    check_watertight,
    make_surface_watertight,
    scale_to_unit_sphere,
    sample_surface_for_deepsdf,
    compute_signed_distance_libigl
)

from config_v import (
    PATIENT_MESHES_DIR, 
    ATRIA_TAGS_METADATA,
    DATA_DIR,
    PATIENTS_NPY_DATA_DIR,
    PATIENTS_COORDS_AND_SDFS_DIR,
)


def create_vtu_from_carpbin(
        source_data_dir,
        save_data_dir = None,
        base_source_files_name = "vol_gen.tagged.quality_rdx_fib",
        out_mesh_name = None,
        input_format = "carp_bin"
    ):

    if save_data_dir is None:
        save_data_dir = source_data_dir

    if out_mesh_name is None:
        out_mesh_name = base_source_files_name

    command = [
        "meshtool",
        "convert",
        f"-imsh", None, 
        f"-ifmt", input_format,
        f"-omsh", None,
        f"-ofmt", "vtu"
    ]
    
    if not [f for f in os.listdir(source_data_dir) if f.endswith(".vtu")]: # if there isn't already a .vtu mesh in the source directory

        try:
            command[3] = os.path.join(source_data_dir, base_source_files_name)

            command[7] = os.path.join(save_data_dir, out_mesh_name)

            result = subprocess.run(command, check=True, text=True, capture_output=True)

        except subprocess.CalledProcessError as e:

                logger.error("meshtool failed with error:")

                print(e.stderr)
    else:
        logger.warning("A .vtu file is already present in source directory, skipping creation.")
    
    return

def split_cell_data_tags( mesh_tags, tags_metadata = ATRIA_TAGS_METADATA ):
    """
        Split the tags found in all_tags into tags for right/left epi/endo atrium.
    """
    tags_split = {}

    for key in ["RA_TAGS", "LA_TAGS", "RA_ENDO_TAGS", "LA_ENDO_TAGS", "RA_EPI_TAGS", "LA_EPI_TAGS"]:
        if key in tags_metadata.keys():
            tags_split[key] = [tag for tag in tags_metadata[key] if tag in mesh_tags]
        else:
            logger.warning("Available tags metadata dictionary does not contain key {key}.")

    return tags_split

# def make_surface_consistently_oriented(surface_mesh: pv.PolyData):

#     # check if passed mesh is actually watertight
#     if not check_watertight(surface_mesh):
#         raise ValueError("Trying to orient mesh that is not watertight!")

#     # Ensures inside/outside orientation is consistent
#     surface_mesh.compute_normals(
#         cell_normals=True,
#         point_normals=False,
#         auto_orient_normals=True,  # automatically flips normals to be consistent
#         split_vertices=False,      # keep shared vertices
#         inplace=True
#     )

#     # consistent orientation: normals oriented outward
#     # pick a point inside --> centroid SHOULD be
#     center = surface_mesh.center

#     cell_centers = surface_mesh.cell_centers().points
#     normals = surface_mesh.cell_normals

#     idx = np.random.choice(len(normals), size=10, replace=False)
#     sign = np.sign(np.sum(np.einsum("ij,ij->i", normals[idx], cell_centers[idx] - center)))
#     if sign < 0:
#         surface_mesh.flip_normals()

#     return

def extract_raw_atria_surfaces(mesh, tags_metadata = ATRIA_TAGS_METADATA):

    logger.info("Extracting raw epicardium and left/right endocardium surfaces from volumetric mesh ...")

    mesh.field_data.clear()

    if hasattr(mesh, "cell_data"):
        if "elemTag" in mesh.cell_data:
            elemTags = mesh.cell_data["elemTag"]
            elemtagskey =  "elemTag"
        elif "elemTags" in mesh.cell_data:
            elemTags = mesh.cell_data["elemTags"]
            elemtagskey =  "elemTags"
        else:
            raise ValueError(f"Unknown key to access elements tags in mesh, expected elemTags or elemTag")   
    else:
        raise TypeError("Loaded mesh needs cell_data attribute tagging elements.")

    mesh_tags = set(elemTags) # for lookup 

    split_tags = split_cell_data_tags(mesh_tags, tags_metadata) # TODO: add extracting tags info from reading the .aug file directly?

    RA_endo_tags = split_tags["RA_ENDO_TAGS"]

    LA_endo_tags = split_tags["LA_ENDO_TAGS"]

    RA_epi_tags = split_tags["RA_EPI_TAGS"]

    LA_epi_tags = split_tags["LA_EPI_TAGS"]

    # # =============================================================== #
    # #  Extract surfaces
    # # =============================================================== #
    whole_surface = mesh.extract_surface()
    surface_elemTags = whole_surface.cell_data[elemtagskey] 

    # ===== epicardium ===== #
    # epicardium: 97 is tag for "RA_FO", which is part of endocardium also but it fills a hole in epi connecting it to the left epicardium sometimes
    # if I don't put it may happen I extract two disconnected components for some particular patients [...]
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, RA_epi_tags + LA_epi_tags + [97]) )
    surf = surf.connectivity(extraction_mode = 'largest') 
    epicardium_surface = surf.extract_surface()

    # ===== right atrium endocardium ===== #
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, RA_endo_tags) )
    surf = surf.connectivity(extraction_mode = 'largest')
    RA_endo_surface = surf.extract_surface()

    # ===== left atrium endocardium ===== #
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, LA_endo_tags) )
    surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
    LA_endo_surface = surf.extract_surface()

    return epicardium_surface, RA_endo_surface, LA_endo_surface

def extract_closed_atria_surfaces(mesh : pv.UnstructuredGrid | pv.PolyData, tags_metadata = ATRIA_TAGS_METADATA):
    """
        Extracts raw epicardium, left/right endocardium surfaces from the passed volumetric mesh using elements' tags,
        then closes the surfaces, returning watertight meshes.

        Doesn't perform ANY smoothing or geometric change either than closing holes.

        TODO: add option to close meshes individually, and smooth them also
    """

    logger.info("Extracting epicardium and left/right endocardium surfaces from volumetric mesh ...")
        
    if hasattr(mesh, "cell_data"):
        if "elemTag" in mesh.cell_data:
            elemTags = mesh.cell_data["elemTag"]
            elemtagskey =  "elemTag"
        elif "elemTags" in mesh.cell_data:
            elemTags = mesh.cell_data["elemTags"]
            elemtagskey =  "elemTags"
        else:
            raise ValueError(f"Unknown key to access elements tags in mesh, expected elemTags or elemTag")   
    else:
        raise TypeError("Mesh needs cell_data attribute tagging elements.")

    mesh_tags = set(elemTags) # for lookup 

    split_tags = split_cell_data_tags(mesh_tags, tags_metadata) # TODO: add extracting tags info from reading the .aug file directly?

    RA_endo_tags = split_tags["RA_ENDO_TAGS"]

    LA_endo_tags = split_tags["LA_ENDO_TAGS"]

    RA_epi_tags = split_tags["RA_EPI_TAGS"]

    LA_epi_tags = split_tags["LA_EPI_TAGS"]

    # # =============================================================== #
    # #  Extract surfaces
    # # =============================================================== #
    whole_surface = mesh.extract_surface()
    surface_elemTags = whole_surface.cell_data[elemtagskey] 

    # ===== epicardium ===== #
    # epicardium: 97 is tag for "RA_FO", which is part of endocardium also but it fills a hole in epi connecting it to the left epicardium sometimes
    # if I don't put it may happen I extract two disconnected components for some particular patients [...]
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, RA_epi_tags + LA_epi_tags + [97]) )
    surf = surf.connectivity(extraction_mode = 'largest') 
    surf = surf.extract_surface()
    epicardium_surface = make_surface_watertight(surf)
    if not check_watertight(epicardium_surface):
        logger.error(f"Mesh is not watertight (found boundary edges)")

    patches = epicardium_surface.extract_cells( epicardium_surface.cell_data["isholepatch"] == 1)

    # ===== left atrium endocardium ===== #
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, LA_endo_tags) )
    surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
    surf = surf.extract_surface()
    merged = surf.merge(
        patches,
        merge_points=True,
        tolerance=1e-4 * np.mean(surf.length)
    )
    merged = merged.connectivity(extraction_mode = 'largest')
    LA_endo_surface = merged.extract_surface()
    LA_endo_surface = make_surface_watertight(LA_endo_surface) # to close small imperfections that may remain
    if not check_watertight(LA_endo_surface):
        logger.error(f"Mesh is not watertight (found boundary edges)")

    # ===== right atrium endocardium ===== #
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, RA_endo_tags) )
    surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
    surf = surf.extract_surface()
    merged = surf.merge(
        patches,
        merge_points=True,
        tolerance=1e-4 * np.mean(surf.length)
    )
    merged = merged.connectivity(extraction_mode = 'largest')
    RA_endo_surface = merged.extract_surface()
    RA_endo_surface = make_surface_watertight(RA_endo_surface) # to close small imperfections that may remain
    if not check_watertight(RA_endo_surface):
        logger.error(f"Mesh is not watertight (found boundary edges)")

    # clear field data that has been copied from patches or some original mesh stuff ...
    for m in (epicardium_surface, LA_endo_surface, RA_endo_surface):
        m.field_data.clear()

    return {"epicardium_surface" : epicardium_surface, "LA_endo_surface" : LA_endo_surface, "RA_endo_surface" : RA_endo_surface}

def extract_processed_atria_surfaces(patient_name, reference_name, reference_mesh, source_dir = PATIENT_MESHES_DIR):
    """
        Extracts epicardium, left and right endocardium surfaces from the original volumetric mesh,
        then aligns them to the reference mesh, storing alignment data in the original mesh .vtu file, and returns
        the scaled, closed, and aligned surfaces.
    """

    patient_dir = Path(source_dir) / patient_name

    # VOLUMETRIC MESH
    original_mesh_path = next( patient_dir.glob("*.vtu"), None)
    if not original_mesh_path:
        raise FileNotFoundError(f"Volumetric mesh for patient {patient_name} not found in {patient_dir}.")
    original_mesh = pv.read(original_mesh_path)

    # extract CLOSED surfaces, from ORIGINAL volumetric mesh to avoid extracting from manipulated vertices
    extracted_surfaces = extract_closed_atria_surfaces(original_mesh) # from here I am SURE meshes are watertight. But then I am gonna scale and move them --> numerically this can introduce some errors
    # retrieve extracted closed surfaces
    LA_endo = extracted_surfaces["LA_endo_surface"]
    RA_endo = extracted_surfaces["RA_endo_surface"]
    epicardium = extracted_surfaces["epicardium_surface"]

    # center -> scale
    original_scaled = original_mesh.copy()
    # careful to not have the points passed already scaled by this function !
    # Python passes references to objects; only in-place mutation affects the caller. the function now should be implemented to avoid any in place modification
    centre, max_radius = scale_to_unit_sphere(original_scaled.points, return_transf_params=True)
    original_scaled.points -= centre
    original_scaled.points /= max_radius     
    # apply scaling and alignment to extracted surfaces
    epicardium.points -= centre
    epicardium.points /= max_radius
    LA_endo.points -= centre
    LA_endo.points /= max_radius
    RA_endo.points -= centre
    RA_endo.points /= max_radius

    # now align: use ICP on point clouds AT THE STANDARD UNIT SCALE !! resulting rotation and traslation data will be in this scale
    if patient_name != reference_name: # skip alignment for reference mesh
        reference_mesh_scaled = reference_mesh.copy()
        centre_ref, max_radius_ref = scale_to_unit_sphere(reference_mesh_scaled.points, return_transf_params=True)
        reference_mesh_scaled.points -= centre_ref
        reference_mesh_scaled.points /= max_radius_ref
        # ICP to align
        logger.info("Aligning ...")
        _, R, t = align_to_reference_mesh(
            source_mesh = original_scaled.extract_surface(),
            target_mesh = reference_mesh_scaled.extract_surface(),
            num_source_samples = 10000,
            num_target_samples = 10000,
            max_iter=100,
            verbose_out=False
        )

    if patient_name == reference_name:
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
    else:
        epicardium.points = apply_icp_result(R, t, epicardium.points)
        LA_endo.points = apply_icp_result(R, t, LA_endo.points)
        RA_endo.points = apply_icp_result(R, t, RA_endo.points)

    # store scaling and alignment data in original mesh file 
    original_mesh.field_data[f'alignto{reference_name}-rotation'] = R
    original_mesh.field_data[f'alignto{reference_name}-traslation'] = t
    original_mesh.field_data['centre-centroid'] = centre 
    original_mesh.field_data['scale-tounitradius'] = max_radius 

    # store scaling factor also in every surface data to find it easily later to get back to original scale, without having to load the original mesh
    epicardium.field_data['scale-tooriginalrange'] = max_radius 
    LA_endo.field_data['scale-tooriginalrange'] = max_radius 
    RA_endo.field_data['scale-tooriginalrange'] = max_radius   

    # sanity check: make sure numerical transformations haven't broken watertightness
    if not check_watertight(epicardium):
        logger.error("Epicardium mesh isn't watertight: found boundary edges")

    if not check_watertight(LA_endo):
        logger.error("LA endocardium mesh isn't watertight: found boundary edges")
    
    if not check_watertight(RA_endo):
        logger.error("RA endocardium isn't watertight: found boundary edges")

    ### to go back precisely from original --> extracted in the ORIGINAL SCALE:
    # original_mesh.points -= centre
    # original_mesh.points /= max_radius
    # original_mesh.points = apply_icp_result(R,t,original_mesh.points)
    # original_mesh.points *= epicardium.field_data["scale-tooriginalrange"]

    return {"epicardium_surface" : epicardium, "LA_endo_surface" : LA_endo, "RA_endo_surface" : RA_endo, "original_mesh" : original_mesh}

def _create_processed_surfaces_meshes(
    source_dir = PATIENT_MESHES_DIR,
    save_to_dir = PATIENT_MESHES_DIR,
    reference_patient = "AF069"
):
    """
        Helper to only create all processed meshes first
    """

    patients = [f.name for f in source_dir.iterdir() if f.is_dir()]

    reference_mesh = pv.read(source_dir / "AF069" / "AF069.vtu")

    for patient in patients:
        
        logger.warning(f"Processing patient {patient}")

        patient_save_dir = Path(save_to_dir) / patient

        # if next( patient_save_dir.rglob("*-processed.vtp"), None) is not None:
        #     logger.warning(f"Processed meshes already present at {str(patient_save_dir)}, skipping creation")
        #     continue
        
        Path( patient_save_dir ).mkdir( exist_ok=True)
        
        extracted = extract_processed_atria_surfaces(patient, reference_patient, reference_mesh, source_dir)

        LA_endo = extracted["LA_endo_surface"]
        RA_endo = extracted["RA_endo_surface"]
        epicardium = extracted["epicardium_surface"]
        original = extracted["original_mesh"]

        logger.info("Saving processed surfaces meshes")
        
        epicardium.save( patient_save_dir / "epicardium-processed.vtp")
        LA_endo.save( patient_save_dir / "la_endo-processed.vtp")
        RA_endo.save( patient_save_dir / "ra_endo-processed.vtp")

        original.save( patient_save_dir / f"{patient}.vtu")

    return

def _create_deepsdf_data_npy(
    source_dir,
    save_to_dir,
    reference_patient = "AF069",
    num_epi_samples=None,
    num_lendo_samples=None,
    num_rendo_samples=None,
    rho = 0.75,
    lamb = 0.1,
    create_processed_meshes=False,
    store_processed_meshes=False,
):
    # TODO: save npz files instead of npy arrays, to more clearly indicate from which surface SDF are from instead of picking a convention
    # this would also mean modifying dataloader and training step !
    """
    Save .npy files for each patient in `source_dir`, containing sampled points
    and signed distance functions (SDFs) from specified cardiac regions.

    Each saved array has shape at most (N, 6), where:

        [:, :3]  : 3D coordinates
        [:, 3]   : SDF from epicardium surface (if requested)
        [:, 4]   : SDF from left endocardium (if requested)
        [:, 5]   : SDF from right endocardium (if requested)

    Only the requested SDF columns are included. The column order always follows
    the sequence: epicardium → left endocardium → right endocardium. 

    Args:
        `source_dir` (`str`): Directory containing patient folders with surface meshes. 
            See Notes for expected formats.
        `save_to_dir` (`str`): Directory in which to save the generated `.npy` files.
        `reference_patient` (`str`): Patient folder name used as reference for alignment if needed.
        `num_epicardium_samples` (`int | None`): Number of points to sample from the epicardium surface.
        `num_left_endocardium_samples` (`int | None`): Number of points to sample from the left endocardium surface.
        `num_right_endocardium_samples` (`int | None`): Number of points to sample from the right endocardium surface.
        `create_processed_meshes` (`bool`): If True, extracts and processes meshes from `.vtu` files at runtime.
        `store_processed_meshes` (`bool`): If True, saves the processed meshes alongside original files.

    Notes:
        - If `create_processed_meshes` is False, `source_dir` is expected to contain surface meshes in `<region>-processed.vtp` format,
          already scaled and aligned consistently.
          No geometry trasformations are performed, the meshes are only flagged non-destructively if not watertight before SDF computation.
        - If `create_processed_meshes` is True, `source_dir` must contain each patient's `.vtu` mesh.
          The required surfaces will be extracted and processed at runtime, and alignment information (rotation matrix + traslation vector)
          will be stored as data of the original mesh using `field_data` method of `pyvista.UnstructuredGrid`. 
        - Processed meshes can be optionally saved using `store_processed_meshes=True`.
    """

    source_dir = Path(source_dir)
    save_to_dir = Path(save_to_dir)

    # ------------------------------------------
    # Sampling settings
    # ------------------------------------------
    if (
        num_epi_samples is None
        and num_lendo_samples is None
        and num_rendo_samples is None
    ):
        raise ValueError("Number of samples not specified for any region.")

    num_samp_per_scene = 0
    opt = ""

    if num_epi_samples is not None:
        num_samp_per_scene += num_epi_samples
        opt += "epi_"

    if num_lendo_samples is not None:
        num_samp_per_scene += num_lendo_samples
        opt += "la_"

    if num_rendo_samples is not None:
        num_samp_per_scene += num_rendo_samples
        opt += "ra_"

    # ------------------------------------------
    # Load reference mesh if needed for processed surfaces extraction
    # ------------------------------------------
    if create_processed_meshes:
        ref_dir = source_dir / reference_patient
        mesh_path = next(ref_dir.glob("*.vtu"), None)
        if mesh_path is None:
            raise FileNotFoundError(
                f"No volumetric '.vtu' mesh found for reference patient '{reference_patient}' in {ref_dir}."
            )
        reference_mesh = pv.read(mesh_path)
        logger.warning(
            "Requested creation of processed surfaces before sampling and SDF computation. "
            "This will extract, scale, and align all patients; alignment data will be stored "
            "in each patient's original mesh file as field_data attributes."
        )
    else:
        logger.warning(f"Using already processed meshes from directory {source_dir}.")

    # ------------------------------------------
    # Iterate over all patients
    # ------------------------------------------
    source_dirs = list( source_dir.iterdir() )

    for idx, patient_dir in enumerate(source_dirs):

        if not patient_dir.is_dir():
            continue

        patient_name = patient_dir.name
        logger.warning(f"Processing patient {patient_name}: {idx + 1} / {len(source_dirs)}.")

        # Output filename
        save_to_dir.mkdir(exist_ok=True)
        out_name = f"{patient_name}-{opt}{num_samp_per_scene}_coords_and_sdf.npy"
        out_path = save_to_dir / out_name

        if out_path.is_file():
            logger.warning(
                f"Data file with requested creation modality already exists for patient {patient_name}, skipping."
            )
            continue

        files = list(patient_dir.iterdir())

        epicardium = None
        LA_endo = None
        RA_endo = None

        # Extract surfaces if requested
        extracted = (
            extract_processed_atria_surfaces(patient_name, reference_mesh, source_dir)
            if create_processed_meshes
            else None
        )

        # -----------------------------
        # Load or extract epicardium
        # -----------------------------
        if num_epi_samples is not None:
            epi_file = next(
                (f for f in files if f.is_file() and "epicardium-processed.vtp" in f.name),
                None,
            )
            if epi_file and not create_processed_meshes:
                epicardium = pv.read(epi_file)
            elif extracted is not None:
                epicardium = extracted["epicardium_surface"]
            else:
                raise ValueError(
                    "Epicardium samples requested, but no `epicardium-processed.vtp` file found and "
                    "`create_processed_meshes=False`."
                )

        # -----------------------------
        # Load or extract LA endocardium
        # -----------------------------
        if num_lendo_samples is not None:
            la_file = next(
                (f for f in files if f.is_file() and "la_endo-processed.vtp" in f.name),
                None,
            )
            if la_file and not create_processed_meshes:
                LA_endo = pv.read(la_file)
            elif extracted is not None:
                LA_endo = extracted["LA_endo_surface"]
            else:
                raise ValueError(
                    "Left-endocardium samples requested, but no `la_endo-processed.vtp` file found and "
                    "`create_processed_meshes=False`."
                )

        # -----------------------------
        # Load or extract RA endocardium
        # -----------------------------
        if num_rendo_samples is not None:
            ra_file = next(
                (f for f in files if f.is_file() and "ra_endo-processed.vtp" in f.name),
                None,
            )
            if ra_file and not create_processed_meshes:
                RA_endo = pv.read(ra_file)
            elif extracted is not None:
                RA_endo = extracted["RA_endo_surface"]
            else:
                raise ValueError(
                    "Right-endocardium samples requested, but no `ra_endo-processed.vtp` file found and "
                    "`create_processed_meshes=False`."
                )

        # ------------------------------------------
        # Save processed meshes if requested
        # ------------------------------------------
        if create_processed_meshes and store_processed_meshes:
            logger.info("Saving processed surfaces meshes")
            if epicardium is not None:
                epicardium.save(patient_dir / "epicardium-processed.vtp")
            if LA_endo is not None:
                LA_endo.save(patient_dir / "la_endo-processed.vtp")
            if RA_endo is not None:
                RA_endo.save(patient_dir / "ra_endo-processed.vtp")


        # ------------------------------------------
        # Sample surfaces
        # ------------------------------------------
        logger.info("Sampling surfaces ...")
        query_sets = []

        if epicardium is not None:
            query_sets.append(
                sample_surface_for_deepsdf(
                    epicardium.copy(),
                    number_of_points=num_epi_samples,
                    use_deepsdf_convention=True,
                    rho=rho,
                    lamb=lamb,
                    ratio=48 / 50,
                )
            )

        if LA_endo is not None:
            query_sets.append(
                sample_surface_for_deepsdf(
                    LA_endo.copy(),
                    number_of_points=num_lendo_samples,
                    use_deepsdf_convention=True,
                    rho=rho,
                    lamb=lamb,
                    ratio=48 / 50,
                )
            )

        if RA_endo is not None:
            query_sets.append(
                sample_surface_for_deepsdf(
                    RA_endo.copy(),
                    number_of_points=num_rendo_samples,
                    use_deepsdf_convention=True,
                    rho=rho,
                    lamb=lamb,
                    ratio=48 / 50,
                )
            )

        query_points = np.concatenate(query_sets).astype(np.float32)

        # ------------------------------------------
        # Compute SDFs
        # ------------------------------------------
        logger.info("Computing SDF ...")
        sdfs = []

        if epicardium is not None:
            sdfs.append(
                compute_signed_distance_libigl(mesh=epicardium, query_points=query_points)
                # compute_signed_distance_o3d(mesh=epicardium, query_points=query_points)
            )
        if LA_endo is not None:
            sdfs.append(
                compute_signed_distance_libigl(mesh=LA_endo, query_points=query_points)
                #compute_signed_distance_o3d(mesh=LA_endo, query_points=query_points)
            )
        if RA_endo is not None:
            sdfs.append(
                compute_signed_distance_libigl(mesh=RA_endo, query_points=query_points)
                #compute_signed_distance_o3d(mesh=RA_endo, query_points=query_points)
            )

        sdfs = np.stack(sdfs, axis=1).astype(np.float32)

        points = pv.PolyData(query_points)
        points["sdf_epi"] = sdfs[:,0]
        points["sdf_la"] = sdfs[:,1]
        points["sdf_ra"] = sdfs[:,2]
        
        plotter = pv.Plotter()
        plotter.add_mesh(points, scalars="sdf_epi", cmap = "jet_r", render_points_as_spheres=True)
        plotter.show()
        plotter = pv.Plotter()
        plotter.add_mesh(points, scalars="sdf_la", cmap = "jet_r", render_points_as_spheres=True)
        plotter.show()
        plotter = pv.Plotter()
        plotter.add_mesh(points, scalars="sdf_ra", cmap = "jet_r", render_points_as_spheres=True)
        plotter.show()

        # ------------------------------------------
        # Save final data
        # ------------------------------------------
        dat = np.hstack([query_points, sdfs]).astype(np.float32)
        np.save(out_path, dat, allow_pickle=False)
        logger.info("Saved coords and sdfs.")

        gc.collect()

        break

    logger.info(" Done. ")



if __name__ == "__main__":

    pass

    #TODO: example usage

    num_epi_samples = 30000
    num_lendo_samples = 35000
    num_rendo_samples = 35000
    num = num_epi_samples + num_lendo_samples + num_rendo_samples

    _create_deepsdf_data_npy(
        source_dir=PATIENT_MESHES_DIR,
        save_to_dir= PATIENTS_COORDS_AND_SDFS_DIR / f"single_patients_{num}pts_npy",
        num_epi_samples=num_epi_samples,
        num_lendo_samples=num_lendo_samples,
        num_rendo_samples=num_rendo_samples,
        create_processed_meshes=False,
        store_processed_meshes=False
    )