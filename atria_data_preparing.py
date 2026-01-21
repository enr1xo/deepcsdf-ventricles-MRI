from pathlib import Path
import numpy as np
from loguru import logger
import pyvista as pv
import subprocess
import os
import gc
import json
from utils.align_atrial_mesh import apply_icp_result, align_to_reference_mesh
from scipy.spatial import KDTree
import pymeshfix
from utils.surface_utils import (
    check_watertight,
    scale_to_unit_sphere, make_trimesh_from_pv,
    sample_surface_for_deepsdf,
    compute_signed_distance_o3d,
)

from config import (
    PATIENT_MESHES_DIR, 
    ATRIA_TAGS_METADATA,
    DATA_DIR,
    PATIENTS_NPY_DATA_DIR,
    TRAIN_DATA_DIR,
    TEST_DATA_DIR
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
        logger.warning("File .vtu already present in source directory, skipping creation.")
    
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

def propagate_surface_cell_data_tags(original_surface, target_surface, elemtagskey):

    orig_tags = original_surface.cell_data[elemtagskey]
    
    orig_centers = original_surface.cell_centers().points
    
    closed_centers = target_surface.cell_centers().points
    
    tree = KDTree(orig_centers)

    new_tags = np.empty(target_surface.n_cells, dtype=orig_tags.dtype)

    for i, p in enumerate(closed_centers):
        _, idx = tree.query(p)
        new_tags[i] = orig_tags[idx]

    target_surface.cell_data[elemtagskey] = new_tags

    return target_surface

def make_surface_with_tags_watertight(surface_mesh: pv.PolyData, elemtagskey):
    """
        Closes surface, and keeps cell_data 'elemtagskey' attribute herediting from original surface's elements tags,
        additionally stores cell_data attribute 'isholepatch' indicating if cells are original or added to close holes.
    """
    
    clean_mesh = surface_mesh.clean(
        tolerance=0.0,     # merge points at exactly the same coordinates
        inplace=False
    )
    
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

    # assign original tags back to the surface
    clean_mesh = propagate_surface_cell_data_tags(surface_mesh, clean_mesh, elemtagskey=elemtagskey)

    mf = pymeshfix.MeshFix(vertices, faces)

    mf.repair() # this also close holes
    
    vertices_fixed, faces_fixed = mf.v, mf.f
    faces_pv = np.hstack([np.full((faces_fixed.shape[0], 1), 3), faces_fixed]).astype(np.int64)
    faces_pv = faces_pv.ravel()

    surface_closed = pv.PolyData(vertices_fixed, faces_pv)

    if not check_watertight(surface_closed):
        logger.warning(f"Mesh is not watertight (found boundary edges)")

    # mark patches holes cells
    edge_lengths = np.linalg.norm(
        vertices[faces[:, 0]] - vertices[faces[:, 1]],
        axis=1
    )
    tol = 1e-3 * np.mean(edge_lengths) # to scale with actual mesh size
    # orig_face_sets = {} # to lookup faces, stored as tuples of actual vertices coordinates
    # for face_id, tri in enumerate(faces):  # faces = original triangles
    #     # # ====== ❗ Why this is broken
    #     # # You are:
    #     # # comparing floating-point vertex coordinates
    #     # # after MeshFix + clean + normals
    #     # # expecting bitwise equality
    #     # # This cannot be made reliable.
    #     # key = frozenset(tuple(vertices[v]) for v in tri) # ---> This line relies on exact float equality and will always break eventually
    #     # SOLUTION: "quantize" the vertex key
    #     key = frozenset( tuple(np.round(vertices[v] / tol).astype(np.int64)) for v in tri) 
    #     orig_face_sets[key] = clean_mesh.cell_data[elemtagskey][face_id]
    orig_face_sets = {}
    for face_id, tri in enumerate(faces):
        key = frozenset(
            tuple(np.round(vertices[v] / tol).astype(np.int64))
            for v in tri
        )
        orig_face_sets[key] = clean_mesh.cell_data[elemtagskey][face_id]

    closed_face_sets = [
        frozenset(
            tuple(np.round(vertices_fixed[v] / tol).astype(np.int64))
            for v in tri
        )
        for tri in faces_fixed
    ]# to iterate over: faces of original mesh + closed holes

    all_tags = np.full(len(faces_fixed), -1, dtype=clean_mesh.cell_data[elemtagskey].dtype) # original tags, default is -1 for new cells

    isholepatch = np.ones(len(faces_fixed), dtype=np.int8)
    for i, face_key in enumerate(closed_face_sets):
        if face_key in orig_face_sets: 
            isholepatch[i] = 0 
            all_tags[i] = orig_face_sets[face_key]   # assign also original element tag

    surface_closed.cell_data["isholepatch"] = isholepatch # true (not 0) ==> not part of original mesh, so is part of a hole patch
    surface_closed.cell_data[elemtagskey] = all_tags # original tags, default is -1 for new cells

    # now I just need to assign a meaningful cell_data[elemtagskey] to new cells
    patch_indices = np.where( surface_closed.cell_data["isholepatch"] )[0] # patches' cells indices in the full mesh 
    patches = surface_closed.extract_cells(patch_indices) # all patches
    patches_split = patches.connectivity() # separate into connected components
    region_ids = np.unique(patches_split.cell_data["RegionId"]) # to iterate easily over the connected components

    orig_faces = surface_closed.faces.reshape(-1, 4)[surface_closed.cell_data["isholepatch"] == 0, 1:4] # faces arrays NOT part of any patch
    orig_cells_indices = np.where(surface_closed.cell_data["isholepatch"] == 0)[0] # indices of those cells in the full mesh

    tags = surface_closed.cell_data[elemtagskey].copy()

    for rid in region_ids:
        patch_single_mask = patches_split.cell_data["RegionId"] == rid # indices of cells in the current patch

        # vertices indexes of the CURRENT patch, as cell IDs in the full mesh
        patch_cells_orig_indices = patch_indices[patch_single_mask]
        patch_vertices_idxs = np.unique(
            surface_closed.faces.reshape(-1, 4)[patch_cells_orig_indices, 1:4].flatten()
        )

        # find original faces that share any vertex with this patch only
        touch_mask = np.any(np.isin(orig_faces, patch_vertices_idxs), axis=1)
        candidate_faces_idx = np.where(touch_mask)[0]

        # assign tag from first face found
        if candidate_faces_idx.size == 0:
            # fallback if something breaks
            tag_to_assign = -1
        else:
            tag_to_assign = surface_closed.cell_data[elemtagskey][orig_cells_indices[candidate_faces_idx[0]]]

        # assign the same tag to all cells of this patch in surface_closed
        tags[patch_cells_orig_indices] = tag_to_assign

    surface_closed.cell_data[elemtagskey] = tags



    return surface_closed

def extract_raw_atria_surfaces(patient_name, source_dir, tags_metadata = ATRIA_TAGS_METADATA):

    patient_dir = source_dir / patient_name

    mesh_path = next( patient_dir.rglob("*.vtu"), None)

    if mesh_path is not None:

        logger.info("Extracting raw epicardium and left/right endocardium surfaces from original volume mesh ...")
        
        mesh = pv.read( mesh_path )

        mesh.field_data.clear()

        if hasattr(mesh, "cell_data"):
            if "elemTag" in mesh.cell_data:
                elemTags = mesh.cell_data["elemTag"]
                elemtagskey =  "elemTag"
            elif "elemTags" in mesh.cell_data:
                elemTags = mesh.cell_data["elemTags"]
                elemtagskey =  "elemTags"
            else:
                raise ValueError(f"Unknown key to access elements tags in mesh at {mesh_path}, expected elemTags or elemTag")   
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

def extract_closed_atria_surfaces(patient_name, source_dir, tags_metadata = ATRIA_TAGS_METADATA):
    """
        Extracts raw epicardium, left/right endocardium surfaces from original volumetric mesh using elements' tags,
        then closes the surfaces, returning watertight meshes.

        Doesn't perform ANY smoothing or geometric change either than closing holes.
    """
    patient_dir = source_dir / patient_name

    mesh_path = next( patient_dir.rglob("*.vtu"), None)

    if mesh_path is not None:

        logger.info("Extracting epicardium and left/right endocardium surfaces from original volume mesh ...")
        
        mesh = pv.read( mesh_path )

        if hasattr(mesh, "cell_data"):
            if "elemTag" in mesh.cell_data:
                elemTags = mesh.cell_data["elemTag"]
                elemtagskey =  "elemTag"
            elif "elemTags" in mesh.cell_data:
                elemTags = mesh.cell_data["elemTags"]
                elemtagskey =  "elemTags"
            else:
                raise ValueError(f"Unknown key to access elements tags in mesh at {mesh_path}, expected elemTags or elemTag")   
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
        surface_closed  = make_surface_with_tags_watertight(surface_mesh=surf.extract_surface(), elemtagskey=elemtagskey) 
        epicardium_surface = surface_closed.extract_surface()

        # patch LA and RA with the same epicardium patches directly:
        # naively: merge all of them, then extract larger connected component. should work, easy
        # 2. select first just the patches I want: I pick exactly the tags I need for LA / RA. clean, but I have to be sure I get all of the tags correctly
        #    some patches may close holes I don't even "know" about in the sense that they don't have a specific tag like veins and valves do
        patches = epicardium_surface.extract_cells( epicardium_surface.cell_data["isholepatch"] == 1 )

        # ===== right atrium endocardium ===== #
        surf = whole_surface.extract_cells( np.isin( surface_elemTags, RA_endo_tags) )
        surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
        surf = surf.extract_surface()
        # merge with all patches, extract largest connected component
        merged = surf.merge(patches, merge_points=True)
        merged = merged.connectivity(extraction_mode = 'largest')
        RA_endo_surface = merged.extract_surface()

        # ===== left atrium endocardium ===== #
        surf = whole_surface.extract_cells( np.isin( surface_elemTags, LA_endo_tags) )
        surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
        surf = surf.extract_surface()
        # merge with all patches, extract largest connected component
        merged = surf.merge(patches, merge_points=True)
        merged = merged.connectivity(extraction_mode = 'largest')
        LA_endo_surface = merged.extract_surface()

        # clear field data that has been copied from patches or some original mesh stuff ...
        for m in (epicardium_surface, LA_endo_surface, RA_endo_surface):
            m.field_data.clear()

        #TODO: (I don't need it for now) add isholepatch cell data also in LA and RA again. They all have element tags data. 

    return {"epicardium_surface" : epicardium_surface, "LA_endo_surface" : LA_endo_surface, "RA_endo_surface" : RA_endo_surface}

def extract_processed_atria_surfaces(patient_name, reference_name, reference_mesh, source_dir = PATIENT_MESHES_DIR):
    """
        Extracts epicardium, left and right endocardium surfaces from the original volumetric mesh,
        then aligns them to the reference mesh, storing alignment data in the original mesh .vtu file, and returns
        the scaled, closed, and aligned surfaces.
    """

    patient_dir = Path(source_dir) / patient_name

    original_mesh_path = next( patient_dir.glob("*.vtu"), None)
    if not original_mesh_path:
        raise FileNotFoundError(f"Volumetric mesh for patient {patient_name} not found in {patient_dir}.")
    original_mesh = pv.read(original_mesh_path)

    # extract CLOSED surfaces, directly from the ORIGINAL mesh, not aligned, not scaled.
    extracted_surfaces = extract_closed_atria_surfaces(patient_name, source_dir)

    # retrieve extracted closed surfaces, apply scaling and alignment
    LA_endo = extracted_surfaces["LA_endo_surface"]
    RA_endo = extracted_surfaces["RA_endo_surface"]
    epicardium = extracted_surfaces["epicardium_surface"]

    # center -> scale
    original_scaled = original_mesh.copy()
    _, centre, max_radius = scale_to_unit_sphere(original_scaled.points, return_transf_params=True)
    original_scaled.points -= centre
    original_scaled.points /= max_radius     

    # center and scale also extracted surfaces
    epicardium.points -= centre
    epicardium.points /= max_radius
    LA_endo.points -= centre
    LA_endo.points /= max_radius
    RA_endo.points -= centre
    RA_endo.points /= max_radius

    # now align: use ICP on point clouds AT THE STANDARD UNIT SCALE !! resulting rotation and traslation data will be in this scale

    if patient_name != reference_name: # skip alignment for reference mesh
        reference_mesh_scaled = reference_mesh.copy()
        _, centre_ref, max_radius_ref = scale_to_unit_sphere(reference_mesh_scaled.points, return_transf_params=True)
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

        epicardium.points = apply_icp_result(R, t, epicardium.points)
        LA_endo.points = apply_icp_result(R, t, LA_endo.points)
        RA_endo.points = apply_icp_result(R, t, RA_endo.points)
    
    if patient_name == reference_name:
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)

    # store scaling and alignment data in original mesh file 
    original_mesh.field_data[f'alignto{reference_name}-rotation'] = R
    original_mesh.field_data[f'alignto{reference_name}-traslation'] = t
    original_mesh.field_data['centre-centroid'] = centre 
    original_mesh.field_data['scale-tounitradius'] = max_radius 

    original_mesh.save(original_mesh_path)

    # store scaling factor also in every surface data to find it easily later to get back to original scale, without having to load the original mesh
    epicardium.field_data['scale-tounitradius'] = max_radius 
    LA_endo.field_data['scale-tounitradius'] = max_radius 
    RA_endo.field_data['scale-tounitradius'] = max_radius     

    return {"epicardium_surface" : epicardium, "LA_endo_surface" : LA_endo, "RA_endo_surface" : RA_endo}

def _create_deepsdf_data_npy(
    source_dir = PATIENT_MESHES_DIR,
    save_to_dir = PATIENTS_NPY_DATA_DIR,
    reference_patient = "AF069",
    num_epi_samples=None,
    num_lendo_samples=None,
    num_rendo_samples=None,
    rho = 0.75,
    lamb = 0.1,
    use_scans_for_sdf = False,
    create_processed_meshes=False,
    store_processed_meshes=False,
):
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
          already scaled and aligned consistently. No geometry checks are performed.
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
        logger.warning("Using already processed meshes.")

    # ------------------------------------------
    # Iterate over all patients
    # ------------------------------------------
    source_dirs = list( source_dir.iterdir() )

    for idx, patient_dir in enumerate(source_dirs):

        if not patient_dir.is_dir():
            continue

        patient_name = patient_dir.name
        logger.info(f"Processing patient {patient_name}: {idx + 1} / {len(source_dirs)}.")

        # Output filename
        out_name = f"{patient_name}_{opt}{num_samp_per_scene}_coords_and_sdf.npy"
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
        # Bring sufaces to standardized range as requested:
        # 1. scale all of the original meshes by the SAME value, to maintain respective dimensions
        # 2. Scale all of the original meshes by their own value, to fit them inside a unit sphere
        # --> actually always do this second thing now !!! then the various scales can be stored as field data and used optionally to later scale the data if requested
        # ------------------------------------------

        # ------------------------------------------
        # Sample surfaces
        # ------------------------------------------
        logger.info("Sampling surfaces...")
        query_sets = []

        if epicardium is not None:
            query_sets.append(
                sample_surface_for_deepsdf(
                    epicardium,
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
                    LA_endo,
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
                    RA_endo,
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

        for surface in [epicardium, LA_endo, RA_endo]:
            if surface is not None:
                sdfs.append(
                    compute_signed_distance_o3d(mesh=surface, query_points=query_points)
                )

        sdfs = np.stack(sdfs, axis=1).astype(np.float32)

        # ------------------------------------------
        # Save final data
        # ------------------------------------------
        dat = np.hstack([query_points, sdfs]).astype(np.float32)
        np.save(out_path, dat, allow_pickle=False)
        logger.info("Saved coords and sdfs.")

        gc.collect()
    
    logger.info(" Done. ")


if __name__ == "__main__":

    pass

    #TODO: example usage

    # patient_files_dir = Path("/home/davidenava_linux/AtriaProject/ProjectMLRuns/data/single_patients_vtksdf_npy")

    # create_train_test_split(patient_files_dir)

    # _create_deepsdf_data_npy(
    #     source_dir=Path("/home/davidenava_linux/DATASETS/AtrialGeometriesDataProcessed"),
    #     save_to_dir= DATA_DIR / "single_patients_100000pts_npy",
    #     num_epi_samples=30000,
    #     num_lendo_samples=35000,
    #     num_rendo_samples=35000
    # )

    from config import PATIENT_MESHES_DIR
    from time import time

    patients = [f.name for f in PATIENT_MESHES_DIR.iterdir()]
    reference_mesh = pv.read(PATIENT_MESHES_DIR / "AF069" / "AF069.vtu")

    for patient in  patients:
        patient="LEU_NORM_F004"
        patient_dir = PATIENT_MESHES_DIR / patient 
        mesh_original = pv.read( patient_dir / f"{patient}.vtu")

        print(patient)

        # extracted = extract_processed_atria_surfaces(patient, "AF069", reference_mesh, PATIENT_MESHES_DIR)

        extracted = extract_closed_atria_surfaces(patient, source_dir=PATIENT_MESHES_DIR)
        LA_endo = extracted["LA_endo_surface"]
        RA_endo = extracted["RA_endo_surface"]
        epicardium = extracted["epicardium_surface"]

        # logger.info("Saving processed surfaces meshes")
        
        # epicardium.save( patient_dir / "epicardium-processed.vtp")
        # LA_endo.save( patient_dir / "la_endo-processed.vtp")
        # RA_endo.save( patient_dir / "ra_endo-processed.vtp")

        plotter = pv.Plotter()
        plotter.add_mesh(epicardium, color = "lightgray", opacity=0.5)
        plotter.add_mesh(LA_endo, color="red", opacity=0.5)
        plotter.add_mesh(RA_endo,  color="skyblue", opacity=0.5)
        plotter.show_grid()
        plotter.show()

        break



    