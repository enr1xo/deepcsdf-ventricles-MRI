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
import time

from utils.surface_utils import (
    check_watertight,
    scale_to_unit_sphere,
    sample_surface_for_deepsdf,
    compute_signed_distance_libigl
)

from config import (
    PATIENT_MESHES_DIR, 
    VENTRICLE_TAGS_METADATA,
    DATA_DIR,
    PATIENTS_NPY_DATA_DIR,
    PATIENTS_COORDS_AND_SDFS_DIR,
)

from pathlib import Path

def resolve_patient_dir(source_dir, patient_name: str) -> Path:
    """
    Ritorna la cartella che contiene davvero i dati del paziente.
    Supporta sia:
      source_dir/AF001/*.vtu
    che:
      source_dir/AF001/AF001/*.vtu
    """
    base = Path(source_dir) / patient_name
    nested = base / patient_name

    if nested.exists() and nested.is_dir():
        return nested
    return base

def find_existing_vtu(patient_root: Path) -> Path | None:
    """Trova un .vtu nel ramo del paziente (evita *_processed*)."""
    vtus = [p for p in patient_root.rglob("*.vtu") if "processed" not in p.stem.lower()]
    if not vtus:
        return None
    # preferisci un vtu che si chiama come la cartella paziente
    preferred = [p for p in vtus if p.stem.lower() == patient_root.name.lower()]
    return preferred[0] if preferred else vtus[0]


def ensure_vtu_exists(patient_root: Path, out_dir: Path | None = None) -> Path:
    """
    Cerca un .vtu nel ramo del paziente; se non c'è, lo crea da CARP bin o dal VTK se questo c'è.
    Auto-detect del base: prende il primo .elem trovato (es. vol_fib.elem -> base vol_fib).
    """
    patient_root = Path(patient_root)        

    out_dir = Path(out_dir) if out_dir is not None else patient_root

    # 1) se già esiste, ok
    existing = find_existing_vtu(patient_root)
    if existing is not None:
        return existing

    # 1.1) se c'è il .vtk, cra il vtu da questo
    vtk_files = [p for p in patient_root.rglob("*.vtk") if "processed" not in p.stem.lower()]
    # debug
    l = [p for p in patient_root.rglob("*.vtk")]
    print("PATIETTNT ROOT", l)
    print(vtk_files)
    print(patient_root)
    #

    if vtk_files:
        vtk_path = vtk_files[0]
        mesh = pv.read(vtk_path)

        if not isinstance(mesh, pv.UnstructuredGrid):
            raise TypeError(f"{vtk_path} non è voluemtrico (serve unstructuregrid).")
        
        out_dir.mkdir(parents=True, exist_ok=True)
        out_vtu = (out_dir / patient_root.name).with_suffix(".vtu")

        logger.warning(f"VTU non trovato per {patient_root.name}")
        mesh.save(out_vtu)
        return out_vtu


    # 2) trova un .elem (CARP bin) ricorsivamente
    elem_files = list(patient_root.rglob("*.elem"))
    if not elem_files:
        raise FileNotFoundError(f"Nessun file .elem trovato sotto {patient_root} (non posso creare il .vtu).")

    elem = elem_files[0]
    base_path = elem.with_suffix("")  # toglie .elem -> base

    # 3) crea .vtu (salvalo come <patient>.vtu nella cartella out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / patient_root.name  # senza estensione; meshtool aggiunge .vtu
    out_vtu = out_base.with_suffix(".vtu")

    command = [
        "meshtool", "convert",
        "-imsh", str(base_path),
        "-ifmt", "carp_bin",
        "-omsh", str(out_base),
        "-ofmt", "vtu"
    ]

    logger.warning(f"[VTU] Non trovato per {patient_root.name}. Creo VTU da: {base_path}")
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error("meshtool convert failed:")
        logger.error(e.stderr)
        raise

    # 4) verifica che ora esista
    if not out_vtu.exists():
        # fallback: cerca di nuovo (nel caso meshtool abbia creato con un nome diverso)
        created = find_existing_vtu(patient_root)
        if created is None:
            raise FileNotFoundError(f"Conversione eseguita ma nessun .vtu trovato/creato sotto {patient_root}.")
        return created

    return out_vtu

# per creare file VTU a partire da mesh generate con CARP
def create_vtu_from_carpbin(
        source_data_dir,
        save_data_dir=None,
        base_source_files_name="vol_gen.tagged.quality_rdx_fib",
        out_mesh_name=None,
        input_format="carp_bin"
    ):

    source_data_dir = Path(source_data_dir)

    if save_data_dir is None:
        save_data_dir = source_data_dir
    save_data_dir = Path(save_data_dir)

    if out_mesh_name is None:
        out_mesh_name = base_source_files_name

    # Se esiste già un .vtu da qualche parte nel ramo, non rifare
    if next(source_data_dir.rglob("*.vtu"), None) is not None:
        logger.warning(f".vtu già presente in {source_data_dir}, skip conversion.")
        return

    # Trova la base carp bin (es: .../vol_gen.tagged.quality_rdx_fib)
    # Cerchiamo il file .elem come prova (meshtool usa la base senza estensione)
    elem = next(source_data_dir.rglob(base_source_files_name + ".elem"), None)
    if elem is None:
        raise FileNotFoundError(
            f"Non trovo '{base_source_files_name}.elem' in {source_data_dir} (ricorsivo). "
            f"Controlla base_source_files_name."
        )

    base_path = elem.with_suffix("")  # toglie .elem -> base
    out_path = save_data_dir / (out_mesh_name + ".vtu")

    command = [
        "meshtool", "convert",
        "-imsh", str(base_path),
        "-ifmt", input_format,
        "-omsh", str(out_path.with_suffix("")),
        "-ofmt", "vtu"
    ]

    logger.info(f"Creating VTU: {out_path}")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return


# non servirà, da rifare con i nuovi dati
def split_cell_data_tags( mesh_tags, tags_metadata = VENTRICLE_TAGS_METADATA):
    """
        Split the tags found in all_tags into tags for right/left epi/endo ventricle.
    """
    tags_split = {}

    for key in ["RV_TAGS", "LV_TAGS", "RV_ENDO_TAGS", "LV_ENDO_TAGS", "RV_EPI_TAGS", "LV_EPI_TAGS"]:
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

def make_surface_watertight(surface_mesh: pv.PolyData):
    """
        Closes surface, additionally stores cell_data attribute 'isholepatch' indicating if cells are original or added to close holes.
    """

    # initial sanity check
    surface_mesh = surface_mesh.triangulate() # make sure is all triangular mesh
    surface_mesh = surface_mesh.clean(
        tolerance=1e-12,     
        inplace=False,
    )

    vertices = surface_mesh.points
    faces = surface_mesh.faces.reshape((-1, 4))[:, 1:4]

    orig_tri_count = faces.shape[0]

    mf = pymeshfix.MeshFix(vertices, faces)

    mf.repair() # this also close holes: faces after repair   = [ new patch faces | original faces ] appends new faces at the beginning !!
    
    vertices_repaired, faces_repaired = mf.points, mf.faces
    is_holepatch = np.zeros(faces_repaired.shape[0], dtype=np.int8)
    is_holepatch[:-orig_tri_count] = 1

    faces_pv = np.hstack([np.full((faces_repaired.shape[0], 1), 3), faces_repaired]).astype(np.int64)
    faces_repaired = faces_pv.ravel()
    surface_closed = pv.PolyData(vertices_repaired, faces_repaired)
    surface_closed.cell_data["isholepatch"] = is_holepatch

    return surface_closed

def extract_raw_ventricle_surfaces(mesh, tags_metadata = VENTRICLE_TAGS_METADATA):

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

    RV_endo_tags = split_tags["RV_ENDO_TAGS"]

    LV_endo_tags = split_tags["LV_ENDO_TAGS"]

    RV_epi_tags = split_tags["RV_EPI_TAGS"]

    LV_epi_tags = split_tags["LV_EPI_TAGS"]

    # # =============================================================== #
    # #  Extract surfaces
    # # =============================================================== #
    whole_surface = mesh.extract_surface()
    surface_elemTags = whole_surface.cell_data[elemtagskey] 

    # ===== epicardium ===== #
    # epicardium: 97 is tag for "RA_FO", which is part of endocardium also but it fills a hole in epi connecting it to the left epicardium sometimes
    # if I don't put it may happen I extract two disconnected components for some particular patients [...]
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, RV_epi_tags + LV_epi_tags + [97]) )
    surf = surf.connectivity(extraction_mode = 'largest') 
    epicardium_surface = surf.extract_surface()

    # ===== right venticle endocardium ===== #
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, RV_endo_tags) )
    surf = surf.connectivity(extraction_mode = 'largest')
    RV_endo_surface = surf.extract_surface()

    # ===== left ventricle endocardium ===== #
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, LV_endo_tags) )
    surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
    LV_endo_surface = surf.extract_surface()

    return epicardium_surface, RV_endo_surface, LV_endo_surface

def extract_closed_ventricle_surfaces(mesh, tags_metadata = VENTRICLE_TAGS_METADATA):
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

    RV_endo_tags = split_tags["RV_ENDO_TAGS"]

    LV_endo_tags = split_tags["LV_ENDO_TAGS"]

    RV_epi_tags = split_tags["RV_EPI_TAGS"]

    LV_epi_tags = split_tags["LV_EPI_TAGS"]

    # # =============================================================== #
    # #  Extract surfaces
    # # =============================================================== #
    whole_surface = mesh.extract_surface()
    surface_elemTags = whole_surface.cell_data[elemtagskey] 

    # ===== epicardium ===== #
    # epicardium: 97 is tag for "RA_FO", which is part of endocardium also but it fills a hole in epi connecting it to the left epicardium sometimes
    # if I don't put it may happen I extract two disconnected components for some particular patients [...]
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, RV_epi_tags + LV_epi_tags + [97]) )
    surf = surf.connectivity(extraction_mode = 'largest') 
    surf = surf.extract_surface()
    epicardium_surface = make_surface_watertight(surf)
    if not check_watertight(epicardium_surface):
        logger.error(f"Mesh is not watertight (found boundary edges)")

    patches = epicardium_surface.extract_cells( epicardium_surface.cell_data["isholepatch"] == 1)

    # ===== left ventricle endocardium ===== #
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, LV_endo_tags) )
    surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
    surf = surf.extract_surface()
    merged = surf.merge(
        patches,
        merge_points=True,
        tolerance=1e-4 * np.mean(surf.length)
    )
    merged = merged.connectivity(extraction_mode = 'largest')
    LV_endo_surface = merged.extract_surface()
    LV_endo_surface = make_surface_watertight(LV_endo_surface) # to close small imperfections that may remain
    if not check_watertight(LV_endo_surface):
        logger.error(f"Mesh is not watertight (found boundary edges)")

    # ===== right ventricle endocardium ===== #
    surf = whole_surface.extract_cells( np.isin( surface_elemTags, RV_endo_tags) )
    surf = surf.connectivity(extraction_mode = 'largest') # be sure is one connected component only, still has elemTags cell data
    surf = surf.extract_surface()
    merged = surf.merge(
        patches,
        merge_points=True,
        tolerance=1e-4 * np.mean(surf.length)
    )
    merged = merged.connectivity(extraction_mode = 'largest')
    RV_endo_surface = merged.extract_surface()
    RV_endo_surface = make_surface_watertight(RV_endo_surface) # to close small imperfections that may remain
    if not check_watertight(RV_endo_surface):
        logger.error(f"Mesh is not watertight (found boundary edges)")

    # clear field data that has been copied from patches or some original mesh stuff ...
    for m in (epicardium_surface, LV_endo_surface, RV_endo_surface):
        m.field_data.clear()

    return {"epicardium_surface" : epicardium_surface, "LV_endo_surface" : LV_endo_surface, "RV_endo_surface" : RV_endo_surface}

def extract_processed_ventricle_surfaces(patient_name, reference_name, reference_mesh, source_dir = PATIENT_MESHES_DIR):
    """
        Extracts epicardium, left and right endocardium surfaces from the original volumetric mesh,
        then aligns them to the reference mesh, storing alignment data in the original mesh .vtu file, and returns
        the scaled, closed, and aligned surfaces.
    """

    patient_dir = resolve_patient_dir(source_dir, patient_name)

    # VOLUMETRIC MESH: trova o crea il .vtu
    patient_root = Path(source_dir) / patient_name
    original_mesh_path = ensure_vtu_exists(patient_root)

    logger.info(f"Loading volumetric mesh: {original_mesh_path}")
    original_mesh = pv.read(original_mesh_path)


    # extract CLOSED surfaces, from ORIGINAL volumetric mesh to avoid extracting from manipulated vertices
    extracted_surfaces = extract_closed_ventricle_surfaces(original_mesh) # from here I am SURE meshes are watertight. But then I am gonna scale and move them --> numerically this can introduce some errors
    # retrieve extracted closed surfaces
    LV_endo = extracted_surfaces["LV_endo_surface"]
    RV_endo = extracted_surfaces["RV_endo_surface"]
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
    LV_endo.points -= centre
    LV_endo.points /= max_radius
    RV_endo.points -= centre
    RV_endo.points /= max_radius

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
        LV_endo.points = apply_icp_result(R, t, LV_endo.points)
        RV_endo.points = apply_icp_result(R, t, RV_endo.points)

    # store scaling and alignment data in original mesh file 
    original_mesh.field_data[f'alignto{reference_name}-rotation'] = R
    original_mesh.field_data[f'alignto{reference_name}-traslation'] = t
    original_mesh.field_data['centre-centroid'] = centre 
    original_mesh.field_data['scale-tounitradius'] = max_radius 

    # store scaling factor also in every surface data to find it easily later to get back to original scale, without having to load the original mesh
    epicardium.field_data['scale-tooriginalrange'] = max_radius 
    LV_endo.field_data['scale-tooriginalrange'] = max_radius 
    RV_endo.field_data['scale-tooriginalrange'] = max_radius   

    # sanity check: make sure numerical transformations haven't broken watertightness
    if not check_watertight(epicardium):
        logger.error("Epicardium mesh isn't watertight: found boundary edges")

    if not check_watertight(LV_endo):
        logger.error("LV endocardium mesh isn't watertight: found boundary edges")
    
    if not check_watertight(RV_endo):
        logger.error("RV endocardium isn't watertight: found boundary edges")

    ### to go back precisely from original --> extracted in the ORIGINAL SCALE:
    # original_mesh.points -= centre
    # original_mesh.points /= max_radius
    # original_mesh.points = apply_icp_result(R,t,original_mesh.points)
    # original_mesh.points *= epicardium.field_data["scale-tooriginalrange"]

    return {"epicardium_surface" : epicardium, "LV_endo_surface" : LV_endo, "RV_endo_surface" : RV_endo, "original_mesh" : original_mesh}

def _create_processed_surfaces_meshes(
    source_dir = PATIENT_MESHES_DIR,
    save_to_dir = PATIENT_MESHES_DIR,
    reference_patient = "AF001"):
    """
        Helper to only create all processed meshes first
    """

    patients = [f.name for f in source_dir.iterdir() if f.is_dir() and "single_patients_100000pts" not in f.name]

    ref_dir = resolve_patient_dir(source_dir, reference_patient)
    # ref_candidates = list(ref_dir.rglob("*.vtu"))
    ref_candidates = [
                        p for p in ref_dir.rglob("*.vtu")
                        if "single_patients_100000pts" not in str(p)
                    ]

    ref_preferred = [p for p in ref_candidates if p.stem.lower() == reference_patient.lower()]
    ref_path = ref_preferred[0] if ref_preferred else ref_candidates[0]
    reference_mesh = pv.read(ref_path)
    

    for patient in patients:

        logger.warning(f"Processing patient {patient}")

        patient_save_dir = Path(save_to_dir) / patient

        # if next( patient_save_dir.rglob("*-processed.vtp"), None) is not None:
        #     logger.warning(f"Processed meshes already present at {str(patient_save_dir)}, skipping creation")
        #     continue
        
        Path( patient_save_dir ).mkdir( exist_ok=True)
        
        extracted = extract_processed_ventricle_surfaces(patient, reference_patient, reference_mesh, source_dir)

        LV_endo = extracted["LV_endo_surface"]
        RV_endo = extracted["RV_endo_surface"]
        epicardium = extracted["epicardium_surface"]
        original = extracted["original_mesh"]

        logger.info("Saving processed surfaces meshes")
        
        epicardium.save( patient_save_dir / "epicardium-processed.vtp")
        LV_endo.save( patient_save_dir / "lv_endo-processed.vtp")
        RV_endo.save( patient_save_dir / "rv_endo-processed.vtp")

        original.save( patient_save_dir / f"{patient}.vtu")

    return

# prende un vtu
def _create_deepsdf_data_npy(
    source_dir,
    save_to_dir,
    reference_patient = "",
    num_epi_samples=None,
    num_lendo_samples=None,
    num_rendo_samples=None,
    sigma = 0.025,
    lamb = 0.1,
    rho = 0.75,
    create_processed_meshes=True,
    store_processed_meshes=True,
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
        opt += "lv_"

    if num_rendo_samples is not None:
        num_samp_per_scene += num_rendo_samples
        opt += "rv_"

    # ------------------------------------------
    # Load reference mesh if needed for processed surfaces extraction
    # ------------------------------------------
    if create_processed_meshes:
        ref_dir = source_dir / reference_patient
        print(f"reference directory: {ref_dir}") #############################################################################################################
        ref_vtu_path = ensure_vtu_exists(ref_dir)
        reference_mesh = pv.read(ref_vtu_path)

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

         # DEBUG
        if patient_name == "AF070" or patient_name == "S66":
            continue

        if patient_name == "single_patients_100000pts_npy" or "single_patients_100000pts" in patient_name:
            continue


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

        real_patient_dir = resolve_patient_dir(source_dir, patient_name)
        files = list(real_patient_dir.rglob("*"))  # così trova anche i .vtp dentro AF001/AF001/


        epicardium = None
        LV_endo = None
        RV_endo = None

        # Extract surfaces if requested
        reference_name= "AF001"
        extracted = (
            extract_processed_ventricle_surfaces(patient_name,reference_name,reference_mesh, source_dir)
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
        # Load or extract LV endocardium
        # -----------------------------
        if num_lendo_samples is not None:
            lv_file = next(
                (f for f in files if f.is_file() and "lv_endo-processed.vtp" in f.name),
                None,
            )
            if lv_file and not create_processed_meshes:
                LV_endo = pv.read(lv_file)
            elif extracted is not None:
                LV_endo = extracted["LV_endo_surface"]
            else:
                raise ValueError(
                    "Left-endocardium samples requested, but no `lv_endo-processed.vtp` file found and "
                    "`create_processed_meshes=False`."
                )

        # -----------------------------
        # Load or extract RV endocardium
        # -----------------------------
        if num_rendo_samples is not None:
            rv_file = next(
                (f for f in files if f.is_file() and "rv_endo-processed.vtp" in f.name),
                None,
            )
            if rv_file and not create_processed_meshes:
                RV_endo = pv.read(rv_file)
            elif extracted is not None:
                RV_endo = extracted["RV_endo_surface"]
            else:
                raise ValueError(
                    "Right-endocardium samples requested, but no `rv_endo-processed.vtp` file found and "
                    "`create_processed_meshes=False`."
                )

        # ------------------------------------------
        # Save processed meshes if requested
        # ------------------------------------------
        if create_processed_meshes and store_processed_meshes:
            logger.info("Saving processed surfaces meshes")
            if epicardium is not None:
                epicardium.save(real_patient_dir / "epicardium-processed.vtp")
            if LV_endo is not None:
                LV_endo.save(real_patient_dir / "lv_endo-processed.vtp")
            if RV_endo is not None:
                RV_endo.save(real_patient_dir / "rv_endo-processed.vtp")


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
                    sigma=sigma,
                    rho=rho,
                    lamb=lamb,
                    ratio=48 / 50,
                )
            )

        if LV_endo is not None:
            query_sets.append(
                sample_surface_for_deepsdf(
                    LV_endo.copy(),
                    number_of_points=num_lendo_samples,
                    use_deepsdf_convention=True,
                    sigma=sigma,
                    rho=rho,
                    lamb=lamb,
                    ratio=48 / 50,
                )
            )

        if RV_endo is not None:
            query_sets.append(
                sample_surface_for_deepsdf(
                    RV_endo.copy(),
                    number_of_points=num_rendo_samples,
                    use_deepsdf_convention=True,
                    sigma=sigma,
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
        if LV_endo is not None:
            sdfs.append(
                compute_signed_distance_libigl(mesh=LV_endo, query_points=query_points)
                #compute_signed_distance_o3d(mesh=LA_endo, query_points=query_points)
            )
        if RV_endo is not None:
            sdfs.append(
                compute_signed_distance_libigl(mesh=RV_endo, query_points=query_points)
                #compute_signed_distance_o3d(mesh=RA_endo, query_points=query_points)
            )

        sdfs = np.stack(sdfs, axis=1).astype(np.float32)

        # points = pv.PolyData(query_points)
        # points["sdf_epi"] = sdfs[:,0]
        # points["sdf_la"] = sdfs[:,1]
        # points["sdf_ra"] = sdfs[:,2]
        
        # plotter = pv.Plotter()
        # plotter.add_mesh(points, scalars="sdf_epi", cmap = "jet_r", render_points_as_spheres=True)
        # plotter.show()
        # plotter = pv.Plotter()
        # plotter.add_mesh(points, scalars="sdf_la", cmap = "jet_r", render_points_as_spheres=True)
        # plotter.show()
        # plotter = pv.Plotter()
        # plotter.add_mesh(points, scalars="sdf_ra", cmap = "jet_r", render_points_as_spheres=True)
        # plotter.show()

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

    # _create_processed_surfaces_meshes(
    #     source_dir  = Path("/home/navarri/AtriaProject/DATASETS/AtrialGeometriesOriginal"),
    #     save_to_dir = Path("/home/navarri/AtriaProject/DATASETS/AtrialGeometries"),
    #     reference_patient="AF069"
    # )

    sickness_list = ["AF", "SV", "2017", "VT", "LeuBBB", "LeuNORM"]
    reference_patient_list = ["AF001", "yrm0342_v1", "S62", "VT001_MUG1", "LEU_BBB_21001", "LEU_NORM_0016"]

    sickness = "AF"

    if sickness == sickness_list[0]:
        reference_patient = reference_patient_list[0]

    elif sickness == sickness_list[1]:
        reference_patient = reference_patient_list[1]
    
    elif sickness == sickness_list[2]:
        reference_patient = reference_patient_list[2]
    
    elif sickness == sickness_list[3]:
        reference_patient = reference_patient_list[3]
    
    elif sickness == sickness_list[4]:
        reference_patient = reference_patient_list[4]

    elif sickness == sickness_list[5]:
        reference_patient = reference_patient_list[5]

    # 5k
    # num_epi_samples = 1500
    # num_lendo_samples = 1750
    # num_rendo_samples = 1750

    # 3k
    # num_epi_samples = 900
    # num_lendo_samples = 1050
    # num_rendo_samples = 1050

    # samples number comparable to MRI-specific models samples count
    num_epi_samples = 5850
    num_lendo_samples = 6825 
    num_rendo_samples = 6825
    
    num = num_epi_samples + num_lendo_samples + num_rendo_samples

    # sigmas = [0.25, 0.025, 0.0025]
    # lambdas = [0.25, 0.5, 0.75]
    # rhos = [0.5, 1, 2]

    sigmas = [0.025]
    lambdas = [0.75]
    rhos = [0.5]

    combination = 1
    combs = len(sigmas) * len(lambdas) * len(rhos)

    parallel = True

    if not parallel:
        print(f"\n--- NON-PARALELL PREPROCESSING ---\n")
    else:
        print(f"\n--- PARALELL PREPROCESSING ---\n")

    for sigma in sigmas:
        for lam in lambdas:
            for rho in rhos:

                if not parallel:               
                    print(f"\nSamples number: {num}")
                    print(f"\nPreprocessing sampling combination {combination} / {combs}.")

                    save_dir = PATIENTS_COORDS_AND_SDFS_DIR / f"S_{sigma}-L_{lam}-R_{rho}"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    start_time = time.time()

                    _create_deepsdf_data_npy(
                        source_dir=PATIENT_MESHES_DIR,
                        save_to_dir= save_dir,
                        reference_patient=reference_patient,
                        num_epi_samples=num_epi_samples,
                        num_lendo_samples=num_lendo_samples,
                        num_rendo_samples=num_rendo_samples,
                        sigma=sigma,
                        lamb=lam,
                        rho=rho,
                        create_processed_meshes=False,
                        store_processed_meshes=True
                    )
                    
                    end_time = time.time()

                    elapsed = end_time - start_time

                    print(f"\nCombination S={sigma}, L={lam}, R={rho} took {elapsed:.2f} seconds ({elapsed/60:.2f} min)")

                    combination += 1

                else:
                    from Z_enricos_stuff.parallel_preprocessing import _create_deepsdf_data_npy_parallel
                    print(f"\nSamples number: {num}")
                    print(f"\nPreprocessing sampling combination {combination} / {combs}.")

                    save_dir = PATIENTS_COORDS_AND_SDFS_DIR / f"S_{sigma}-L_{lam}-R_{rho}"
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # DEBUG
                    # patient_dir = PATIENT_MESHES_DIR / "LEU_BBB_21056"
                    # fine debug

                    start_time = time.time()

                    _create_deepsdf_data_npy_parallel(
                        source_dir=PATIENT_MESHES_DIR,
                        # source_dir=patient_dir,
                        save_to_dir= save_dir,
                        reference_patient=reference_patient,
                        num_epi_samples=num_epi_samples,
                        num_lendo_samples=num_lendo_samples,
                        num_rendo_samples=num_rendo_samples,
                        sigma=sigma,
                        lamb=lam,
                        rho=rho,
                        create_processed_meshes=False,
                        store_processed_meshes=True,
                        max_workers=6
                    )
                    
                    end_time = time.time()

                    elapsed = end_time - start_time

                    print(f"\nCombination S={sigma}, L={lam}, R={rho} took {elapsed:.2f} seconds ({elapsed/60:.2f} min)")

                    combination += 1


    
    patient = reference_patient

    data = np.load(PATIENTS_NPY_DATA_DIR / f"{patient}-epi_lv_rv_{num}_coords_and_sdf.npy") 

    # coords = data[:,:3]
    # sdfs = data[:,3:]

    # points = pv.PolyData(coords)
    # points["sdf_epi"] = sdfs[:,0]
    # points["sdf_lv"] = sdfs[:,1]
    # points["sdf_rv"] = sdfs[:,2]

    # plotter = pv.Plotter()
    # plotter.add_mesh(points, scalars="sdf_epi", cmap="jet_r", render_points_as_spheres = True)
    # plotter.show()

    # plotter = pv.Plotter()
    # plotter.add_mesh(points, scalars="sdf_lv", cmap="jet_r", render_points_as_spheres = True)
    # plotter.show()

    # plotter = pv.Plotter()
    # plotter.add_mesh(points, scalars="sdf_rv", cmap="jet_r", render_points_as_spheres = True)
    # plotter.show()


        

