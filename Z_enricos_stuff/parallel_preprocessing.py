"""
This code tryes to parallelize the preprocessing phase for the deepSDF network
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pyvista as pv
import gc
import os

from ventricles_data_preparing import resolve_patient_dir, \
        ensure_vtu_exists, \
        extract_processed_ventricle_surfaces, \
        sample_surface_for_deepsdf, \
        compute_signed_distance_libigl

#------------------------------------------------------------------------------------------------
def process_single_patient(
        patient_dir,
        source_dir,
        save_to_dir,
        reference_patient,
        num_epi_samples,
        num_lendo_samples,
        num_rendo_samples,
        sigma,
        lamb,
        rho,
        create_processed_meshes,
        store_processed_meshes,
        ):
    
    source_dir = Path(source_dir)
    save_to_dir = Path(save_to_dir)
    patient_dir = Path(patient_dir)

    patient_name = patient_dir.name

    if not patient_dir.is_dir():
        return patient_name, False, "Not a directory"
    
    # DEBUG / exclusions
    if patient_name in {"AF070", "S66"}:
        return patient_name, True, "Skipped by debug rule"

    if patient_name == "single_patients_100000pts_npy" or "single_patients_100000pts" in patient_name:
        return patient_name, True, "Skipped auxiliary folder"
    
    if (
        num_epi_samples is None
        and num_lendo_samples is None
        and num_rendo_samples is None
    ):
        return patient_name, False, "number of samples not specified"
    
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
    
    save_to_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{patient_name}-{opt}{num_samp_per_scene}_coords_and_sdf.npy"
    out_path = save_to_dir / out_name

    if out_path.is_file():
        return patient_name, True, "Already exists, skipped."
    
    try:
        real_patient_dir = resolve_patient_dir(source_dir, patient_name)
        # print("source_dir:", source_dir, type(source_dir))
        # print("patient_name:", patient_name, type(patient_name))
        # print("real_patient_dir:", real_patient_dir, type(real_patient_dir))
        files = list(real_patient_dir.rglob("*"))

        epicardium = None
        LV_endo = None
        RV_endo = None

        extracted = None
        if create_processed_meshes:
            ref_dir = source_dir / reference_patient
            ref_vtu_path = ensure_vtu_exists(ref_dir)
            reference_mesh = pv.read(ref_vtu_path)

            extracted = extract_processed_ventricle_surfaces(
                patient_name,
                reference_patient,
                reference_mesh,
                source_dir
            )

        # epi
        if num_epi_samples is not None:
            epi_file = next(
                (f for f in files if f.is_file() and "epicardium-processed.vtp" in f.name),
                None
            )

            if epi_file and not create_processed_meshes:
                epicardium = pv.read(epi_file)
            elif extracted is not None:
                epicardium = extracted["epicardium_surface"]
            else:
                raise ValueError("Epicardium samples requested, but no `epicardium-processed.vtp` file found and "
                    "`create_processed_meshes=False`.")
        
        # lv 
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

        # RV endocardium
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

        if create_processed_meshes and store_processed_meshes:
            if epicardium is not None:
                epicardium.save(real_patient_dir / "epicardium-processed.vtp")
            if LV_endo is not None:
                LV_endo.save(real_patient_dir / "lv_endo-processed.vtp")
            if RV_endo is not None:
                RV_endo.save(real_patient_dir / "rv_endo-processed.vtp")

        # sample surfaces
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
                    ratio=48/50
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
                    ratio=48/50
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
                    ratio=48/50
                )
            )
        
        query_points = np.concatenate(query_sets).astype(np.float32)

        # compute SDFs
        sdfs = []

        if epicardium is not None:
            sdfs.append(compute_signed_distance_libigl(mesh=epicardium, query_points=query_points))
        if LV_endo is not None:
            sdfs.append(compute_signed_distance_libigl(mesh=LV_endo, query_points=query_points))
        if RV_endo is not None:
            sdfs.append(compute_signed_distance_libigl(mesh=RV_endo, query_points=query_points))

        sdfs = np.stack(sdfs, axis=1).astype(np.float32)

        dat = np.hstack([query_points, sdfs]).astype(np.float32)
        np.save(out_path, dat, allow_pickle=False)

        gc.collect()
        return patient_name, True, None

    except Exception as e:
        return patient_name, False, str(e)


#----------------------------------------------------------------------------
def _create_deepsdf_data_npy_parallel(
        source_dir,
        save_to_dir,
        reference_patient="",
        num_epi_samples=None,
        num_lendo_samples=None,
        num_rendo_samples=None,
        sigma=0.025,
        lamb=0.1,
        rho=0.75,
        create_processed_meshes=True,
        store_processed_meshes=True,
        max_workers=None,
        ):
    
    source_dir = Path(source_dir)
    save_to_dir = Path(save_to_dir)
    save_to_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = [
    p for p in source_dir.iterdir()
    if p.is_dir() and (
        (p / "epicardium-processed.vtp").exists()
        or (p / "lv_endo-processed.vtp").exists()
        or (p / "rv_endo-processed.vtp").exists()
    )
]

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) -1)
    
    print(f"[INFO] Found {len(source_dirs)} patient folders.")
    print(f"[INFO] Using {max_workers} workers.")

    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for patient_dir in source_dirs:
            futures.append(
                executor.submit(
                    process_single_patient,
                    patient_dir,
                    source_dir,
                    save_to_dir,
                    reference_patient,
                    num_epi_samples,
                    num_lendo_samples,
                    num_rendo_samples,
                    sigma,
                    lamb,
                    rho,
                    create_processed_meshes,
                    store_processed_meshes,
                )
            )
        
        for idx, future in enumerate(as_completed(futures), start=1):
            patient_name, success, msg = future.result()

            if success:
                print(f"[{idx}/{len(futures)}] [OK] {patient_name} - {msg if msg else 'done'}")
            else:
                print(f"[{idx}/{len(futures)}] [ERROR] {patient_name} - {msg}")

            results.append((patient_name, success, msg))
    
    n_ok = sum(1 for _, success, _ in results if success)
    n_fail = len(results) - n_ok

    print("\n[SUMMARY]")
    print(f"Success: {n_ok}")
    print(f"Failed: {n_fail}")

    return results