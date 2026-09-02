"""
visualizziamo un file npy con i dati di un ventricolo, e lo confrontiamo con la mesh del ventricolo stesso
in una visualizzazione interattiva di pyvista.
"""

from pathlib import Path
import numpy as np
import pyvista as pv

#NPY_PATH = Path("/home/rizzardi/Schreibtisch/MRI_model/generated_npy_three_axis_LA_volume_2mm")
#CARDIAC_SURFS_PATH = Path("/home/rizzardi/Schreibtisch/AF001_aligned_processed")

NPY_PATH = Path(r"C:\Users\e.rizzardi\OneDrive\Desktop")
CARDIAC_SURFS_PATH = Path(r"C:\Users\e.rizzardi\OneDrive\Desktop\AF001_aligned_processed")

NPY_SUFFIX = "_echo_samples.npy"
NPY_SUFFIX = "_three_axis_mri_grid_samples.npy"
#NPY_SUFFIX = "-epi_lv_rv_19500_coords_and_sdf.npy"

PATIENT_ID = "AF001"
#npy_file = NPY_PATH / f"{PATIENT_ID}_three_axis_mri_samples.npy"
npy_file = NPY_PATH / f"{PATIENT_ID}_{NPY_SUFFIX}"

epi_surf_file = CARDIAC_SURFS_PATH / f"{PATIENT_ID}" /f"epicardium-processed.vtp"
lv_surf_file = CARDIAC_SURFS_PATH / f"{PATIENT_ID}" /f"lv_endo-processed.vtp"
rv_surf_file = CARDIAC_SURFS_PATH / f"{PATIENT_ID}" /f"rv_endo-processed.vtp"

"""
Visualizziamo un file NPY con i punti campionati di un ventricolo
e lo confrontiamo con le relative superfici cardiache.

Formato NPY atteso:
    colonna 0: x
    colonna 1: y
    colonna 2: z
    colonna 3: sdf_epi
    colonna 4: sdf_lv
    colonna 5: sdf_rv
    colonna 6: mask_epi
    colonna 7: mask_lv
    colonna 8: mask_rv
"""

from pathlib import Path

import numpy as np
import pyvista as pv


# ============================================================
# PARAMETRI
# ============================================================

#NPY_PATH = Path(
    #"/home/rizzardi/Schreibtisch/MRI_model/generated_npy_three_axis_grid")

#CARDIAC_SURFS_PATH = Path(
    #"/home/rizzardi/Schreibtisch/AF001_aligned_processed")

#PATIENT_ID = "AF013"

#NPY_SUFFIX = "_three_axis_mri_grid_samples.npy"

# Modalità con cui colorare i punti:
#
# "sdf_epi"
# "sdf_lv"
# "sdf_rv"
# "mask_epi"
# "mask_lv"
# "mask_rv"
# "constant"
POINT_COLOR_MODE = "sdf_epi"

POINT_SIZE = 5.0
SURFACE_OPACITY = 0.30

# Se True, usa solo i punti per cui la mask della superficie
# selezionata vale 1.
FILTER_BY_MASK = False

# Limite opzionale al numero di punti visualizzati.
# Mettere None per visualizzarli tutti.
MAX_POINTS_TO_DISPLAY = None

# Se True, seleziona casualmente i punti quando viene applicato
# MAX_POINTS_TO_DISPLAY.
RANDOM_SUBSAMPLING = True

RANDOM_SEED = 42

# Se le superfici sono in micrometri e i punti NPY in millimetri,
# impostare SURFACE_SCALE = 0.001.
#
# Se sono già nello stesso sistema di coordinate, lasciare 1.0.
SURFACE_SCALE = 1.0


# ============================================================
# PATH
# ============================================================

patient_dir = CARDIAC_SURFS_PATH / PATIENT_ID

npy_file = NPY_PATH / f"{PATIENT_ID}{NPY_SUFFIX}"

# epi_surf_file = (
#     patient_dir
#     / f"{PATIENT_ID}_epicardium-processed.vtp"
# )

# lv_surf_file = (
#     patient_dir
#     / f"{PATIENT_ID}_lv_endo-processed.vtp"
# )

# rv_surf_file = (
#     patient_dir
#     / f"{PATIENT_ID}_rv_endo-processed.vtp"
# )


# ============================================================
# FUNZIONI
# ============================================================

def check_file(path: Path, description: str) -> None:
    """Controlla che un file esista."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} non trovato:\n{path}"
        )


def load_surface(path: Path, scale: float = 1.0) -> pv.PolyData:
    """
    Carica una superficie con PyVista e applica eventualmente
    un fattore di scala.
    """
    surface = pv.read(path)

    if surface.n_points == 0:
        raise ValueError(
            f"La superficie non contiene punti:\n{path}"
        )

    surface = surface.extract_surface(
        #algorithm="dataset_surface"
    ).triangulate()

    if scale != 1.0:
        surface.points = surface.points * scale

    return surface


def select_points_to_display(
    data: np.ndarray,
    max_points: int | None,
    random_subsampling: bool,
    seed: int,
) -> np.ndarray:
    """Riduce opzionalmente il numero di punti visualizzati."""

    if max_points is None or len(data) <= max_points:
        return data

    if max_points <= 0:
        raise ValueError(
            "MAX_POINTS_TO_DISPLAY deve essere positivo o None."
        )

    if random_subsampling:
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            len(data),
            size=max_points,
            replace=False,
        )
    else:
        indices = np.linspace(
            0,
            len(data) - 1,
            max_points,
            dtype=int,
        )

    return data[indices]


def filter_data_by_selected_mask(
    data: np.ndarray,
    color_mode: str,
) -> np.ndarray:
    """
    Filtra i punti usando la mask associata alla modalità scelta.
    """

    mode_to_mask_column = {
        "sdf_epi": 6,
        "sdf_lv": 7,
        "sdf_rv": 8,
        "mask_epi": 6,
        "mask_lv": 7,
        "mask_rv": 8,
    }

    mask_column = mode_to_mask_column.get(color_mode)

    if mask_column is None:
        print(
            "FILTER_BY_MASK ignorato: "
            "POINT_COLOR_MODE='constant'."
        )
        return data

    valid = data[:, mask_column] > 0.5

    print(
        f"Punti con mask valida per {color_mode}: "
        f"{np.count_nonzero(valid)}/{len(data)}"
    )

    return data[valid]


def create_point_cloud(
    data: np.ndarray,
    color_mode: str,
) -> tuple[pv.PolyData, str | None]:
    """
    Crea la point cloud PyVista e restituisce il nome dello
    scalare usato per colorarla.
    """

    points = data[:, :3]

    cloud = pv.PolyData(points)

    scalar_columns = {
        "sdf_epi": 3,
        "sdf_lv": 4,
        "sdf_rv": 5,
        "mask_epi": 6,
        "mask_lv": 7,
        "mask_rv": 8,
    }

    if color_mode == "constant":
        return cloud, None

    if color_mode not in scalar_columns:
        raise ValueError(
            f"POINT_COLOR_MODE non riconosciuto: {color_mode}\n"
            f"Valori validi: {list(scalar_columns)} oppure 'constant'."
        )

    column = scalar_columns[color_mode]
    cloud[color_mode] = data[:, column]

    return cloud, color_mode


def print_npy_summary(data: np.ndarray) -> None:
    """Stampa un riepilogo del contenuto del file NPY."""

    print("\n" + "=" * 60)
    print("RIEPILOGO FILE NPY")
    print("=" * 60)

    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Numero punti: {len(data)}")

    xyz = data[:, :3]

    print("\nCoordinate:")
    print(f"  X: [{xyz[:, 0].min():.4f}, {xyz[:, 0].max():.4f}]")
    print(f"  Y: [{xyz[:, 1].min():.4f}, {xyz[:, 1].max():.4f}]")
    print(f"  Z: [{xyz[:, 2].min():.4f}, {xyz[:, 2].max():.4f}]")

    if data.shape[1] >= 6:
        print("\nSDF:")
        for name, column in [
            ("EPI", 3),
            ("LV", 4),
            ("RV", 5),
        ]:
            values = data[:, column]
            print(
                f"  {name}: "
                f"min={values.min():.4f}, "
                f"max={values.max():.4f}, "
                f"mean={values.mean():.4f}"
            )

    if data.shape[1] >= 9:
        print("\nMask:")
        for name, column in [
            ("EPI", 6),
            ("LV", 7),
            ("RV", 8),
        ]:
            values = data[:, column]
            unique_values, counts = np.unique(
                values,
                return_counts=True,
            )

            summary = ", ".join(
                f"{value:g}: {count}"
                for value, count in zip(unique_values, counts)
            )

            print(f"  {name}: {summary}")

    print("=" * 60 + "\n")


# ============================================================
# CALLBACK PER CHECKBOX
# ============================================================

def make_visibility_callback(actor):
    """Crea una callback per mostrare/nascondere un actor."""

    def callback(visible: bool) -> None:
        actor.SetVisibility(visible)

    return callback


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Controllo file
    # --------------------------------------------------------

    check_file(npy_file, "File NPY")
    check_file(epi_surf_file, "Superficie epicardica")
    check_file(lv_surf_file, "Superficie endocardica LV")
    check_file(rv_surf_file, "Superficie endocardica RV")

    # --------------------------------------------------------
    # Caricamento NPY
    # --------------------------------------------------------

    data = np.load(npy_file)

    if data.ndim != 2:
        raise ValueError(
            f"Il file NPY deve essere una matrice 2D, "
            f"ma ha shape {data.shape}."
        )

    if data.shape[1] < 6:
        raise ValueError(
            "Il file NPY deve contenere almeno 6 colonne:\n"
            "[x, y, z, sdf_epi, sdf_lv, sdf_rv]."
        )

    if POINT_COLOR_MODE.startswith("mask") and data.shape[1] < 9:
        raise ValueError(
            "Per visualizzare le mask il file NPY deve avere "
            "almeno 9 colonne."
        )

    if FILTER_BY_MASK and data.shape[1] < 9:
        raise ValueError(
            "FILTER_BY_MASK=True richiede un file NPY "
            "con almeno 9 colonne."
        )

    if not np.all(np.isfinite(data)):
        n_invalid = np.count_nonzero(~np.isfinite(data))

        print(
            f"Attenzione: trovati {n_invalid} valori NaN o infiniti."
        )

        valid_rows = np.all(np.isfinite(data), axis=1)
        data = data[valid_rows]

        print(
            f"Punti rimasti dopo la rimozione: {len(data)}"
        )

    print_npy_summary(data)

    # --------------------------------------------------------
    # Filtraggio e subsampling
    # --------------------------------------------------------

    if FILTER_BY_MASK:
        data = filter_data_by_selected_mask(
            data=data,
            color_mode=POINT_COLOR_MODE,
        )

    data = select_points_to_display(
        data=data,
        max_points=MAX_POINTS_TO_DISPLAY,
        random_subsampling=RANDOM_SUBSAMPLING,
        seed=RANDOM_SEED,
    )

    if len(data) == 0:
        raise ValueError(
            "Non è rimasto nessun punto da visualizzare."
        )

    print(f"Punti visualizzati: {len(data)}")

    # --------------------------------------------------------
    # Caricamento superfici
    # --------------------------------------------------------

    epi_surface = load_surface(
        epi_surf_file,
        scale=SURFACE_SCALE,
    )

    # ========================================================
    # DEBUG COORDINATE / SCALA
    # ========================================================

    points_debug = data[:, :3]

    print("\n" + "=" * 60)
    print("DEBUG COORDINATE")
    print("=" * 60)

    print("\nNPY bounds:")
    print("  min:", points_debug.min(axis=0))
    print("  max:", points_debug.max(axis=0))
    print("  centro:", points_debug.mean(axis=0))

    print("\nEPI bounds:")
    print(epi_surface.bounds)
    print("EPI center:")
    print(epi_surface.center)

    print("\nDimensioni NPY:")
    print(points_debug.max(axis=0) - points_debug.min(axis=0))

    print("\nDimensioni EPI:")
    print([
        epi_surface.bounds.x_max - epi_surface.bounds.x_min,
        epi_surface.bounds.y_max - epi_surface.bounds.y_min,
        epi_surface.bounds.z_max - epi_surface.bounds.z_min,
    ])

    print("=" * 60)
    # fine debug

    lv_surface = load_surface(
        lv_surf_file,
        scale=SURFACE_SCALE,
    )

    rv_surface = load_surface(
        rv_surf_file,
        scale=SURFACE_SCALE,
    )

    print("\nSuperfici caricate:")
    print(
        f"  EPI: {epi_surface.n_points} punti, "
        f"{epi_surface.n_cells} celle"
    )
    print(
        f"  LV : {lv_surface.n_points} punti, "
        f"{lv_surface.n_cells} celle"
    )
    print(
        f"  RV : {rv_surface.n_points} punti, "
        f"{rv_surface.n_cells} celle"
    )

    # --------------------------------------------------------
    # Creazione point cloud
    # --------------------------------------------------------

    point_cloud, scalar_name = create_point_cloud(
        data=data,
        color_mode=POINT_COLOR_MODE,
    )

    # debug
    print("\nDEBUG POINT CLOUD")
    print("Numero punti point_cloud:", point_cloud.n_points)
    print("Numero celle point_cloud:", point_cloud.n_cells)
    print("Bounds point_cloud:", point_cloud.bounds)
    # fine debug


    # --------------------------------------------------------
    # Plotter
    # --------------------------------------------------------

    plotter = pv.Plotter(
        window_size=(1600, 1000)
    )

    plotter.set_background("white")

    # Superfici
    epi_actor = plotter.add_mesh(
        epi_surface,
        color="lightgray",
        opacity=SURFACE_OPACITY,
        smooth_shading=True,
        label="Epicardio",
    )

    lv_actor = plotter.add_mesh(
        lv_surface,
        color="lightcoral",
        opacity=SURFACE_OPACITY,
        smooth_shading=True,
        label="Endocardio LV",
    )

    rv_actor = plotter.add_mesh(
        rv_surface,
        color="lightblue",
        opacity=SURFACE_OPACITY,
        smooth_shading=True,
        label="Endocardio RV",
    )

    # Punti
    # if scalar_name is None:
    #     points_actor = plotter.add_mesh(
    #         point_cloud,
    #         color="black",
    #         point_size=POINT_SIZE,
    #         render_points_as_spheres=True,
    #         label="Punti NPY",
    #     )

    # else:
    #     scalar_values = point_cloud[scalar_name]

    #     add_mesh_kwargs = {
    #         "scalars": scalar_name,
    #         "point_size": POINT_SIZE,
    #         "render_points_as_spheres": True,
    #         "label": f"Punti: {scalar_name}",
    #         "show_scalar_bar": True,
    #     }

    #     # Le mask hanno valori discreti 0 e 1.
    #     if scalar_name.startswith("mask"):
    #         add_mesh_kwargs.update(
    #             {
    #                 "cmap": ["darkred", "limegreen"],
    #                 "clim": [0.0, 1.0],
    #                 "categories": True,
    #                 "scalar_bar_args": {
    #                     "title": scalar_name,
    #                     "n_labels": 2,
    #                 },
    #             }
    #         )

    #     else:
    #         # Per gli SDF usiamo una scala simmetrica rispetto a zero.
    #         max_abs = float(
    #             np.nanmax(np.abs(scalar_values))
    #         )

    #         if max_abs == 0:
    #             max_abs = 1.0

    #         add_mesh_kwargs.update(
    #             {
    #                 "cmap": "coolwarm",
    #                 "clim": [-max_abs, max_abs],
    #                 "scalar_bar_args": {
    #                     "title": scalar_name,
    #                 },
    #             }
    #         )

    #     points_actor = plotter.add_mesh(
    #         point_cloud,
    #         **add_mesh_kwargs,
    #     )

    # # Punti - TEST SEMPLICE
    # points_actor = plotter.add_points(
    #     data[:, :3],
    #     color="black",
    #     point_size=15,
    #     render_points_as_spheres=True,
    #     label="Punti NPY",
    # )
    
    # Punti
    if scalar_name is None:
        points_actor = plotter.add_points(
            data[:, :3],
            color="black",
            point_size=POINT_SIZE,
            render_points_as_spheres=False,
            label="Punti NPY",
        )

    else:
        scalar_values = point_cloud[scalar_name]

        max_abs = float(np.nanmax(np.abs(scalar_values)))

        if max_abs == 0:
            max_abs = 1.0

        points_actor = plotter.add_points(
            data[:, :3],
            scalars=scalar_values,
            point_size=POINT_SIZE,
            render_points_as_spheres=False,
            cmap="coolwarm",
            clim=[-max_abs, max_abs],
            show_scalar_bar=True,
            scalar_bar_args={
                "title": scalar_name,
            },
            label=f"Punti: {scalar_name}",
        )

    # --------------------------------------------------------
    # Testo informativo
    # --------------------------------------------------------

    plotter.add_text(
        (
            f"Paziente: {PATIENT_ID}\n"
            f"Punti visualizzati: {len(data)}\n"
            f"Colorazione: {POINT_COLOR_MODE}"
        ),
        position="upper_left",
        font_size=11,
        color="black",
    )

    # --------------------------------------------------------
    # Checkbox
    # --------------------------------------------------------

    checkbox_x = 10
    checkbox_start_y = 150
    checkbox_step = 45
    checkbox_size = 30

    plotter.add_checkbox_button_widget(
        make_visibility_callback(points_actor),
        value=True,
        position=(checkbox_x, checkbox_start_y),
        size=checkbox_size,
    )

    plotter.add_text(
        "Punti NPY",
        position=(checkbox_x + 40, checkbox_start_y + 5),
        font_size=10,
        color="black",
    )

    plotter.add_checkbox_button_widget(
        make_visibility_callback(epi_actor),
        value=True,
        position=(
            checkbox_x,
            checkbox_start_y - checkbox_step,
        ),
        size=checkbox_size,
    )

    plotter.add_text(
        "Epicardio",
        position=(
            checkbox_x + 40,
            checkbox_start_y - checkbox_step + 5,
        ),
        font_size=10,
        color="black",
    )

    plotter.add_checkbox_button_widget(
        make_visibility_callback(lv_actor),
        value=True,
        position=(
            checkbox_x,
            checkbox_start_y - 2 * checkbox_step,
        ),
        size=checkbox_size,
    )

    plotter.add_text(
        "Endocardio LV",
        position=(
            checkbox_x + 40,
            checkbox_start_y - 2 * checkbox_step + 5,
        ),
        font_size=10,
        color="black",
    )

    plotter.add_checkbox_button_widget(
        make_visibility_callback(rv_actor),
        value=True,
        position=(
            checkbox_x,
            checkbox_start_y - 3 * checkbox_step,
        ),
        size=checkbox_size,
    )

    plotter.add_text(
        "Endocardio RV",
        position=(
            checkbox_x + 40,
            checkbox_start_y - 3 * checkbox_step + 5,
        ),
        font_size=10,
        color="black",
    )

    # --------------------------------------------------------
    # Elementi grafici
    # --------------------------------------------------------

    plotter.add_axes(
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
    )

    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
    )

    plotter.add_legend(
        bcolor="white",
        border=True,
        size=(0.18, 0.15),
    )

    plotter.view_isometric()
    plotter.reset_camera()

    plotter.show(
        title=f"{PATIENT_ID} - NPY e superfici cardiache"
    )


if __name__ == "__main__":
    main()