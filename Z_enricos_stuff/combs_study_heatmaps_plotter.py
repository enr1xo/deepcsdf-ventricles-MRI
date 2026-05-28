"""
Questo codice prende in input il path ad un csv.
Il csv che si aspetta è quello prodotto a partire dalle combinazioni tramite lo script "combs_study_csv_generator.py".
Il csv ha 81 righe, 3 per ogni combinazione (27 in totale), oguna delle quali contiene la media e la std della metrica nelle 3 superici.

Ogni combinazione è composta da 3 parametri: sigma, lambda e rho.
Ogni parametro può assumere 3 valori, la cui combinazione è indicata nella colnna combination come ad esempio: S_0.0025-L_0.25-R_0.5
    il che significa:
        sigma = 0.0025
        lambda = 0.25
        rho = 0.5

Per ogni superfice questo codice fa:
    fissa una metrica
        fissa un valore di sigma
            genera una matrice 3x3 facendo variare lambda e rho,
            in ogni cella della matrice mettiamo il valore della metrica della relativa combinazione
        itera sui sigma
    itera sulle metriche
itera sulle superfici

le metrici dovranno avere valori crescenti di lambda e rho andando in un caso verso destra e nell'altro verso il basso.

Alla fine avremo quindi 27 matrici 3x3, siccome abbiamo 3 metriche e 3 superfici.

Il plot che voglaimo creare è il seguente:
    ha 3 righe e 3 colonne
    ogni riga è una metrica
    ogni colonna un valore di sigma, crescente verso destra

in ogni cella della griglia, riportiamo la matrice precendentemente costruita, dove però al posto del valore numerico mettiamo un colore ad una certa intensità in base al valore.
Generiamo in sostanza una heat map.

I valori in cui possono variare i parametri sono:

sigma = [0.0025, 0.025, 0.25]
lambda = [0.25, 0.5, 0.75]
rho = [0.5, 1, 2]
"""

# region HowToRun
# python combs_study_heatmaps_plotter.py \
#   -i path/to/metrics_avg_and_std_27combs_<3k or 5k>.csv \
#   -n name_of_the_plot either 3k or 5k \
#   -o /path/to/save/plots
# endregion

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------- da modifiare (_VALUES) done
# SIGMA_VALUES = [0.0025, 0.025, 0.25]
# LAMBDA_VALUES = [0.25, 0.5, 0.75]
# RHO_VALUES = [0.5, 1, 2]

DEPTH_VALUES = [3, 5, 7]
WIDTH_VALUES = [64, 128, 256, 512]
LATENT_VALUES = [16, 32, 64, 128]

SURFACE_ORDER = ["epicardium", "lv_endo", "rv_endo"]
METRICS_ORDER = ["chamfer", "haussdorff", "lddmm"]

METRIC_LIMITS = {
    "chamfer": (0.62, 2.6),
    "haussdorff": (3.0, 12),
    "lddmm": (0.0, 0.1),   # esempio, metti i tuoi valori
}
#----------------------------------------------------------------------- da modificare (description) done
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Legge il CSV riassuntivo prodotto da combs_study_csv_generator.py "
            "e genera, per ogni superficie, un plot 3x3 in cui "
            "ogni cella contiene una heatmap 3x3 costruita usando "
            "i valori medi delle metriche."
        )
    )
    parser.add_argument(
        "-i", "--input_csv",
        required=True,
        help="path al csv riassuntivo prodotto da combs_study_csv_generator.py"
    )
    parser.add_argument(
        "-n", "--plot_name",
        required=True,
        help="nome base del plot, ad esempio combs_metrics_heatmap_3k"
    )
    parser.add_argument(
        "-o", "--output_path",
        required=True,
        help="cartella dove salvare i plot"
    )
    return parser.parse_args()


#----------------------------------------------------------------------------- da modificare (pattern) done
def parse_combination(combination_str):
    """
    extract sigma, lambda and tho from strings like:
        S_0.0025-L_0.25-R_0.5
    """

    # pattern = r"S_([0-9.]+)-L_([0-9.]+)-R_([0-9.]+)"
    pattern = r"5k_D([0-9.]+)_W([0-9.]+)_L([0-9.]+)"
    match = re.fullmatch(pattern, combination_str.strip())

    if match is None:
        raise ValueError(f"unknown combination format: {combination_str}")
    
    sigma = float(match.group(1))
    lam = float(match.group(2))
    rho = float(match.group(3))

    return sigma, lam, rho

#-------------------------------------------------------------------------------
def find_metric_mean_columns(df):
    """
    finds all te columns that end with _mean and return the metric's name.
    ex: chamfer_mean -> chamfer
    """

    metric_cols = [c for c in df.columns if c.endswith("_mean")]
    metrics = [c[:-5] for c in metric_cols]

    ordered_metrics = [ m for m in METRICS_ORDER if m in metrics]
    remaining_metrics = [m for m in metrics if m not in ordered_metrics]

    return ordered_metrics + remaining_metrics

#------------------------------------------------------------------------------
def prepare_dataframe(input_csv):
    df = pd.read_csv(input_csv, sep=None, engine="python")

    required_cols = {"combination", "organ"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"required columns are missing: {sorted(missing)}")

    parsed_params = df["combination"].apply(parse_combination)
    # df["sigma"] = parsed_params.apply(lambda x: x[0])
    # df["lambda"] = parsed_params.apply(lambda x: x[1])
    # df["rho"] = parsed_params.apply(lambda x: x[2])

    df["dims"] = parsed_params.apply(lambda x: x[0])
    df["width"] = parsed_params.apply(lambda x: x[1])
    df["latent"] = parsed_params.apply(lambda x: x[2])

    return df

#---------------------------------------------------------------------------- da modificare (chiavi con cui prendo i valori di sigma, lambda e rho, saranno W,L e D)
def build_heatmap_matrix(df_surface, metric, sigma_value: None, depth_values:None):
    """
    assembles a 3x3 metrix with:
        - rows = increasing rho going downwards
        - cols = increasing lambda going right
    """

    # matrix = np.full((len(RHO_VALUES), len(LAMBDA_VALUES)), np.nan)
    matrix = np.full((len(WIDTH_VALUES), len(LATENT_VALUES)), np.nan)

    metric_col = f"{metric}_mean"

    if metric_col not in df_surface.columns:
        return matrix
    
    # filtered = df_surface[df_surface["sigma"] == sigma_value]
    filtered = df_surface[df_surface["dims"] == depth_values]

    # for i, rho in enumerate(RHO_VALUES):
    #     for j, lam in enumerate(LAMBDA_VALUES):
    #         match = filtered[
    #             (filtered["lambda"] == lam) &
    #             (filtered["rho"] == rho)
    #         ]

    #         if len(match) == 0:
    #             continue
    #         if len(match) > 1:
    #             raise ValueError(f"more rows for combo: sigma = {sigma_value}, \nlambda={lam}, \nrho={rho} \nfor metric={metric}")
            
    #         matrix[i,j] = match.iloc[0][metric_col]
    
    for i, width in enumerate(WIDTH_VALUES):
        for j, latent in enumerate(LATENT_VALUES):
            match = filtered[
                (filtered["width"] == width) &
                (filtered["latent"] == latent)
            ]

            if len(match) == 0:
                continue
            if len(match) > 1:
                raise ValueError(f"more rows for combo: dimension = {depth_values}, \nlatent={latent}, \nwidth={width} \nfor metric={metric}")
            
            matrix[i,j] = match.iloc[0][metric_col]
    
    return matrix

#------------------------------------------------------------
def compute_metric_color_limits(metric):
    if metric not in METRIC_LIMITS:
        raise ValueError(f"Nessun range definito per la metrica: {metric}")
    return METRIC_LIMITS[metric]

#---------------------------------------------------------- da modificare (labels, )
def plot_surface_heatmaps(df, surface, plot_name, output_dir, metrics):

    df_surface = df[df["organ"] == surface].copy()

    if df_surface.empty:
        print(f"[WARNING] Nessun dato trovato per la superficie: {surface}")
        return
    
    available_metrcs = [m for m in metrics if f"{m}_mean" in df_surface.columns]
    if not available_metrcs:
        print(f"[WARNING] Nessuna metrica disponibile per la superficie: {surface}")
        return
    
    nrows = len(available_metrcs)
    # ncols = len(SIGMA_VALUES)
    ncols = len(DEPTH_VALUES)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2*ncols, 4.0*nrows),
        squeeze=False,
        constrained_layout=True
    )

    fig.suptitle(surface, fontsize=16)

    sigma_value = None
    depth_value = None

    for row_idx, metric, in enumerate(available_metrcs):
        vmin, vmax = compute_metric_color_limits(metric)
        last_im = None

        # for col_idx, sigma_value in enumerate(SIGMA_VALUES):
        for col_idx, depth_value in enumerate(DEPTH_VALUES):
            ax = axes[row_idx, col_idx]
            # matrix = build_heatmap_matrix(df_surface, metric, sigma_value)
            matrix = build_heatmap_matrix(df_surface, metric, sigma_value, depth_value)

            im = ax.imshow(
                matrix,
                origin="upper",
                aspect="equal",
                vmin=vmin,
                vmax=vmax
            )

            last_im = im

            # ax.set_xticks(range(len(LAMBDA_VALUES)))
            # ax.set_xticklabels([str(x) for x in LAMBDA_VALUES])

            # ax.set_yticks(range(len(RHO_VALUES)))
            # ax.set_yticklabels([str(y) for y in RHO_VALUES])

            # ax.set_xlabel("lambda")
            # ax.set_ylabel("rho")

            # if row_idx == 0:
            #     ax.set_title(f"sigma = {sigma_value}")
            
            ax.set_xticks(range(len(LATENT_VALUES)))
            ax.set_xticklabels([str(x) for x in LATENT_VALUES])

            ax.set_yticks(range(len(WIDTH_VALUES)))
            ax.set_yticklabels([str(y) for y in WIDTH_VALUES])

            # ax.set_xlabel("lambda")
            # ax.set_ylabel("rho")

            ax.set_xlabel("latent")
            ax.set_ylabel("width")

            # if row_idx == 0:
            #     ax.set_title(f"sigma = {sigma_value}")

            if row_idx == 0:
                ax.set_title(f"depth = {depth_value}")

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i,j]
                    if np.isnan(value):
                        text = "NaN"
                    else:
                        text = f"{value:.3f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white")
        
        cbar = fig.colorbar(
            last_im,
            ax=axes[row_idx, :],
            fraction=0.04,
            pad=0.08
        )
        cbar.set_label(metric)

        axes[row_idx, 0].annotate(
            metric,
            xy=(-0.55, 0.5),
            xycoords="axes fraction",
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold"
        )

    # plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    output_file = output_dir / f"{plot_name}_{surface}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] salvato: {output_file}")


# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_path)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV non trovato: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(input_csv)
    metrics = find_metric_mean_columns(df)

    if not metrics:
        raise ValueError("Non ho trovato colonne *_mean nel CSV.")

    surfaces = [s for s in SURFACE_ORDER if s in df["organ"].unique()]
    remaining_surfaces = [s for s in df["organ"].unique() if s not in surfaces]
    surfaces.extend(remaining_surfaces)

    for surface in surfaces:
        plot_surface_heatmaps(
            df=df,
            surface=surface,
            plot_name=args.plot_name,
            output_dir=output_dir,
            metrics=metrics
        )


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()