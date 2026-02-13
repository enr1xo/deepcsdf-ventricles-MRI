from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json


COLORS_PALETTE = {
    # --- Pastel tones ---
    "pastel_blue": "#AECBFA",
    "pastel_green": "#B7E1A1",
    "pastel_pink": "#F4C2C2",
    "pastel_orange": "#FFD1A9",
    "pastel_yellow": "#FFF4A3",
    "pastel_purple": "#CBA3F4",
    "pastel_teal": "#A3E4D7",
    "pastel_red": "#F7A1A1",
    "pastel_gray": "#D8D8D8",
    "pastel_brown": "#E3C7A1",

    # --- Fluorescent / Neon tones ---
    "neon_green": "#39FF14",      # Matrix green
    "neon_blue": "#04D9FF",       # Cyan-blue glow
    "neon_pink": "#FF10F0",       # Magenta pink
    "neon_orange": "#FF6700",     # Bright orange
    "neon_purple": "#BF00FF",     # Electric purple
    "neon_yellow": "#F5FF00",     # Highlighter yellow
    "neon_red": "#FF073A",        # Vibrant red
    "neon_turquoise": "#00FFEF",  # Fluorescent aqua
    "neon_lime": "#CFFF04",       # Lime acid green
    "neon_magenta": "#FF00C8",    # Deep magenta
    "neon_orange2" : "#FF6464",
    "neon_cyan" : '#00FFFF',
    "electric_blue" : '#007BFF',

    # --- Extra vivid but not eye-burning ---
    "sky_blue": "#4FC3F7",
    "mint_green": "#98FF98",
    "coral": "#FF7F50",
    "violet": "#EE82EE",
    "sun_yellow": "#FFD300",
    "aqua": "#00FFFF",
    "hot_pink": "#FF69B4",
    "light_lavender": "#D8B7FF",
    "apple_green": "#8DB600",
    "deep_cerulean": "#007BA7",
}


# ================================================================ #
# region latent space visualization
# ================================================================ #
def map_categories(patient_names, categories = ["AF", "LEU_NORM"]):
    # I just love python
    return [ categories["AF" not in name] for name in patient_names ] 

def plot_PCA(latents, patients_names, save_fname = None):

    categories = map_categories(patients_names)
    map_colors = lambda s:  COLORS_PALETTE["neon_red"] if s == "AF" else  COLORS_PALETTE["neon_green"]
    y = [map_colors(s) for s in categories]

    pca = PCA(n_components=2)

    # Fit and transform
    latents_embedded = pca.fit_transform(latents)

    # Plot
    plt.scatter(latents_embedded[:,0], latents_embedded[:,1], c=y, s=50)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA embedding of latent codes')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
            label='AF',
            markerfacecolor=COLORS_PALETTE["neon_red"],
            markersize=8),
        Line2D([0], [0], marker='o', color='w',
            label='NORM',
            markerfacecolor=COLORS_PALETTE["neon_green"],
            markersize=8)
    ]

    plt.legend(handles=legend_elements, loc='upper left')

    if save_fname is None:
        plt.show()
    else:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight') 
        plt.close()        

    return

def plot_tSNE(latents, patients_names, reduce_dim_first = False, learning_rate = 100, max_iter = 1000, perplexity = 15, save_fname = None):

    categories = map_categories(patients_names)
    map_colors = lambda s:  COLORS_PALETTE["neon_red"] if s == "AF" else  COLORS_PALETTE["neon_green"]
    y = [map_colors(s) for s in categories]

    # It is highly recommended to use another dimensionality reduction method 
    # (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50)
    # if the number of features is very high.
    if reduce_dim_first:
        pca = PCA(n_components=50)
        latents = pca.fit_transform(latents)

    tsne = TSNE(n_components=2, perplexity=10, learning_rate=100, max_iter=1000, random_state=42)

    # Fit and transform
    X_embedded = tsne.fit_transform(latents)

    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y, s=50)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE embedding of latent codes')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
            label='AF',
            markerfacecolor=COLORS_PALETTE["neon_red"],
            markersize=8),
        Line2D([0], [0], marker='o', color='w',
            label='NORM',
            markerfacecolor=COLORS_PALETTE["neon_green"],
            markersize=8)
    ]

    plt.legend(handles=legend_elements, loc='upper left')

    if save_fname is None:
        plt.show()
    else:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight') 
        plt.close()     
        
    return

def plot_UMAP(latents, patients_names, n_neighbors = 15, min_dist = 0.05, save_fname = None):

    categories = map_categories(patients_names)
    map_colors = lambda s:  COLORS_PALETTE["neon_red"] if s == "AF" else  COLORS_PALETTE["neon_green"]
    y = [map_colors(s) for s in categories]

    umap_embedder = umap.UMAP(
        n_neighbors=n_neighbors,  # controls local vs global
        min_dist=min_dist,    # tightness of clusters
        n_components=2,  # output dims
        random_state=42  # reproducibility
    )

    # Fit & transform data
    latents_embedded = umap_embedder.fit_transform(latents)  # X = your high-dimensional data

    # Plot
    plt.scatter(latents_embedded[:,0], latents_embedded[:,1], c=y, s=50)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP embedding of latent codes')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
            label='AF',
            markerfacecolor=COLORS_PALETTE["neon_red"],
            markersize=8),
        Line2D([0], [0], marker='o', color='w',
            label='NORM',
            markerfacecolor=COLORS_PALETTE["neon_green"],
            markersize=8)
    ]

    plt.legend(handles=legend_elements, loc='upper left')

    if save_fname is None:
        plt.show()
    else:
        plt.savefig(save_fname, dpi=300, bbox_inches='tight') 
        plt.close()     

    return



# ================================================================ #
# region helpers
# ================================================================ #
def get_dataset_patients_names(data: dict):
    patient_names = []
    # file names are <patient_name>-<suffix>.npy
    for fullfname in data:
        patient_name = fullfname.split("-")[0]
        patient_names.append(patient_name)

    return patient_names

def associate_trained_embeddings_with_patients(version_dir):

    latents = np.load( version_dir / "latents.npy" )

    specs = json.load( open(version_dir / "hparams.json") )

    # this is the same file the dataloader uses in SDFSamples dataloader when in "train" mode !!
    train_fname = DATA_DIR / specs["TrainSplit"]

    patient_names = get_dataset_patients_names( json.load(open(train_fname)) )

    return latents, patient_names


if __name__ == "__main__":

    from pathlib import Path

    # LATENTS_DIR = Path("experiments/training_sweeps/RegLambda/version_3")

    # # # latents_name = "latent_codes_89_patients_version_114-codereg=0.000200-epochs=250"
    # # latents_name = "latent_codes_109_patients_version_89-codereg=0.000200-epochs=250"
    # # fname = LATENTS_DIR / str(latents_name + ".npz")
    # # latent_dict = np.load(fname)

    # patients_names = []
    # latent_codes = []
    # for name, code in latent_dict.items():
    #     if name not in ["AF001", "AF069", "LEU_NORM_F004"]:
    #         patients_names.append(name)
    #         latent_codes.append(code)

    # latent_codes = np.array(latent_codes)

    # IMAGES_DIR = Path("/home/davidenava_linux/AtriaProject/deepcsdf-atria/results/images")

    # # save_fname = IMAGES_DIR / f"PCA-{latents_name}.svg"
    # plot_PCA(latent_codes, patients_names, None)

    # # save_fname = IMAGES_DIR / f"tSNE-{latents_name}.svg"
    # plot_tSNE(latent_codes, patients_names, learning_rate=80, save_fname = None)

    # # save_fname = IMAGES_DIR / f"UMAP-{latents_name}.svg"
    # plot_UMAP(latent_codes, patients_names, save_fname = None)

    # Add project root to sys.path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))

    # Now you can import config
    from config import DATA_DIR

    version_dir = Path("experiments/training_sweeps/RegLambda/version_0")

    latents, patient_names = associate_trained_embeddings_with_patients(version_dir)

    colors = map_categories(patient_names)

    for i,latent in enumerate(latents):
        #if colors[i] == "AF":
            c = COLORS_PALETTE["coral"] if colors[i] == "AF" else COLORS_PALETTE["electric_blue"]
            plt.scatter( np.arange(0, latents.shape[-1]), latent, c = c, s=15)
    plt.show()

    plot_PCA(latents, patient_names)

    T_max = 0.0
    for perp in [5,10,15,20]:
        for lr in [50,100,150,200]:
            tsne = TSNE(n_components=2, perplexity=perp, learning_rate=lr, max_iter=1000, random_state=42)
            latents_embedded = tsne.fit_transform(latents)
            T = trustworthiness(latents, latents_embedded)
            if T > T_max:
                T_max = T
                perplexity = perp
                learning_rate = lr

    plot_tSNE(latents, patient_names, learning_rate=learning_rate, perplexity=perplexity)
    # plot_UMAP(latents, patient_names)

    