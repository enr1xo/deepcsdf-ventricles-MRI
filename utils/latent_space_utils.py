from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
import seaborn as sns
import pandas as pd
    

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
# region helpers
# ================================================================ #
def get_dataset_patients_names(data: dict):
    patient_names = []
    # file names are <patient_name>-<suffix>.npy
    for fullfname in data:
        patient_name = fullfname.split("-")[0]
        patient_names.append(patient_name)

    return patient_names

def associate_trained_embeddings_with_patients(version_dir, DATA_DIR):

    latents = np.load( version_dir / "latents.npy" )

    specs = json.load( open(version_dir / "hparams.json") )

    # this is the same file the dataloader uses in SDFSamples dataloader when in "train" mode !!
    train_fname = DATA_DIR / specs["TrainSplit"]

    patient_names = get_dataset_patients_names( json.load(open(train_fname)) )

    return latents, patient_names





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

def plot_PCA_explained_variance(latents, save_fname=None):

    pca = PCA()
    pca.fit(latents)
    
    explained_ratio = np.cumsum(pca.explained_variance_ratio_)

    effective_dim = np.argmax(explained_ratio >= 0.95) + 1 

    plt.figure(figsize=(8,5))
    plt.plot(np.arange(1, len(explained_ratio)+1), explained_ratio, marker='o')
    plt.axvline(effective_dim, color='r', linestyle='--', label=f'95% variance: {effective_dim} dims')
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA: Explained variance vs components")
    plt.grid(True)

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

def plot_embedding(ax, x, y, title, colors_af_norm, colors_train_test):

    # Aura (Train/Test)
    ax.scatter(x, y,
               s=350,
               c=colors_train_test,
               alpha=0.6,
               edgecolor='none')

    # Main point (AF/NORM)
    ax.scatter(x, y,
               s=80,
               c=colors_af_norm,
               edgecolors='black',
               linewidth=0.5)

    pad_x = 0.1 * (x.max() - x.min())
    pad_y = 0.1 * (y.max() - y.min())

    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)

    ax.set_title(title, fontsize=18)
    ax.set_aspect('equal')
    ax.set_box_aspect(1)

def plot_pca_varpca_tsne_umap(
        experiment_name, vnum,
        latents_train, latents_test, 
        train_patients, test_patients,
        IMAGES_DIR
):

    latents_all = np.vstack([latents_train, latents_test])

    patient_names = train_patients + test_patients

    colors_train_test = [ COLORS_PALETTE["pastel_orange"] if i < len(train_patients) else "plum" for i in range(len(patient_names))]

    colors_af_norm = [ COLORS_PALETTE["neon_red"] if "AF" in name else COLORS_PALETTE["neon_green"] for name in patient_names ]

    import matplotlib.patches as mpatches

    fig, axes = plt.subplot_mosaic(
        [["PCA", "explained_var"],
        ["tSNE", "UMAP"]],
        constrained_layout=True,
        figsize=(10,10)
    )

    pca_full = PCA()
    pca_full.fit(latents_all)

    explained = np.cumsum(pca_full.explained_variance_ratio_)
    effective_dim = np.argmax(explained >= 0.95) + 1 

    axes["explained_var"].plot(explained, c=COLORS_PALETTE["coral"], marker='o')
    axes["explained_var"].axvline(effective_dim, color='r', linestyle='--', label=f'95% variance: {effective_dim} dims')
    axes["explained_var"].set_title("PCA Explained Variance", fontsize=18)
    axes["explained_var"].grid(True)
    axes["explained_var"].set_xlabel("Components")
    axes["explained_var"].set_ylabel("Cumulative Variance")
    axes["explained_var"].set_ylim(0, 1.05)
    axes["explained_var"].set_box_aspect(1) 

    pca = PCA(n_components=2)
    latents_embedded = pca.fit_transform(latents_all)

    plot_embedding(
        axes["PCA"],
        latents_embedded[:, 0],
        latents_embedded[:, 1],
        "PCA",
        colors_af_norm,
        colors_train_test
    )


    tsne = TSNE(
        n_components=2,
        perplexity=15,
        learning_rate=150,
        max_iter=1000,
        random_state=42
    )

    latents_embedded = tsne.fit_transform(latents_all)

    plot_embedding(
        axes["tSNE"],
        latents_embedded[:, 0],
        latents_embedded[:, 1],
        "t-SNE",
        colors_af_norm,
        colors_train_test
    )


    umap_embedder = umap.UMAP(
        n_neighbors=15,
        min_dist=0.01,
        n_components=2,
        random_state=42
    )

    latents_embedded = umap_embedder.fit_transform(latents_all)

    plot_embedding(
        axes["UMAP"],
        latents_embedded[:, 0],
        latents_embedded[:, 1],
        "UMAP",
        colors_af_norm,
        colors_train_test
    )

    legend_elements = [
        mpatches.Patch(facecolor=COLORS_PALETTE["neon_red"], edgecolor='black', label='AF'),
        mpatches.Patch(facecolor=COLORS_PALETTE["neon_green"], edgecolor='black', label='NORM'),
        mpatches.Patch(facecolor=COLORS_PALETTE["pastel_orange"], edgecolor='black', label='Train'),
        mpatches.Patch(facecolor="plum", edgecolor='black', label='Test')
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),  # center horizontally, slightly below
        ncol=4,
        fontsize=16,
        frameon=True
    )

    save_fname = IMAGES_DIR / f"{experiment_name}/{experiment_name}-version_{vnum}-latents-all-embeddings_combined.pdf"
    plt.savefig(save_fname, dpi=300, bbox_inches="tight")

    save_fname = IMAGES_DIR / f"{experiment_name}/{experiment_name}-version_{vnum}-latents-all-embeddings_combined.svg"
    plt.savefig(save_fname, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    return



def mahalanobis(latents):
    # assuming latents are (N, latent size) !

    mu = np.mean(latents, axis=0)  # shape (64,)

    X_centered = latents - mu      # shape (N, 64)

    cov = np.cov(X_centered, rowvar=False)  # shape (64, 64)

    epsilon = 1e-6
    cov_reg = cov + epsilon * np.eye(cov.shape[0])

    inv_cov = np.linalg.inv(cov_reg)

    mahl = np.sqrt(np.sum((X_centered @ inv_cov) * X_centered, axis=1))  # shape (N,)

    return mahl

def plot_latent_correlation(latent_codes, figsize = (10, 8), fontsize = 12, show_numbers = False, save_fname=None):
    """
    latent_codes: np.array or torch tensor of shape (N, latent_dim)
    save_fname: optional, file path to save the heatmap
    """

    # just to assign easily labels I want
    if not isinstance(latent_codes, pd.DataFrame):
        latent_codes = pd.DataFrame(latent_codes, columns=[f"{i+1}" for i in range(latent_codes.shape[1])])

    # Compute correlation matrix
    corr_matrix = latent_codes.corr()

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        corr_matrix,
        annot=show_numbers,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}  # We'll adjust ticks below
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    plt.title("Latent Code Correlation Heatmap", fontsize=fontsize + 2)

    # ---- Adjust colorbar ticks ----
    cbar = ax.collections[0].colorbar
    vmin, vmax = cbar.vmin, cbar.vmax
    cbar_ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in cbar_ticks])
    cbar.ax.tick_params(labelsize=fontsize)


    if save_fname:
        plt.tight_layout()
        plt.savefig(save_fname, dpi=300, transparent=True)

    plt.show()

