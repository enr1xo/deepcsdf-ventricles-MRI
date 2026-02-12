from config import EXPERIMENTS_DIR, RESULTS_DIR, IMAGES_DIR, RECONSTRUCTED_MESHES_DIR, LATENTS_DIR, METRICS_DIR

dirs_to_create = [
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    IMAGES_DIR,
    RECONSTRUCTED_MESHES_DIR,
    LATENTS_DIR,
    METRICS_DIR,
]

for d in dirs_to_create:
    d.mkdir(parents=True, exist_ok=True)

print("All experiment directories are ready.")