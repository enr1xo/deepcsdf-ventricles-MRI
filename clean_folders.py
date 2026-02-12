from pathlib import Path

def clean_folder(folder_path):

    folder = Path(folder_path)

    for path in folder.iterdir():
        if path.is_dir():
            clean_folder(path)  
            path.rmdir()     
        else:    
            path.unlink()

    return


folders_to_clean = ["experiments/logs-train-temp"] #, "experiments/training_sweeps/RegLambdaAndCs"]

for folder_path in folders_to_clean:
    clean_folder(folder_path)