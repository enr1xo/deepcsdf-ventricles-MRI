import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from functools import reduce
import ast
import re
import numpy as np
import pyvista as pv
import json
from pathlib import Path
from config import METRICS_DIR, PATIENT_MESHES_DIR, IMAGES_DIR

def flatten_dict_keys(d, parent_key="", sep="."):
    """
        Flatten dictionary so all keys are at the same level

        ex. {
                "Network_specs" : { 
                    "latent_size" : 64
                    }
                "NumEpochs" : 100
            }
        
            becomes
            
            {
                "Network_specs.latent_size" : 64
                "NumEpochs" : 100
            }
                
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict_keys(v, new_key, sep=sep))
        else:
            items[new_key] = v

    return items

def parse_value(x):
    # numpy array or list from LDDMM
    if isinstance(x, (np.ndarray, list, tuple)):
        return float(x[0])
    if isinstance(x, str):
        return float(ast.literal_eval(x)[0])
    # scalar from chamfer or haussdorff
    return float(x)

def build_dataframe_from_versions_metrics(
    metrics_directory,
    experiment_name,
    search_csv_regex,
    add_column_name_1, add_column_name_2,
    get_specs_flatkey1: str, get_specs_flatkey2: str
)-> Tuple[pd.DataFrame, pd.DataFrame]:
    
    metrics_dir = Path("results/metrics") # Path(metrics_directory)

    dfs_train = []
    dfs_test = []

    # for these experiments, names are just like {experiment_name}-{version}-{metric}-{which_shapes}.csv
    for file_path in list( metrics_dir.glob(search_csv_regex) ):

        # go fetch the specs file to add columns I need to differentiate versions
        pattern = r"version_\w+"
        match = re.search(pattern, file_path.name)
        version_num = match.group(0).split("_")[-1] 

        with open(f"experiments/{experiment_name}/version_{version_num}/hparams.json") as f:
            specs = json.load(f)

        df = pd.read_csv(file_path)

        df["version"] = int(version_num)

        specs = flatten_dict_keys(specs)

        df[add_column_name_1] = str( specs[get_specs_flatkey1] )
        df[add_column_name_2] = str( specs[get_specs_flatkey2] )

        if "train" in str(file_path.name):    
            dfs_train.append(df)
        elif "test" in file_path.name:
            dfs_test.append(df)

    if len(dfs_train) > 0:
        df_all_train = pd.concat(dfs_train, ignore_index=True)
        df_all_train["value"] = df_all_train["value"].apply(parse_value)
    else:
        df_all_train = None

    if len(dfs_test) > 0:
        df_all_test = pd.concat(dfs_test, ignore_index=True)
        df_all_test["value"] = df_all_test["value"].apply(parse_value)
    else:
        df_all_test = None

    return df_all_train, df_all_test










if __name__ == "__main__":

    experiment_name = "LipAndAct"
    exp_subdir = ""

    build_dataframe_from_versions_metrics(
        metrics_directory=METRICS_DIR,
        experiment_name=experiment_name,
        search_csv_regex=f"*{experiment_name}*.parquet",
        add_column_name_1="alpha",
        add_column_name_2="act",
        get_specs_flatkey1="Network_specs.lipschitz_layers",
        get_specs_flatkey2="Network_specs.activation"
    )

    