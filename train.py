""" Train deepsdf model to learn atrial shapes. Needs a specification file that gives all data, decoder, training parameters and optionally post-processing directories
"""
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import warnings
from time import time
from pathlib import Path
import lightning as pl
from model.atria_dataloader import SDFBalancedDataModuleGPU #SDFDataModuleGPU #SDFDataModule #SDFBalancedDataModule
from model.atria_deepsdf_decoder import Decoder, DeepSDFBalancedGPU #DeepSDFGPU #DeepSDF 
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger

# warnings.filterwarnings("ignore")
# logger.remove()
# logger.add(sys.stdout, level="INFO", filter=lambda record: record["level"].name == "INFO") # info → stdout
# logger.add(sys.stderr, level="ERROR") # error → stderr


# # =========== setup for A100 GPU =========== #
# torch.set_float32_matmul_precision("high")   # "medium" if instability (NaNs / loss spikes) appear
# PRECISION = "bf16-mixed"

# # =========== setup for RTX 3090 GPU =========== #
torch.set_float32_matmul_precision("medium") # "medium" if instability (NaNs / loss spikes) appear
PRECISION = "16-mixed"

from config import SPECS_FILES_DIR, DATA_DIR, EXPERIMENTS_DIR

SPECS_FILE = SPECS_FILES_DIR / "specs_deepsdfatria.json" # default if not specified on execution


# ============================== #
# HELPERS
# ============================== #
def deep_update(specs_base: dict, override_specs: dict) -> dict:
    for k, v in override_specs.items():
        if (
            k in specs_base
            and isinstance(specs_base[k], dict)
            and isinstance(v, dict)
        ):
            deep_update(specs_base[k], v)
        else:
            specs_base[k] = v
    return specs_base


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


def check_override_specs_validity(override_specs, specs_base):
    # flatten both structures to easily check even nested keys
    override_specs = flatten_dict_keys(override_specs)
    specs_base = flatten_dict_keys(specs_base)

    for key in override_specs.keys():
        if specs_base.get(key, None) is None:
            raise ValueError(f"Requested invalid key to override: '{key}', check valid keys in specs file.")
        
    return


# ============================== #
# CHECKPOINTING - manual
# ============================== #
class SaveDecoderCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        decoder = pl_module.decoder
        out = (Path(trainer.log_dir) / "decoder_weights.pth")
        torch.save(decoder.state_dict(), out)



# ============================== #
# TRAINING
# ============================== #
EXPERIMENT_NAME = "deepsdfatria_training_local"

def train(specs: dict, show_progress = False):

    # region LOAD DATA 
    datamodule = SDFBalancedDataModuleGPU(
        specs = specs
    )

    datamodule.setup("fit")  

    NumScenes = datamodule.num_fit_scenes 

    print(f"Loaded {NumScenes} scenes in total.")

    # region MODEL
    decoder = Decoder(**specs["Network_specs"])

    print( decoder.description() )

    model = DeepSDFBalancedGPU(
        decoder = decoder,
        specs = specs
    )

    model.set_embedding( num_scenes = NumScenes )


    # region CHECKPOINTS and LOGS 

    # experiment logger
    logger_tb = TensorBoardLogger(
        save_dir = EXPERIMENTS_DIR, # All experiment folders (named by name/version) will be created inside this directory.
        name = EXPERIMENT_NAME,
        default_hp_metric=False,
        log_graph=False
    )   

    version_dir = Path(logger_tb.log_dir)
    checkpoint_dir = version_dir / "checkpoints"    # --> created inside every version_x folder (every run)

    # load checkpoint to resume training if wanted
    load_version = specs.get("resume_training_from_version","-")
    saved_ckpt_dir = Path( EXPERIMENTS_DIR /  str(logger_tb.name) / load_version / "checkpoints" )
    ckpt_file_path = None
    if saved_ckpt_dir.exists():
        ckpt_files =  list( Path(saved_ckpt_dir).glob("*.ckpt") )
        if ckpt_files: # pick last checkpoint by epoch
            ckpt_file_path = max(ckpt_files, key = lambda file: int( re.search(r"epoch_([0-9]+)", file.name ).group(1) ) ) 
        else:
            logger.warning(f"Found directory for checkpoints for wanted version: {load_version}, but contained no .ckpt files")  

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch}",
        auto_insert_metric_name=False,
        every_n_epochs= specs.get("checkpoint_every_n_epochs", 10000),
        save_top_k=-1,
    )

    # region TRAIN
    NumEpochs = specs.get("NumEpochs", 5)

    trainer = pl.Trainer(
        accelerator="gpu",     
        devices=1,             
        max_epochs= NumEpochs, 
        precision=PRECISION,  
        enable_model_summary=False,  
        enable_progress_bar=show_progress,
        log_every_n_steps=10,
        logger=logger_tb,
        enable_checkpointing=True,
        callbacks = [SaveDecoderCallback(), checkpoint_callback]
    )

    tic = time()
    trainer.fit(model, datamodule = datamodule, ckpt_path = ckpt_file_path)
    toc = time() - tic

    logger.info(f"Time elapsed for fit: {toc:.2f} seconds ")

    return version_dir.name



if __name__ == "__main__":
    
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--specs_file_path", type=str, default = None)
    parser.add_argument("--experiment_name", type=str, default = None,
                        help="Default is deepsdf_atria_training, and becomes the directory " \
                        "name in which checkpoints and logs are saved, under version_x folder for each run."
    )
    parser.add_argument("--train_mode", type=str, default="use_specs_file")
    parser.add_argument("--override_specs", type=str, default=None)
    parser.add_argument("--show_progress", action="store_true")
    args = parser.parse_args()

    match args.train_mode:

        case "use_specs_file":
            if args.specs_file_path is not None:
                SPECS_FILE = str(args.specs_file_path)
            
            if args.experiment_name is not None:
                EXPERIMENT_NAME = str(args.experiment_name)

            version = train( specs = json.load(open(SPECS_FILE)), show_progress = args.show_progress )

        case "compose_specs_from_options":

            if args.experiment_name is not None:
                EXPERIMENT_NAME = str(args.experiment_name)

            if args.specs_file_path is not None:
                SPECS_FILE = str(args.specs_file_path)
            
            # now overwrite specs fields with wanted specs
            specs = json.load(open(SPECS_FILE))
            override_specs = json.loads(args.override_specs) # in args arriva dal bash come STRINGA json

            # be sure requested override fields are actually valid:
            check_override_specs_validity(override_specs, specs)

            specs = deep_update(specs, override_specs)

            version = train( specs = specs, show_progress = args.show_progress )

    # this is to then be captured from a bash file and retrieve the version that has been trained to send myself an email
    print(f"TRAINING_DONE_VERSION={version}", flush=True)

