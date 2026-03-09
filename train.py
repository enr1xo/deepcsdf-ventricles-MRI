""" Train deepsdf model to learn atrial shapes. Needs a specification file that gives all data, decoder, training parameters and optionally post-processing directories
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import json
import torch
import numpy as np
from time import time
from pathlib import Path
try:
    import lightning as pl # pyright: ignore[reportMissingImports]
    from lightning.pytorch.loggers import TensorBoardLogger # pyright: ignore[reportMissingImports]
    from lightning.pytorch.callbacks import ModelCheckpoint # pyright: ignore[reportMissingImports]
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger # pyright: ignore[reportMissingImports]
    from pytorch_lightning.callbacks import ModelCheckpoint # pyright: ignore[reportMissingImports]
    
from model.deepsdf_dataloader import SDFDataModule
from model.deepsdf_decoder import Decoder, DeepSDF 

from config import SPECS_FILES_DIR, EXPERIMENTS_DIR


# Set precision 
PRECISION = "32"

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0).lower()
    print(f"Detected GPU: {gpu_name}")

    if any(x in gpu_name for x in ["h100", "a100", "l40s"]): # Setup for H100 / A100 / L40S
        torch.set_float32_matmul_precision("high")   # "medium" if instability appears
        PRECISION = "bf16-mixed"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    elif "3090" in gpu_name: # Setup for RTX 3090        
        torch.set_float32_matmul_precision("medium")
        PRECISION = "16-mixed"

    elif "1050" in gpu_name:  # Setup for GTX 1050
        PRECISION = "32"

    else:
        PRECISION = "32"
else:
    print("CUDA not available, using CPU with full precision.")
    PRECISION = "32"

print(f"Using precision: {PRECISION}")


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
# region CALLBACKS - manual use of hooks, so I don't pollute the Module DeepSDF itself, and I keep it more reusable and clean
# ============================== #
# Instead of calling hooks like on_train_start and on_train_end in the Module I define new callbacks
class SaveDecoderCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        decoder = pl_module.decoder
        out = (Path(trainer.log_dir) / "decoder_weights.pth")
        torch.save(decoder.state_dict(), out)

class SaveSpecsCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.logger is not None:
            json_path = Path(trainer.logger.log_dir) / "hparams.json"
            with open(json_path, "w") as f:
                json.dump(pl_module.specs, f, indent=4)

class SaveEmbeddingsCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        npy_path = Path(trainer.log_dir) / "latents.npy"
        embeddings = pl_module.lat_vecs["trainable"].weight.data.cpu().numpy()
        np.save(npy_path, embeddings)

class MonitorLatentChannelsCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=2000):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if self.log_every_n_epochs is not None:
            epoch = trainer.current_epoch
            if (epoch + 1) % self.log_every_n_epochs != 0:
                return

            decoder = pl_module.decoder
            latent_size = decoder.latent_size

            # ----- latent input channels weights
            layer = getattr(decoder, f"lin0")
            W = layer.weight   # assuming shape: (out_dim, latent_size + 3) (i.e. input is [code, xyz])

            latent_weights = W[:, :latent_size].detach().cpu()
            xyz_weights = W[:, latent_size:latent_size+3].detach().cpu()

            norm_metrics = {
                "latent_layer0_norm" : latent_weights.norm().item(),
                "xyz_layer0_norm" : xyz_weights.norm().item(),
                "layer0_norm_z_over_x_ratio" : latent_weights.norm().item() / xyz_weights.norm().item()
            }

            pl_module.log_dict(norm_metrics, logger=True, on_step=False, on_epoch=True)
        return
    
class MonitorInputGradNormsCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=2000):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        if self.log_every_n_epochs is not None:
            box_lim = 1.05
            resolution = 64
            x = np.linspace(-box_lim, box_lim, resolution)
            y = np.linspace(-box_lim, box_lim, resolution)
            z = np.linspace(-box_lim, box_lim, resolution)
            xx, yy, zz = np.meshgrid(x, y, z)
            xyz = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
            self.xyz = torch.from_numpy(xyz).float()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.log_every_n_epochs is not None:
            epoch = trainer.current_epoch
            if (epoch + 1) % self.log_every_n_epochs != 0:
                return

            decoder = pl_module.decoder
            device = next(decoder.parameters()).device

            code = pl_module.lat_vecs["trainable"](torch.tensor([0], device=device))
            xyz = self.xyz.to(device)

            # ----- gradients norm
            N = xyz.shape[0]
            code_exp = code.repeat(N, 1)  # (N, latent_dim)

            xyz.requires_grad_(True)
            code_exp.requires_grad_(True)

            # freeze decoder params
            for p in decoder.parameters():
                p.requires_grad_(False)

            input_ = torch.cat([code_exp, xyz], dim=1)
            prediction = decoder(input_)

            grad_x, grad_z = torch.autograd.grad(
                prediction,
                [xyz, code_exp],
                grad_outputs=torch.ones_like(prediction),
                retain_graph=False,
                create_graph=False,
            )

            grad_x_norm = grad_x.norm(dim=1).mean()
            grad_z_norm = grad_z.norm() / N

            grad_metrics = {
                "grad_x": grad_x_norm.item(),
                "grad_z": grad_z_norm.item(),
                "grad_z_over_x_ratio": grad_z_norm.item() / (grad_x_norm.item() + 1e-12)
            }

            # unfreeze decoder params
            for p in decoder.parameters():
                p.requires_grad_(True)

            pl_module.log_dict(grad_metrics, logger=True, on_step=False, on_epoch=True)
        return









# ============================== #
# TRAINING
# ============================== #
def train(
    specs: dict,
    experiment_name,
    num_workers = 0,
    log_weights_norm_every_n_epochs = None,
    log_grad_norms_every_n_epochs = None,
    show_progress = False
):

    # region LOAD DATA 
    datamodule = SDFDataModule(
        specs = specs,
        num_workers=num_workers
    )

    datamodule.setup("fit")  

    NumScenes = datamodule.num_fit_scenes 

    # region MODEL
    decoder = Decoder(**specs["Network_specs"])

    print( decoder.description() )

    model = DeepSDF(
        decoder = decoder,
        specs = specs
    )

    model.set_embedding( num_scenes = NumScenes )

    # region LOGS
    # experiment logger
    logger_tb = TensorBoardLogger(
        save_dir = EXPERIMENTS_DIR, # All experiment folders (named by name/version) will be created inside this directory.
        name = experiment_name,
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
            raise ValueError(f"Found directory for checkpoints for wanted version: {load_version}, but contained no .ckpt files")  

    # region CALLBACKS
    callbacks = [
            SaveDecoderCallback(),
            SaveSpecsCallback(),
            SaveEmbeddingsCallback()
    ]

    # callbacks.append(  ModelCheckpoint(
    #         dirpath=checkpoint_dir,
    #         filename="epoch_{epoch}",
    #         auto_insert_metric_name=False,
    #         every_n_epochs= specs.get("checkpoint_every_n_epochs", 1000000),
    #         save_top_k=-1,
    #   )
    # )
    
    if log_weights_norm_every_n_epochs is not None:
            callbacks.append( MonitorLatentChannelsCallback(log_every_n_epochs=log_weights_norm_every_n_epochs) )
    if log_grad_norms_every_n_epochs is not None:
            callbacks.append( MonitorInputGradNormsCallback(log_every_n_epochs=log_grad_norms_every_n_epochs) )
    
    # region TRAIN
    NumEpochs = specs.get("NumEpochs", 5)

    logger_tb.log_hyperparams(specs)

    trainer = pl.Trainer(
        # strategy=None,   # for VSC5, otherwise it sets ntasks and errors, ...
        accelerator="gpu",     
        devices=1,             
        max_epochs= NumEpochs, 
        precision=PRECISION,  
        enable_model_summary=False,  
        enable_progress_bar=show_progress,
        log_every_n_steps=1,
        check_val_every_n_epoch=model.log_val_every_n_epochs if model.log_val_every_n_epochs <= 0 else None,  # run validation
        logger=logger_tb,
        enable_checkpointing=True,
        callbacks = callbacks
    )

    tic = time()
    trainer.fit(model, datamodule = datamodule, ckpt_path = ckpt_file_path)
    toc = time() - tic

    print(f"Time elapsed for fit: {toc:.2f} seconds ")

    return version_dir.name



if __name__ == "__main__":
    
    import argparse
    from pprint import pprint
    import resource

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", type=str, default = None,
                        help="Becomes the directory name in which checkpoints and logs are saved, under version_x folder for each run."
    )
    parser.add_argument("--specs_file_path", "-s", type=str, default = "specs.json")
    parser.add_argument("--train_mode", type=str, default="use_specs_file", choices=["use_specs_file", "compose_specs_from_override"])
    parser.add_argument("--override_specs", type=str, default=None)
    parser.add_argument("--num_workers_dataloader", type=int, default=0)
    parser.add_argument("--log_weights_norm_every", type=int, default=None)
    parser.add_argument("--log_grad_norms_every", type=int, default=None)
    parser.add_argument("--show_progress", action="store_true")
    args = parser.parse_args()

    log_weights_norm_every_n_epochs = args.log_weights_norm_every
    log_grad_norms_every_n_epochs = args.log_grad_norms_every

    match args.train_mode:

        case "use_specs_file":
            specs_file = SPECS_FILES_DIR / str(args.specs_file_path)
            
            if args.experiment_name is not None:
                EXPERIMENT_NAME = str(args.experiment_name)

            version = train( 
                specs = json.load(open(specs_file)),
                experiment_name = EXPERIMENT_NAME,
                num_workers=args.num_workers_dataloader,
                show_progress = args.show_progress,
                log_weights_norm_every_n_epochs=log_weights_norm_every_n_epochs,
                log_grad_norms_every_n_epochs=log_grad_norms_every_n_epochs
            )

        case "compose_specs_from_override":

            if args.experiment_name is not None:
                EXPERIMENT_NAME = str(args.experiment_name)

            specs_file = SPECS_FILES_DIR / str(args.specs_file_path)
            
            # now overwrite specs fields with wanted specs
            specs = json.load(open(specs_file))
            override_specs = json.loads(args.override_specs) # in args arriva dal bash come STRINGA json

            # be sure requested override fields are actually valid:
            check_override_specs_validity(override_specs, specs)

            specs = deep_update(specs, override_specs)

            version = train(
                specs = specs,
                experiment_name=EXPERIMENT_NAME,
                num_workers=args.num_workers_dataloader,
                show_progress = args.show_progress,
                log_weights_norm_every_n_epochs=log_weights_norm_every_n_epochs,
                log_grad_norms_every_n_epochs=log_grad_norms_every_n_epochs
            )

    # this is to then be captured from a bash file and retrieve the version that has been trained to send myself an email
    print(f"TRAINING_DONE_VERSION={version}", flush=True)
















































# """ Train deepsdf model to learn atrial shapes. Needs a specification file that gives all data, decoder, training parameters and optionally post-processing directories
# """
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import re
# import json
# import torch
# from time import time
# from pathlib import Path
# try:
#     import lightning as pl # pyright: ignore[reportMissingImports]
#     from lightning.pytorch.loggers import TensorBoardLogger # pyright: ignore[reportMissingImports]
#     from lightning.pytorch.callbacks import ModelCheckpoint # pyright: ignore[reportMissingImports]
# except ImportError:
#     import pytorch_lightning as pl
#     from pytorch_lightning.loggers import TensorBoardLogger # pyright: ignore[reportMissingImports]
#     from pytorch_lightning.callbacks import ModelCheckpoint # pyright: ignore[reportMissingImports]
    
# from model.deepsdf_dataloader import SDFDataModule
# from model.deepsdf_decoder import Decoder, DeepSDF 


# # # =========== setup for H100 / A100 / L40S GPU =========== #
# # torch.set_float32_matmul_precision("high")   # "medium" if instability (NaNs / loss spikes) appear, "high" is ignored by L40S
# # PRECISION = "bf16-mixed"
# # # next flags ignored by L40S
# # torch.backends.cuda.matmul.allow_tf32 = True
# # torch.backends.cudnn.allow_tf32 = True

# # =========== setup for RTX 3090 GPU =========== #
# torch.set_float32_matmul_precision("medium") # "medium" if instability (NaNs / loss spikes) appear
# PRECISION = "16-mixed"

# # # # =========== setup for GTX 1050 GPU =========== #
# # PRECISION = "32"

# from config import SPECS_FILES_DIR, EXPERIMENTS_DIR

# # ============================== #
# # HELPERS
# # ============================== #
# def deep_update(specs_base: dict, override_specs: dict) -> dict:
#     for k, v in override_specs.items():
#         if (
#             k in specs_base
#             and isinstance(specs_base[k], dict)
#             and isinstance(v, dict)
#         ):
#             deep_update(specs_base[k], v)
#         else:
#             specs_base[k] = v
#     return specs_base

# def flatten_dict_keys(d, parent_key="", sep="."):
#     """
#         Flatten dictionary so all keys are at the same level

#         ex. {
#                 "Network_specs" : { 
#                     "latent_size" : 64
#                     }
#                 "NumEpochs" : 100
#             }
        
#             becomes
            
#             {
#                 "Network_specs.latent_size" : 64
#                 "NumEpochs" : 100
#             }
                
#     """
#     items = {}
#     for k, v in d.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.update(flatten_dict_keys(v, new_key, sep=sep))
#         else:
#             items[new_key] = v

#     return items

# def check_override_specs_validity(override_specs, specs_base):

#     # flatten both structures to easily check even nested keys
#     override_specs = flatten_dict_keys(override_specs)
#     specs_base = flatten_dict_keys(specs_base)

#     for key in override_specs.keys():
#         if specs_base.get(key, None) is None:
#             raise ValueError(f"Requested invalid key to override: '{key}', check valid keys in specs file.")
        
#     return


# # ============================== #
# # CHECKPOINTING - manual
# # ============================== #
# class SaveDecoderCallback(pl.Callback):
#     def on_train_end(self, trainer, pl_module):
#         decoder = pl_module.decoder
#         out = (Path(trainer.log_dir) / "decoder_weights.pth")
#         torch.save(decoder.state_dict(), out)



# # ============================== #
# # TRAINING
# # ============================== #

# def train(specs: dict, experiment_name, num_workers = 0, show_progress = False):

#     # region LOAD DATA 
#     datamodule = SDFDataModule(
#         specs = specs,
#         num_workers=num_workers
#     )

#     datamodule.setup("fit")  

#     NumScenes = datamodule.num_fit_scenes 

#     # region MODEL
#     decoder = Decoder(**specs["Network_specs"])

#     print( decoder.description() )

#     model = DeepSDF(
#         decoder = decoder,
#         specs = specs
#     )

#     model.set_embedding( num_scenes = NumScenes )

#     # region CHECKPOINTS and LOGS 

#     # experiment logger
#     logger_tb = TensorBoardLogger(
#         save_dir = EXPERIMENTS_DIR, # All experiment folders (named by name/version) will be created inside this directory.
#         name = experiment_name,
#         default_hp_metric=False,
#         log_graph=False
#     )   

#     version_dir = Path(logger_tb.log_dir)
#     checkpoint_dir = version_dir / "checkpoints"    # --> created inside every version_x folder (every run)

#     # load checkpoint to resume training if wanted
#     load_version = specs.get("resume_training_from_version","-")
#     saved_ckpt_dir = Path( EXPERIMENTS_DIR /  str(logger_tb.name) / load_version / "checkpoints" )
#     ckpt_file_path = None
#     if saved_ckpt_dir.exists():
#         ckpt_files =  list( Path(saved_ckpt_dir).glob("*.ckpt") )
#         if ckpt_files: # pick last checkpoint by epoch
#             ckpt_file_path = max(ckpt_files, key = lambda file: int( re.search(r"epoch_([0-9]+)", file.name ).group(1) ) ) 
#         else:
#             raise ValueError(f"Found directory for checkpoints for wanted version: {load_version}, but contained no .ckpt files")  

#     checkpoint_callback = ModelCheckpoint(
#         dirpath=checkpoint_dir,
#         filename="epoch_{epoch}",
#         auto_insert_metric_name=False,
#         every_n_epochs= specs.get("checkpoint_every_n_epochs", 10000),
#         save_top_k=-1,
#     )

#     # region TRAIN
#     NumEpochs = specs.get("NumEpochs", 5)

#     logger_tb.log_hyperparams(specs)

#     trainer = pl.Trainer(
#         accelerator="gpu",     
#         devices=1,             
#         max_epochs= NumEpochs, 
#         precision=PRECISION,  
#         enable_model_summary=False,  
#         enable_progress_bar=show_progress,
#         log_every_n_steps=1,
#         logger=logger_tb,
#         enable_checkpointing=True,
#         callbacks = [SaveDecoderCallback(), checkpoint_callback]
#     )

#     tic = time()
#     trainer.fit(model, datamodule = datamodule, ckpt_path = ckpt_file_path)
#     toc = time() - tic

#     print(f"Time elapsed for fit: {toc:.2f} seconds ")

#     return version_dir.name



# if __name__ == "__main__":
    
#     import argparse
#     from pprint import pprint
#     import resource

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--experiment_name", "-e", type=str, default = None,
#                         help="Becomes the directory name in which checkpoints and logs are saved, under version_x folder for each run."
#     )
#     parser.add_argument("--specs_file_path", "-s", type=str, default = "specs_deepsdfatria.json")
#     parser.add_argument("--train_mode", type=str, default="use_specs_file")
#     parser.add_argument("--override_specs", type=str, default=None)
#     parser.add_argument("--num_workers_dataloader", type=int, default=0)
#     parser.add_argument("--show_progress", action="store_true")
#     args = parser.parse_args()

#     match args.train_mode:

#         case "use_specs_file":
#             specs_file = SPECS_FILES_DIR / str(args.specs_file_path)
            
#             if args.experiment_name is not None:
#                 EXPERIMENT_NAME = str(args.experiment_name)

#             version = train( 
#                 specs = json.load(open(specs_file)),
#                 experiment_name = EXPERIMENT_NAME,
#                 num_workers=args.num_workers_dataloader,
#                 show_progress = args.show_progress
#             )

#         case "compose_specs_from_options":

#             if args.experiment_name is not None:
#                 EXPERIMENT_NAME = str(args.experiment_name)

#             specs_file = SPECS_FILES_DIR / str(args.specs_file_path)
            
#             # now overwrite specs fields with wanted specs
#             specs = json.load(open(specs_file))
#             override_specs = json.loads(args.override_specs) # in args arriva dal bash come STRINGA json

#             # be sure requested override fields are actually valid:
#             check_override_specs_validity(override_specs, specs)

#             specs = deep_update(specs, override_specs)

#             version = train(
#                 specs = specs,
#                 experiment_name=EXPERIMENT_NAME,
#                 num_workers=args.num_workers_dataloader,
#                 show_progress = args.show_progress )

#     # this is to then be captured from a bash file and retrieve the version that has been trained to send myself an email
#     print(f"TRAINING_DONE_VERSION={version}", flush=True)

