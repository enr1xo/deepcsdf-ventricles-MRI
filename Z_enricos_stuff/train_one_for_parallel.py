#!/usr/bin/env python3
"""
Train one DeepSDF experiment from a single combination folder.

Expected combination structure:
    combo_dir/
    ├── config.py
    ├── specs_files/
    │   └── specs.json
    ├── train/
    │   └── data_fnames_train.json
    ├── test/
    │   └── data_fnames_test.json
    ├── experiments/
    └── results/
"""

from __future__ import annotations

#================== debug
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
#================================

import argparse
import importlib.util
import json
import re
from pathlib import Path
from time import time

import numpy as np
import torch

try:
    import lightning as pl
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint



from model.deepsdf_dataloader import SDFDataModule
from model.deepsdf_decoder import Decoder, DeepSDF


# =========== setup for H100 / A100 / L40S GPU =========== #
torch.set_float32_matmul_precision("high")
PRECISION = "bf16-mixed"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================== #
# HELPERS
# ============================== #
def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict_keys(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def check_override_specs_validity(override_specs, specs_base):
    override_specs = flatten_dict_keys(override_specs)
    specs_base = flatten_dict_keys(specs_base)

    for key in override_specs.keys():
        if specs_base.get(key, None) is None:
            raise ValueError(f"Requested invalid key to override: '{key}'")

    return


# ============================== #
# CALLBACKS
# ============================== #
class SaveDecoderCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        decoder = pl_module.decoder
        out = Path(trainer.log_dir) / "decoder_weights.pth"
        torch.save(decoder.state_dict(), out)


class SaveSpecsCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.logger is not None:
            json_path = Path(trainer.logger.log_dir) / "hparams.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pl_module.specs, f, indent=4)


class SaveEmbeddingsCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        npy_path = Path(trainer.log_dir) / "latents.npy"
        embeddings = pl_module.lat_vecs["trainable"].weight.data.cpu().numpy()
        np.save(npy_path, embeddings)


# ============================== #
# TRAINING
# ============================== #
def train_one(
    specs: dict,
    experiment_name: str,
    experiments_dir: Path,
    num_workers: int = 0,
    show_progress: bool = False,
):
    datamodule = SDFDataModule(
        specs=specs,
        num_workers=num_workers,
    )
    datamodule.setup("fit")

    num_scenes = datamodule.num_fit_scenes

    decoder = Decoder(**specs["Network_specs"])
    #debug
    # print("\n\n DEBUG\n\n")
    # print(decoder.description())clear
    

    model = DeepSDF(
        decoder=decoder,
        specs=specs,
    )
    model.set_embedding(num_scenes=num_scenes)

    logger_tb = TensorBoardLogger(
        save_dir=experiments_dir,
        name=experiment_name,
        default_hp_metric=False,
        log_graph=False,
    )

    version_dir = Path(logger_tb.log_dir)
    checkpoint_dir = version_dir / "checkpoints"

    load_version = specs.get("resume_training_from_version", "-")
    saved_ckpt_dir = experiments_dir / str(logger_tb.name) / load_version / "checkpoints"

    ckpt_file_path = None
    if saved_ckpt_dir.exists():
        ckpt_files = list(saved_ckpt_dir.glob("*.ckpt"))
        if ckpt_files:
            ckpt_file_path = max(
                ckpt_files,
                key=lambda file: int(re.search(r"epoch_([0-9]+)", file.name).group(1))
            )
        else:
            raise ValueError(
                f"Found checkpoint directory for wanted version '{load_version}', "
                "but it contains no .ckpt files."
            )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch}",
        auto_insert_metric_name=False,
        every_n_epochs=specs.get("checkpoint_every_n_epochs", 1000000),
        save_top_k=-1,
    )

    num_epochs = specs.get("NumEpochs", 5)

    logger_tb.log_hyperparams(specs)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=num_epochs,
        precision=PRECISION,
        enable_model_summary=False,
        enable_progress_bar=show_progress,
        log_every_n_steps=1,
        check_val_every_n_epoch=model.log_val_every_n_epochs,
        logger=logger_tb,
        enable_checkpointing=True,
        callbacks=[
            SaveDecoderCallback(),
            SaveSpecsCallback(),
            SaveEmbeddingsCallback(),
            checkpoint_callback,
        ],
    )

    tic = time()
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_file_path)
    toc = time() - tic

    print(f"Time elapsed for fit: {toc:.2f} seconds")

    return version_dir.name


def resolve_specs_paths(specs: dict, combo_dir: Path) -> dict:
    """
    Make TrainSplit / TestSplit absolute if they are relative.
    """
    specs = dict(specs)

    for key in ["TrainSplit", "TestSplit"]:
        if key in specs:
            p = Path(specs[key])
            if not p.is_absolute():
                specs[key] = str((combo_dir / p).resolve())

    return specs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combo_dir", required=True, type=Path,
        help="Path to one combination folder"
    )
    parser.add_argument(
        "--specs_name", type=str, default="specs.json",
        help="Specs filename inside combo_dir/specs_files/"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Optional override for experiment name"
    )
    parser.add_argument(
        "--override_specs", type=str, default=None,
        help='JSON string with specs overrides'
    )
    parser.add_argument(
        "--num_workers_dataloader", type=int, default=0
    )
    parser.add_argument(
        "--show_progress", action="store_true"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    combo_dir = args.combo_dir.resolve()
    if not combo_dir.exists():
        raise FileNotFoundError(f"Combination directory does not exist: {combo_dir}")

    config_path = combo_dir / "config.py"
    specs_path = combo_dir / "specs_files" / args.specs_name
    experiments_dir = combo_dir / "experiments"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.py: {config_path}")

    if not specs_path.exists():
        raise FileNotFoundError(f"Missing specs file: {specs_path}")

    experiments_dir.mkdir(parents=True, exist_ok=True)

    combo_config = load_module_from_path("combo_config", config_path)

    with open(specs_path, "r", encoding="utf-8") as f:
        specs = json.load(f)

    specs = resolve_specs_paths(specs, combo_dir)

    if args.override_specs is not None:
        override_specs = json.loads(args.override_specs)
        check_override_specs_validity(override_specs, specs)
        specs = deep_update(specs, override_specs)

    experiment_name = args.experiment_name if args.experiment_name is not None else combo_dir.name

    print(f"Combination dir: {combo_dir}")
    print(f"Using specs: {specs_path}")
    print(f"Using experiments dir: {experiments_dir}")
    print(f"Using config: {config_path}")
    print(f"Using train split: {specs['TrainSplit']}")
    print(f"Using test split: {specs['TestSplit']}")
    print(f"Using npy dir: {combo_config.PATIENTS_NPY_DATA_DIR}")

    version = train_one(
        specs=specs,
        experiment_name=experiment_name,
        experiments_dir=experiments_dir,
        num_workers=args.num_workers_dataloader,
        show_progress=args.show_progress,
    )

    print(f"TRAINING_DONE_VERSION={version}", flush=True)


if __name__ == "__main__":
    main()