import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # this should be the RTX

import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from loguru import logger

from config import DATA_DIR, PATIENTS_NPY_DATA_DIR


class SDFSamples(Dataset):

    def __init__(
            self,
            specs: json,
            stage = "train"
        ):
        super().__init__()

        self.sampling = specs.get("sampling_scene_method", "random_seed") # could be random, random_seed, all

        self.num_samp_per_scene = specs.get("num_samp_per_scene", 4096)

        self.stage = stage

        self.data_tot = []

        self.num_scenes = 69

        self.sdf_dim = specs.get("Network_specs", None).get("out_dim", 3)

        self.train_fname = DATA_DIR / specs.get("TrainSplit", None)

        self.test_fname = DATA_DIR / specs.get("TestSplit", None)

        self.data_file = ""

    def _read_data(self):

        if self.stage == "train":
            self.data_file = self.train_fname
        elif self.stage == "test":
            self.data_file = self.test_fname

        if Path(self.data_file).suffix == ".json":
            data_tot = self._unpack_sdfdata_json(self.data_file)
        
        self.num_scenes = len(data_tot)

        self.data_tot = data_tot

    def _unpack_sdfdata_json(self, data_file):
        
        data_tot = []

        loaded = json.load( open(data_file) )

        for fname in loaded:
            
            if fname.endswith(".npy"):
                dat_ = np.load(PATIENTS_NPY_DATA_DIR / fname) # names always relative to PATIENTS_NPY_DATA_DIR !!

                data_tot.append({
                    "coords": dat_[:,:3],
                    "sdf": dat_[:,3:]
                })
            else:
                raise ValueError(f"Expected path to .npy file, got {fname}")

        return data_tot
    
    def __len__(self):
        return self.num_scenes

    def __getitem__(self, index):

        data = self.data_tot[index]

        coords = data["coords"]
        sdf = data["sdf"]

        # very expensive to do every time
        if len(sdf.shape) > 1 and sdf.shape[1] > 1:
            # flatten() always returns a copy of the data, while ravel() returns a view whenever possible.
            nan_ids = np.isnan(sdf.sum(axis=1)).ravel() # collapses all SDFs for one point into a single value
        else:
            nan_ids = np.isnan(sdf).ravel()

        coords = coords[~nan_ids]
        sdf = sdf[~nan_ids]

        # this sampling can REPEAT points !!!
        if self.sampling == "random":
            random_pos = (torch.rand(self.num_samp_per_scene) * coords.shape[0]).long()
        elif self.sampling == "random_seed":
            torch.manual_seed(69) # always subsample the same points
            random_pos = (torch.rand(self.num_samp_per_scene) * coords.shape[0]).long()
        elif self.sampling == "all":
            random_pos = torch.arange(0, coords.shape[0])

        # very expensive to do every time
        coords = torch.index_select(torch.from_numpy(coords), 0, random_pos).float()
        sdf = torch.index_select(torch.from_numpy(sdf), 0, random_pos).float()
        samples = {
            "coords": coords,
            "sdf": sdf,
        }

        return samples, index 

class SDFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        specs: dict,
        num_workers = 4,
        shuffle=True,
        drop_last=True
    ):
        """
            num_workers: Number of parallel CPU workers for loading data.

            shuffle: Whether to shuffle data during training.

            drop_last: Whether to drop incomplete final batches.
        """

        super().__init__()

        self.specs = specs

        self.batch_size = self.specs.get("batch_size", 2)

        self.num_samples_per_scene = self.specs.get("num_samp_per_scene")

        self.sampling_method = self.specs.get("sampling_scene_method", "random_seed")

        self.num_workers = num_workers # not needed if data is already loaded on gpu

        self.shuffle = shuffle

        self.drop_last = drop_last

    def setup(self, stage: str):
        if stage in ["fit", "train"]:
            sdf_dataset = SDFSamples( 
                specs = self.specs,
                stage="train"
            ) 
            sdf_dataset._read_data()

            self.sdf_train = sdf_dataset

            self.num_fit_scenes = len(self.sdf_train)

        if stage in ["test", "predict"]:
            #TODO: add SDFSamplesIter option
            sdf_dataset = SDFSamples( 
                specs = self.specs,
                stage="test"
            ) 
            sdf_dataset._read_data()

            self.sdf_test = sdf_dataset

    def train_dataloader(self):
        logger.info(f"TRAIN DATA LOADED: {len(self.sdf_train)} scenes.")
        return DataLoader(
            self.sdf_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        logger.info(f"TEST DATA LOADED: {len(self.sdf_test)} scenes.")
        return DataLoader(
            self.sdf_test,
            batch_size=1,
            shuffle=False,
        )



# ================================== #
# each batch of points per scene is sampled balancing positive and negative sdfs samples
# ================================== #
class SDFBalancedSamples(Dataset):

    def __init__(
            self,
            specs: json,
            stage = "train"
        ):
        super().__init__()

        self.sampling = specs.get("sampling_scene_method", "random") # could be random, random_seed, all

        self.num_samp_per_scene = specs.get("num_samp_per_scene", 4096)

        self.pos_idxs = []

        self.neg_idxs = []

        self.stage = stage

        self.data_tot = []

        self.num_scenes = 69

        self.sdf_dim = specs.get("Network_specs", None).get("out_dim", 3)

        self.train_fname = DATA_DIR / specs.get("TrainSplit", None)

        self.test_fname = DATA_DIR / specs.get("TestSplit", None)

        self.data_file = ""

    def _read_data(self):

        if self.stage == "train":
            self.data_file = self.train_fname
        elif self.stage == "test":
            self.data_file = self.test_fname

        if Path(self.data_file).suffix == ".json":
            data_tot = self._unpack_sdfdata_json(self.data_file)
        
        self.num_scenes = len(data_tot)

        self.data_tot = data_tot

    def _unpack_sdfdata_json(self, data_file):
        
        data_tot = []

        loaded = json.load( open(data_file) )

        for fname in loaded:

            if fname.endswith(".npy"):
                dat_ = np.load(PATIENTS_NPY_DATA_DIR / fname)

                coords = dat_[:,:3]
                sdf = dat_[:,3:]

                pos_list = []
                neg_list = []

                for s in range(self.sdf_dim):
                    pos = np.where(sdf[:, s] > 0)[0]
                    neg = np.where(sdf[:, s] < 0)[0]
                    if len(pos) == 0:
                        pos = neg
                    if len(neg) == 0:
                        neg = pos

                    pos_list.append(pos)
                    neg_list.append(neg)

                # store per-scene
                self.pos_idxs.append(pos_list)
                self.neg_idxs.append(neg_list)

                data_tot.append({"coords": coords, "sdf": sdf})
        
        return data_tot
    
    def __len__(self):
        return self.num_scenes
  
    def __getitem__(self, index):

        data = self.data_tot[index]

        coords = data["coords"]
        sdf    = data["sdf"]
        
        per_surface = self.num_samp_per_scene // self.sdf_dim
        half = per_surface // 2

        # storage for selected indices
        idxs_total = []

        if self.sampling == "random":
            for s in range(self.sdf_dim):
                pos = self.pos_idxs[index][s]
                neg = self.neg_idxs[index][s]
                pos_sel = pos[np.random.randint(0, len(pos), half)]
                neg_sel = neg[np.random.randint(0, len(neg), half)]
                idxs_total.append(pos_sel)
                idxs_total.append(neg_sel)
        elif self.sampling == "random_seed":
            rng = np.random.default_rng(seed=69)
            for s in range(self.sdf_dim):
                pos = self.pos_idxs[index][s]
                neg = self.neg_idxs[index][s]
                pos_sel = pos[rng.integers(0, len(pos), half)]
                neg_sel = neg[rng.integers(0, len(neg), half)]
                idxs_total.append(pos_sel)
                idxs_total.append(neg_sel)
        elif self.sampling == "all":
            # no balancing
            idxs_total = np.arange(coords.shape[0])
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

        if self.sampling != "all":
            idxs_total = np.concatenate(idxs_total) # --> now becomes a single array of indices

        # add some replacements to get to exactly num_samp_per_scene points
        if len(idxs_total) < self.num_samp_per_scene:
            deficit = self.num_samp_per_scene - len(idxs_total)
            extra = idxs_total[np.random.randint(0, len(idxs_total), deficit)]
            idxs_total = np.concatenate([idxs_total, extra])

        coords = torch.from_numpy( coords[idxs_total] ).float()
        sdf    = torch.from_numpy( sdf[idxs_total] ).float()

        return {"coords": coords, "sdf": sdf}, index

class SDFBalancedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        specs: dict,
        num_workers = 4,
        shuffle=True,
        drop_last=True
    ):
        """
            num_workers: Number of parallel CPU workers for loading data.

            shuffle: Whether to shuffle data during training.

            drop_last: Whether to drop incomplete final batches.
        """

        super().__init__()

        self.specs = specs

        self.batch_size = self.specs.get("batch_size", 2)

        self.num_samples_per_scene = self.specs.get("num_samp_per_scene")

        self.sampling_method = self.specs.get("sampling_scene_method", "random_seed")

        self.num_workers = num_workers # not needed if data is already loaded on gpu

        self.shuffle = shuffle

        self.drop_last = drop_last

    def setup(self, stage: str):
        if stage in ["fit", "train"]:
            #TODO: handle validation correctly
            sdf_dataset = SDFBalancedSamples( 
                specs = self.specs,
                stage="train"
            ) 
            sdf_dataset._read_data()

            self.sdf_train = sdf_dataset

            self.pos_idxs = self.sdf_train.pos_idxs
            self.neg_idxs = self.sdf_train.neg_idxs

            self.num_fit_scenes = len(self.sdf_train)

        if stage in ["test", "predict"]:
            #TODO: add SDFSamplesIter option
            sdf_dataset = SDFBalancedSamples( 
                specs = self.specs,
                stage="test"
            ) 
            sdf_dataset._read_data()

            self.sdf_test = sdf_dataset
            self.test_pos_idxs = self.sdf_test.pos_idxs
            self.test_neg_idxs = self.sdf_test.neg_idxs

            self.num_test_scenes = len(self.sdf_test)

    def train_dataloader(self):
        logger.info(f"TRAIN DATA LOADED: {len(self.sdf_train)} scenes.")
        return DataLoader(
            self.sdf_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        logger.info(f"TEST DATA LOADED: {len(self.sdf_test)} scenes.")
        return DataLoader(
            self.sdf_test,
            batch_size=1,
            shuffle=False,
        )



# ================================== #
# to load all data in GPU first
# ================================== #
class IndicesDataset(Dataset):
    def __init__(self, num_scenes):
        self.num_scenes = num_scenes

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        return idx
    
class SDFDataModuleGPU(pl.LightningDataModule):
    
    def __init__(
        self,
        specs: dict,
        num_workers = 4,
        shuffle=True,
        drop_last=True
    ):
        """
            num_workers: Number of parallel CPU workers for loading data.

            shuffle: Whether to shuffle data during training.

            drop_last: Whether to drop incomplete final batches.
        """

        super().__init__()

        self.specs = specs

        self.batch_size = self.specs.get("batch_size", 16)

        self.num_samples_per_scene = self.specs.get("num_samp_per_scene")

        self.sampling_method = self.specs.get("sampling_scene_method", "random_seed")

        self.num_workers = num_workers # not needed if data is already loaded on gpu

        self.shuffle = shuffle

        self.drop_last = drop_last
    
    def setup(self, stage=None):
        if stage in ["fit", "train", "fit_and_val"]:
            # Carichi i dati come prima
            sdf_train = SDFSamples(
                specs=self.specs,
                stage="train"
            )
            sdf_train._read_data()

            # Qui: sposti tutto su GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.coords_gpu = []
            self.sdf_gpu = []

            for scene in sdf_train.data_tot:
                coords = torch.from_numpy(scene["coords"]).float().to(device)
                sdf = torch.from_numpy(scene["sdf"]).float().to(device)
                self.coords_gpu.append(coords)   
                self.sdf_gpu.append(sdf)         

            # Se vuoi: trasformale in tensori singoli (lista -> torch tensor raggruppato)
            self.coords_gpu = torch.stack(self.coords_gpu)
            self.sdf_gpu = torch.stack(self.sdf_gpu)

            self.num_fit_scenes = len(sdf_train)

            if stage == "fit_and_val":
                # load also validation data, from test file actually ...
                sdf_val = SDFSamples(
                    specs=self.specs,
                    stage="test"
                )
                sdf_val._read_data()

                self.sdf_val = sdf_val

        if stage in ["test", "predict"]:
            #TODO: add SDFSamplesIter option
            sdf_dataset = SDFSamples( 
                specs = self.specs,
                stage="test"
            ) 
            sdf_dataset._read_data()

            self.sdf_test = sdf_dataset

    def train_dataloader(self):
        indices = IndicesDataset(self.num_fit_scenes)
        logger.info(f"TRAIN DATA LOADED: {len(indices)} scenes.")
        return DataLoader(
            indices,  
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # 0 perché tutto è già in GPU
            pin_memory=False
        )
    
    def test_dataloader(self):
        logger.info(f"TEST DATA LOADED: {len(self.sdf_test)} scenes.")
        return DataLoader(
            self.sdf_test,
            batch_size=1,
            shuffle=False,
        )
    
    def val_dataloader(self):
        logger.info(f"VALIDATION DATA LOADED: {len(self.sdf_val)} scenes.")
        return DataLoader(
            self.sdf_val,
            batch_size=1,
            shuffle=False,
        )
         
    
class SDFBalancedDataModuleGPU(pl.LightningDataModule):
    
    def __init__(
        self,
        specs: dict,
        num_workers = 4,
        shuffle=True,
        drop_last=True
    ):
        """
            num_workers: Number of parallel CPU workers for loading data.

            shuffle: Whether to shuffle data during training.

            drop_last: Whether to drop incomplete final batches.
        """

        super().__init__()

        self.specs = specs

        self.batch_size = self.specs.get("batch_size", 16)

        self.num_samples_per_scene = self.specs.get("num_samp_per_scene")

        self.sampling_method = self.specs.get("sampling_scene_method", "random_seed")

        self.num_workers = num_workers # not needed if data is already loaded on gpu

        self.shuffle = shuffle

        self.drop_last = drop_last
    
    def setup(self, stage=None):
        if stage in ["fit", "train", "fit_and_val"]:
            # Carichi i dati come prima
            sdf_train = SDFBalancedSamples(
                specs=self.specs,
                stage="train"
            )
            sdf_train._read_data() # --> now I also have pos_idxs and neg_idxs

            # Tutto su GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.coords_gpu = []
            self.sdf_gpu = []
            self.pos_idx_gpu = []
            self.neg_idx_gpu = []

            for i,scene in enumerate(sdf_train.data_tot):
                coords = torch.from_numpy(scene["coords"]).float().to(device)
                sdf = torch.from_numpy(scene["sdf"]).float().to(device)
                self.coords_gpu.append(coords)   
                self.sdf_gpu.append(sdf)         
                # store also pos and neg idxs on gpu
                # convert pos and neg indices to tensors on the GPU
                pos_idx_scene = [torch.tensor(idx, dtype=torch.long, device=device) for idx in sdf_train.pos_idxs[i]]
                neg_idx_scene = [torch.tensor(idx, dtype=torch.long, device=device) for idx in sdf_train.neg_idxs[i]]
                self.pos_idx_gpu.append(pos_idx_scene)
                self.neg_idx_gpu.append(neg_idx_scene)

            self.coords_gpu = torch.stack(self.coords_gpu)
            self.sdf_gpu = torch.stack(self.sdf_gpu)

            self.num_fit_scenes = len(sdf_train)

            if stage == "fit_and_val":
                # load also validation data, from test file for now ...
                # not on GPU
                sdf_val = SDFBalancedSamples(
                    specs=self.specs,
                    stage="test"
                )
                sdf_val._read_data()

                self.sdf_val = sdf_val



        if stage in ["test", "predict"]:
            #TODO: add SDFSamplesIter option
            sdf_dataset = SDFBalancedSamples( 
                specs = self.specs,
                stage="test"
            ) 
            sdf_dataset._read_data()

            self.sdf_test = sdf_dataset

    def train_dataloader(self):
        indices = IndicesDataset(self.num_fit_scenes)
        logger.info(f"TRAIN DATA LOADED: {len(indices)} scenes.")
        return DataLoader(
            indices,  
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # 0 perché tutto è già in GPU
            pin_memory=False
        )
    
    def test_dataloader(self):
        logger.info(f"TEST DATA LOADED: {len(self.sdf_test)} scenes.")
        return DataLoader(
            self.sdf_test,
            batch_size=1,
            shuffle=False,
        )
    




if __name__ == "__main__":

    pass
