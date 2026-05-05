import os
import torch
try:
    import lightning as pl # pyright: ignore[reportMissingImports]
except ImportError:
    import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

from config import DATA_DIR, PATIENTS_NPY_DATA_DIR


class SDFSamples(Dataset):

    def __init__(
            self,
            specs: json,
            stage = "train"
        ):
        super().__init__()

        self.sampling = specs.get("sampling_scene_method", "random") # could be random, random_seed, all

        self.num_samp_per_scene = specs.get("num_samp_per_scene", 4096)

        self.balance_pos_neg = specs.get("balance_pos_neg", False)

        self.pos_idxs = []

        self.neg_idxs = []

        self.stage = stage

        self.data_tot = []

        self.num_scenes = -1

        self.sdf_dim = specs.get("Network_specs", {}).get("out_dim", 3)

        # # old and correct dataloader uses these 4
        # self.train_fname = DATA_DIR / specs.get("TrainSplit", None) # I don't like DATA_DIR hardcoded actually

        # self.test_fname = DATA_DIR / specs.get("TestSplit", None)

        # self.val_fname = DATA_DIR / specs.get("ValSplit", self.test_fname)

        # self.data_file = ""
        # # end of old dataloader

        # new dataloader for parallel training
        # Use paths from specs if provided, otherwise fall back to old config-based behaviour
        train_split = specs.get("TrainSplit", None)
        test_split = specs.get("TestSplit", None)
        val_split = specs.get("ValSplit", test_split)

        self.train_fname = Path(train_split) if train_split is not None else None
        self.test_fname = Path(test_split) if test_split is not None else None
        self.val_fname = Path(val_split) if val_split is not None else self.test_fname

        # If relative paths are used, keep backward compatibility with old workflow
        if self.train_fname is not None and not self.train_fname.is_absolute():
            self.train_fname = DATA_DIR / self.train_fname
        if self.test_fname is not None and not self.test_fname.is_absolute():
            self.test_fname = DATA_DIR / self.test_fname
        if self.val_fname is not None and not self.val_fname.is_absolute():
            self.val_fname = DATA_DIR / self.val_fname

        # DataSource for the npy files
        data_source = specs.get("DataSource", None)
        if data_source is not None:
            self.data_source = Path(data_source)
        else:
            self.data_source = PATIENTS_NPY_DATA_DIR

        if not self.data_source.exists():
            raise FileNotFoundError(f"DataSource directory does not exist: {self.data_source}")

        self.data_file = ""
        # end od modification for parallel

    def _read_data(self):

        if self.stage == "train":
            self.data_file = self.train_fname
        elif self.stage == "test":
            self.data_file = self.test_fname

        if Path(self.data_file).suffix == ".json":
            data_tot = self._unpack_sdfdata_json(self.data_file)
        else:
            raise ValueError(f"Expected path to .json file listing data, got {self.data_file}")
        
        self.num_scenes = len(data_tot)

        self.data_tot = data_tot

        # convert it already to torch
        for i in range(self.num_scenes):
            self.data_tot[i]["coords"] = torch.from_numpy(self.data_tot[i]["coords"]).float()
            self.data_tot[i]["sdf"]    = torch.from_numpy(self.data_tot[i]["sdf"]).float()

        if self.balance_pos_neg: # convert also these to torch already
            self.pos_idxs = [
                [torch.as_tensor(p, dtype=torch.long) for p in scene]
                for scene in self.pos_idxs
            ]
            self.neg_idxs = [
                [torch.as_tensor(n, dtype=torch.long) for n in scene]
                for scene in self.neg_idxs
            ]

    def _unpack_sdfdata_json(self, data_file):
        
        data_file = Path(data_file).resolve()
        print("\n[DEBUG] data_file used:", data_file)
        print("[DEBUG] data_source used:", self.data_source.resolve())

        loaded = json.load(open(data_file))

        print("[DEBUG] contains 20000?", any("20000" in x for x in loaded))
        print("[DEBUG] contains 5000?", any("5000" in x for x in loaded))

        print("[DEBUG] first 10 entries:")
        for x in loaded[:10]:
            print("   ", x)

        data_tot = []

        loaded = json.load( open(data_file) )

        for fname in loaded:
            # # DEBUG
            # print(loaded)
            # break
            # # fine debug
            
            if fname.endswith(".npy"):
                #old dataloader
                # dat_ = np.load( PATIENTS_NPY_DATA_DIR / fname) # names always relative to PATIENTS_NPY_DATA_DIR !!
                # end olddataloader

                #new dataloader
                dat_ = np.load(self.data_source / fname)
                #end new dataloader
                data_tot.append({
                    "coords": dat_[:,:3],
                    "sdf": dat_[:,3:]
                })

                if self.balance_pos_neg: # build also lists of positive / negative sdf indexes per scene
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
    
            else:
                raise ValueError(f"Expected path to .npy file, got {fname}")
            
        return data_tot

    def balance_batch(self, index, coords, sdf):
        per_surface = self.num_samp_per_scene // self.sdf_dim
        half = per_surface // 2

        idxs_total = []

        if self.sampling == "random":
            for s in range(self.sdf_dim):
                pos = self.pos_idxs[index][s]
                neg = self.neg_idxs[index][s]
                pos_sel = pos[torch.randint(0, len(pos), (half,))]
                neg_sel = neg[torch.randint(0, len(neg), (half,))]
                idxs_total.append(pos_sel)
                idxs_total.append(neg_sel)
        elif self.sampling == "random_seed":
            g = torch.Generator()
            g.manual_seed(69 + index)
            for s in range(self.sdf_dim):
                pos = self.pos_idxs[index][s]
                neg = self.neg_idxs[index][s]
                pos_sel = pos[torch.randint(0, len(pos), (half,), generator=g)]
                neg_sel = neg[torch.randint(0, len(neg), (half,), generator=g)]
                idxs_total.append(pos_sel)
                idxs_total.append(neg_sel)
        elif self.sampling == "all":
            idxs_total = torch.arange(coords.shape[0])
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

        if self.sampling != "all":
            idxs_total = torch.cat(idxs_total)
            # add more samples if num_samp_per_scene isnìt reached already
            if idxs_total.numel() < self.num_samp_per_scene:
                deficit = self.num_samp_per_scene - idxs_total.numel()
                extra = idxs_total[
                    torch.randint(0, idxs_total.numel(), (deficit,))
                ]
                idxs_total = torch.cat([idxs_total, extra])

        coords = coords[idxs_total]
        sdf    = sdf[idxs_total]

        return coords, sdf

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, index):
        
        # TODO: optional: return BALANCED pos/neg sdf samples
        data = self.data_tot[index]

        coords = data["coords"]
        sdf = data["sdf"]

        if self.balance_pos_neg:
            coords, sdf = self.balance_batch(index,coords,sdf)
        else:
            # this sampling CAN REPEAT points !!!
            if self.sampling == "random":
                random_pos = torch.randint(0, coords.shape[0], (self.num_samp_per_scene,))
            elif self.sampling == "random_seed":
                g = torch.Generator()
                g.manual_seed(69 + index) # subsample always same points for each scene
                random_pos = torch.randint(0, coords.shape[0], (self.num_samp_per_scene,), generator=g)
            elif self.sampling == "all":
                random_pos = torch.arange(0, coords.shape[0])

            coords = coords[random_pos]
            sdf    = sdf[random_pos]

        samples = {
            "coords": coords,
            "sdf": sdf,
        }

        return samples, index 

class SDFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        specs: dict,
        num_workers = 23,
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

        self.val_batch_size = self.specs.get("val_batch_size", 2)

        self.sampling_method = self.specs.get("sampling_scene_method", "random_seed")

        self.num_workers = num_workers 

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

            self.num_samples_per_scene = self.sdf_train.num_samp_per_scene

            sdf_dataset = SDFSamples( 
                specs = self.specs,
                stage="test"
            ) 
            sdf_dataset._read_data()

            self.sdf_val = sdf_dataset

            self.num_fit_scenes_val = len(self.sdf_val)

            self.num_samples_per_scene_val = self.sdf_val.num_samp_per_scene

        if stage in ["test"]:
            #TODO: add loading partial files like h5 option
            sdf_dataset = SDFSamples( 
                specs = self.specs,
                stage="test"
            ) 
            sdf_dataset._read_data()

            self.sdf_test = sdf_dataset

            self.num_test_scenes = len(self.sdf_test)

            self.num_samples_per_scene = self.sdf_test.num_samp_per_scene

    def train_dataloader(self):
        if self.sdf_train.balance_pos_neg:
            opt = f"Using balanced pos/neg scenes in training"
        else:
            opt=""
        print(f"TRAIN DATA LOADED: {len(self.sdf_train)} scenes. num_samp_per_scene = {self.num_samples_per_scene}. " + opt)
        return DataLoader(
            self.sdf_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        print(f"TEST DATA LOADED: {len(self.sdf_test)} scenes. Default num_samp_per_scene = {self.num_samples_per_scene}.")
        return DataLoader(
            self.sdf_test,
            batch_size=self.batch_size, # hard coded for now !!
            shuffle=False,
        )

    def val_dataloader(self):
        print(f"VALIDATION DATA LOADED: {len(self.sdf_val)} scenes. Default num_samp_per_scene = {self.num_samples_per_scene_val}.")
        return DataLoader(
            self.sdf_val,
            batch_size=self.val_batch_size, 
            shuffle=False, # I get always the same scene 
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )




if __name__ == "__main__":

    pass
