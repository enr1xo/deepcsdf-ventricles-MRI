import torch
import lightning as pl
import json
import torch.nn as nn
import torch.nn.functional as F
import math
from loguru import logger
from pathlib import Path
from deel.torchlip.modules.linear import SpectralLinear

act_fn = {
    "ReLU": nn.ReLU(), "Tanh" : nn.Tanh(), 
    "Softplus": nn.Softplus(beta=100), 
    "SiLU": nn.SiLU(),
    "GELU" : nn.GELU(), 
    "GELUApprox": nn.GELU(approximate="tanh")
}


class Decoder(nn.Module):

    def __init__(self, **specs): # was **specs
        super().__init__()

        self.latent_size = specs.get("latent_size", 64)

        self.out_dim = specs.get("out_dim", 1)

        self.use_positional_encoding = specs.get("positional_encoding", False)
        
        self.pos_enc_dim = specs.get("pos_enc_dim", 10)

        self.latent_in = specs.get("latent_in", [-1])
        
        self.norm_layers = specs.get("norm_layers", [-1])
        
        self.dropout_prob = specs.get("dropout_prob", 0.2)
        
        self.dropout = specs.get("dropout_layers", [-1])

        self.activation = specs.get("activation", "ReLU")

        self.batch_norm = specs.get("batch_norm", False)

        self.lipschits_layers = specs.get("lipschits_layers", [-1])

        self.hidden_dims = specs.get("dims", [])
        
        self.check_decoder_specs_validity()
            
        if self.use_positional_encoding:
            self.dims = [self.latent_size + 3 + 3 * self.pos_enc_dim * 2] + self.hidden_dims + [self.out_dim]
        else:
            self.dims = [self.latent_size + 3] + self.hidden_dims + [self.out_dim]

        self.num_layers = len(self.dims)

        # .float() Converts all the parameters and buffers in that layer to double precision instead of default float32
        for layer in range(0, self.num_layers - 1): 
            if layer + 1 in self.latent_in:
                out_dim_ = self.dims[layer + 1] - self.dims[0] # if using positional encoding, this might be 0 if first layer becomes too big --> cannot use very large positional encoding !!!
            else:
                out_dim_ = self.dims[layer + 1]

            if layer in self.norm_layers:
                lin = nn.utils.weight_norm(nn.Linear(self.dims[layer], out_dim_)).float()
            elif layer in self.lipschits_layers:
                lin = SpectralLinear(self.dims[layer], out_dim_).float()
            else:
                lin = nn.Linear(self.dims[layer], out_dim_).float()

            setattr(self, f"lin{layer}", lin)

        self.act = act_fn.get(self.activation, "Softplus")

        self.last_tanh = specs.get("last_tanh", False)

    def check_decoder_specs_validity(self):
        # check that if concatenation is wanted, the input layer should be smaller that the layer in which to concatenate it
        # (the layer will be of dimension hidden_dim still, so if input_layer_dim > hidden_dim, there is not enough length to cut it in)
        #     lat_in = self.latent_in[0]
        #     if lat_in != -1:
        #         if self.dims[0] > self.dims[self.latent_in[0]+1]:
                     
        if self.latent_in[0] != -1:
            inp_dim = self.latent_size + 3 

            if self.use_positional_encoding:
                inp_dim += (3 * self.pos_enc_dim * 2)
            
            if inp_dim > self.hidden_dims[self.latent_in[0]]:
                msg = (
                    f"Toggling off shortcut connection: requested input concatenation to hidden layer "
                    f"{self.latent_in[0]}, but hidden layer has dimension {self.hidden_dims[self.latent_in[0]]}, "
                    f"while input layer is {inp_dim}."
                )
                if self.use_positional_encoding:
                    msg += " Positional encoding dimension may be the cause."

                logger.warning(msg)

                self.latent_in = [-1]
        
        return
    
    # input: N x (pos_enc_dim + 3) ??
    def forward(self, input_):
        x = input_

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input_], 1) # concatenating whole input? not only latent code?
            x = lin(x)

            # last layer Tanh
            if layer < self.num_layers - 2:
                if self.batch_norm:
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.act(x)
                if layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.last_tanh:
            x = torch.tanh(x)

        return x

    def description(self):
        desc = f"Decoder network:"

        f = f" \n {self.num_layers} layers with channels {self.dims} "
        if self.lipschits_layers != [-1]:
            f = f + f"\n Using Lipschitz constrained linear layers in {[i+1 for i in self.lipschits_layers]}"
        if self.use_positional_encoding:
            f = f + f"\n Using positional encoding of dimension {self.pos_enc_dim} on input."
        if self.latent_in != [-1]:
            f = f + f"\n Shortcut connection of input to hidden layer {self.latent_in[0]}"
        f = f + f"\n Using latent dimension {self.latent_size}."
        f = f + f"\n Activations: {self.activation}"
        if self.last_tanh:
            f = f + ", tanh on output."
        
        return desc + f

class DeepSDF(pl.LightningModule):
    def __init__(self, decoder, specs):
        super().__init__()

        self.specs = specs
        
        self.decoder: Decoder = decoder
        
        self.lr_weights = specs.get("lr_weights", 0.001)
        self.lr_latents = specs.get("lr_latents", 0.0005)

        self.use_lr_scheduler = specs.get("use_lr_scheduler", False)
        if self.use_lr_scheduler:
            self.lr_weights_final = specs.get("lr_weights_final", 0.001)
            self.lr_latents_final = specs.get("lr_latents_final", 0.0005)

        self.latent_size = decoder.latent_size
        
        self.code_reg_lambda = specs.get("code_reg_lambda", 1e-2)

        self.lat_vecs = {"trainable": nn.Embedding}

        self.use_loss = specs.get("use_loss", "L1Loss")

        if self.use_loss == "L1Loss":
            self.loss_fn = torch.nn.L1Loss(reduction="sum")  #! was reduction="sum"
        elif self.use_loss == "MSELoss":
            self.loss_fn = torch.nn.MSELoss(reduction="sum")
        else:
            self.loss_fn = torch.nn.L1Loss(reduction="sum") 

        self.use_positional_encoding = decoder.use_positional_encoding

        self.pos_enc_dim = decoder.pos_enc_dim

        if self.use_positional_encoding:
            self.register_buffer("freqs", 2.0 ** torch.arange(self.pos_enc_dim))    # to not compute it every time

        self.enforce_minmax = specs.get("enforce_minmax", False)

        clamp_distance = specs.get("clamp_distance", 0.1)

        self.Cs = specs.get("scale_spatial_inputs_by", 100)

        self.lipschits_regularization = self.decoder.lipschits_layers[0] != -1
        self.lipschits_alpha = specs.get("lipschitz_alpha", 2e-6)

    def set_embedding(self, num_scenes = None, embedding=None):
        if num_scenes is None:
            raise ValueError("Must define number of scenes to create correct number of embeddings.")
        
        # if num_scenes != (len(self.sdf_train) + len(self.sdf_val)):
        #     raise ValueError("Setting number of embeddings different than total number of loaded scenes (train + validation datasets) !")
        
        self.num_scenes = num_scenes

        self.lat_vecs["trainable"] = nn.Embedding(
            self.num_scenes,
            self.latent_size,
            dtype=torch.float32
        )

        if embedding is not None:
            self.lat_vecs["trainable"].weight.data = torch.as_tensor(
                embedding, dtype=torch.float32
            )
        else:
            torch.nn.init.normal_(
                self.lat_vecs["trainable"].weight.data,
                0.0,
                1.0 / math.sqrt(self.latent_size),
            )

    def positional_encoding(self, xyz):
        freqs = self.freqs
        xyz_expanded = xyz[..., None, :] * freqs[:, None] * torch.pi / self.Cs #--> remove spatial scaling for sin and cos computation
        encoded = torch.cat([torch.sin(xyz_expanded), torch.cos(xyz_expanded)], dim=-1)
        return torch.cat([xyz, encoded.reshape(*xyz.shape[:-1], -1)], dim=-1)
    
    def configure_optimizers(self):
        if "trainable" in self.lat_vecs.keys():
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.decoder.parameters(),
                        "lr": self.lr_weights,
                        # "lr": self.lr_schedules[0].get_learning_rate(0),
                    },
                    {
                        "params": self.lat_vecs["trainable"].parameters(),
                        "lr": self.lr_latents #0.0005,  # was 0.0005
                        # "lr": self.lr_schedules[1].get_learning_rate(0),
                    },
                ]
            )
            if self.use_lr_scheduler:
                T_max = 100000

                def cosine_lambda(epoch, base_lr, eta_min):
                    return eta_min / base_lr + 0.5 * (1 - eta_min / base_lr) * (1 + math.cos(math.pi * epoch / T_max))

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=[
                        lambda epoch: cosine_lambda(epoch, self.lr_weights, self.lr_weights_final),
                        lambda epoch: cosine_lambda(epoch, self.lr_latents, self.lr_latents_final)
                    ]
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1
                    }
                }
        else:
            optimizer = torch.optim.Adam(params=self.decoder.parameters(), lr=self.lr_weights)

            if self.use_lr_scheduler:
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=10, eta_min=self.lr_weights_final
                    ),
                    "interval": "epoch",  # step per epoch
                    "frequency": 1,
                    "name": "lr_decoder"
                }

                return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    def on_fit_start(self):
        # move manually latents on the device to be sure it's on the same device as data when fitting the model
        self.lat_vecs["trainable"].to(self.device)
        return super().on_fit_start()
    
    def on_train_start(self):
        # d = flatten_dict_for_logging(self.specs)
        # self.logger.log_hyperparams(d)
        # self.logger.log_hyperparams({"hparams_json": json.dumps(self.specs)}) # --> only for mlflow
        if self.logger is not None:
            json_path = Path(self.logger.log_dir) / "specs.json"
            with open(json_path, "w") as f:
                json.dump(self.specs, f, indent=4)

    def training_step(self, batch, batch_idx):

        data = batch[0]
        indices = batch[1]

        coords = data["coords"].to(self.device)
        sdf_gt = data["sdf"].to(self.device)

        num_samp_per_scene = data["coords"].shape[1]

        xyz = coords.reshape(-1, 3) * 100
        if self.use_positional_encoding:
            xyz = self.positional_encoding(xyz)

        sdf_gt = sdf_gt.reshape(-1, self.decoder.out_dim)
        if self.enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, min = -self.clamp_distance, max=self.clamp_distance)

        num_sdf_samples = sdf_gt.shape[0]

        indices.requires_grad = False
        xyz.requires_grad = False
        sdf_gt.requires_grad = False

        indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)

        batch_vecs = self.lat_vecs["trainable"](indices)

        input_ = torch.cat([batch_vecs, xyz], dim=1)

        # NN optimization
        prediction = self.decoder(input_)
        if self.enforce_minmax:
            prediction = torch.clamp(prediction, min = -self.clamp_distance, max=self.clamp_distance)
        
        # Regression loss
        chunk_loss = self.loss_fn(prediction, sdf_gt) / num_sdf_samples

        # LIPLOSS
        lipschits_loss = 1.0
        if self.lipschits_regularization:
            for layer in self.decoder.lipschits_layers:
                weight = self.decoder.__getattr__("lin" + str(layer)).weight
                norm = torch.linalg.matrix_norm(weight, ord=float("inf"))
                lipschits_loss *= F.softplus(norm)
            lipschits_loss = self.lipschits_alpha * lipschits_loss

        # Regularization LOSS
        reg_loss = self.code_reg_lambda * torch.sum( torch.linalg.norm(batch_vecs, dim=1) ) / num_sdf_samples
        
        training_loss = chunk_loss + reg_loss + lipschits_loss

        if self.logger is not None:
            self.log_dict(
                {
                    "reg_loss": reg_loss,
                    "l1_loss": chunk_loss,
                    "lipschits_loss": lipschits_loss,
                },
                logger=True
            )

            self.log_dict({"train_loss": training_loss}, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return training_loss


class DeepSDFGPU(pl.LightningModule):
    def __init__(self, decoder, specs):
        super().__init__()

        self.specs = specs
        
        self.decoder: Decoder = decoder
        
        # self.lr_schedules = get_learning_rate_schedules()

        self.lr_weights = specs.get("lr_weights", 0.001)
        self.lr_latents = specs.get("lr_latents", 0.0005)

        self.use_lr_scheduler = specs.get("use_lr_scheduler", False)
        if self.use_lr_scheduler:
            self.lr_weights_final = specs.get("lr_weights_final", 0.001)
            self.lr_latents_final = specs.get("lr_latents_final", 0.0005)

        self.latent_size = decoder.latent_size
        
        self.code_reg_lambda = specs.get("code_reg_lambda", 1e-2)

        self.lat_vecs = {"trainable": nn.Embedding}

        self.loss_l1 = torch.nn.L1Loss(reduction="sum")  #! was reduction="sum"
        # self.loss_fn = torch.nn.MSELoss(reduction="sum")  #! was reduction="sum"
        # self.loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
        # self.loss_cos = torch.nn.CosineSimilarity(dim=-1)

        self.use_positional_encoding = decoder.use_positional_encoding

        self.pos_enc_dim = decoder.pos_enc_dim

        if self.use_positional_encoding:
            self.register_buffer("freqs", 2.0 ** torch.arange(self.pos_enc_dim))    # This removes millions of sin/cos ops per step.

        self.enforce_minmax = specs.get("enforce_minmax", False)

        clamp_distance = specs.get("clamp_distance", 0.1)
        self.minT = -clamp_distance
        self.maxT = +clamp_distance

        self.lipschits_regularization = self.decoder.lipschits_layers[0] != -1
        self.lipschits_alpha = specs.get("lipschitz_alpha", 2e-6)

        self.test_on_train = True

    def set_embedding(self, num_scenes = None, embedding=None):
        if num_scenes is None:
            raise ValueError("Must define number of scenes to create correct number of embeddings.")
        
        # if num_scenes != (len(self.sdf_train) + len(self.sdf_val)):
        #     raise ValueError("Setting number of embeddings different than total number of loaded scenes (train + validation datasets) !")
        
        self.num_scenes = num_scenes

        self.lat_vecs["trainable"] = nn.Embedding(
            self.num_scenes,
            self.latent_size,
            dtype=torch.float32
        )

        if embedding is not None:
            self.lat_vecs["trainable"].weight.data = torch.as_tensor(
                embedding, dtype=torch.float32
            )
        else:
            torch.nn.init.normal_(
                self.lat_vecs["trainable"].weight.data,
                0.0,
                1.0 / math.sqrt(self.latent_size),
            )

    def positional_encoding(self, xyz):
        freqs = self.freqs
        x_proj = [xyz]
        for freq in freqs:
            x_proj.append(torch.sin(freq * xyz / 100))
            x_proj.append(torch.cos(freq * xyz / 100))
        return torch.cat(x_proj, dim=-1)
    
    def configure_optimizers(self):
        if "trainable" in self.lat_vecs.keys():
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.decoder.parameters(),
                        "lr": self.lr_weights,
                        # "lr": self.lr_schedules[0].get_learning_rate(0),
                    },
                    {
                        "params": self.lat_vecs["trainable"].parameters(),
                        "lr": self.lr_latents #0.0005,  # was 0.0005
                        # "lr": self.lr_schedules[1].get_learning_rate(0),
                    },
                ]
            )
            if self.use_lr_scheduler:
                T_max = 100000

                def cosine_lambda(epoch, base_lr, eta_min):
                    return eta_min / base_lr + 0.5 * (1 - eta_min / base_lr) * (1 + math.cos(math.pi * epoch / T_max))

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=[
                        lambda epoch: cosine_lambda(epoch, self.lr_weights, self.lr_weights_final),
                        lambda epoch: cosine_lambda(epoch, self.lr_latents, self.lr_latents_final)
                    ]
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1
                    }
                }
        else:
            optimizer = torch.optim.Adam(params=self.decoder.parameters(), lr=self.lr_weights)

            if self.use_lr_scheduler:
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=10, eta_min=self.lr_weights_final
                    ),
                    "interval": "epoch",  # step per epoch
                    "frequency": 1,
                    "name": "lr_decoder"
                }

                return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    def on_fit_start(self):
        # move manually latents on the device to be sure it's on the same device as data when fitting the model
        self.lat_vecs["trainable"].to(self.device)
        return super().on_fit_start()
    
    def on_train_start(self):
        # d = flatten_dict_for_logging(self.specs)
        # self.logger.log_hyperparams(d)
        # self.logger.log_hyperparams({"hparams_json": json.dumps(self.specs)}) # --> only for mlflow
        if self.logger is not None:
            json_path = Path(self.logger.log_dir) / "hparams.json"
            with open(json_path, "w") as f:
                json.dump(self.specs, f, indent=4)

    def training_step(self, batch, batch_idx):

        # batch contains only scene indices, all data is already loaded on GPU
        indices = batch  # shape: (batch_size,)

        coords_list = []
        sdf_list = []

        for idx in indices:
            coords_full = self.trainer.datamodule.coords_gpu[idx]  
            sdf_full = self.trainer.datamodule.sdf_gpu[idx]        

            n_pts = coords_full.shape[0]

            # Randomly subsample points
            if self.trainer.datamodule.sampling_method == "random":
                samp_idx = torch.randint(0, n_pts, (self.trainer.datamodule.num_samples_per_scene,), device=coords_full.device)
            elif self.trainer.datamodule.sampling_method == "random_seed":
                torch.manual_seed(69)
                samp_idx = torch.randint(0, n_pts, (self.trainer.datamodule.num_samples_per_scene,), device=coords_full.device)
            elif self.trainer.datamodule.sampling_method == "all":
                samp_idx = torch.arange(n_pts, device=coords_full.device)

            coords_list.append(coords_full[samp_idx])
            sdf_list.append(sdf_full[samp_idx])

        # Stack and flatten for decoder
        coords = torch.stack(coords_list, dim=0)  # (batch_size, num_samples_per_scene, 3)
        sdf_gt = torch.stack(sdf_list, dim=0)     # (batch_size, num_samples_per_scene, sdf_dim)

        num_samp_per_scene = coords.shape[1]

        xyz = coords.reshape(-1, 3) * 100
        if self.use_positional_encoding:
            xyz = self.positional_encoding(xyz)

        sdf_gt = sdf_gt.reshape(-1, self.decoder.out_dim)
        if self.enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, self.minT, self.maxT)

        num_sdf_samples = sdf_gt.shape[0]

        indices.requires_grad = False
        xyz.requires_grad = False
        sdf_gt.requires_grad = False

        indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)
        # indices = indices.to(self.device)  # they are already actually

        batch_vecs = self.lat_vecs["trainable"](indices) #.to(self.device)

        input_ = torch.cat([batch_vecs, xyz], dim=1)

        # print(f"SDF gt range: {torch.min(sdf_gt), torch.max(sdf_gt)}")
        # print(f"COORDS gt range: {torch.min(xyz), torch.max(xyz)}")

        # NN optimization
        prediction = self.decoder(input_)
        if self.enforce_minmax:
            prediction = torch.clamp(prediction, self.minT, self.maxT)
        
        # LIPLOSS
        lipschits_loss = 1.0
        if self.lipschits_regularization:
            for layer in self.decoder.lipschits_layers:
                weight = self.decoder.__getattr__("lin" + str(layer)).weight
                norm = torch.linalg.matrix_norm(weight, ord=float("inf"))
                lipschits_loss *= F.softplus(norm)
            lipschits_loss = self.lipschits_alpha * lipschits_loss

        # REG LOSS
        l2_size_loss = torch.sum(torch.linalg.norm(batch_vecs, dim=1))

        reg_loss = l2_size_loss * self.code_reg_lambda / num_sdf_samples

        chunk_loss = self.loss_l1(prediction, sdf_gt) / num_sdf_samples
        
        training_loss = chunk_loss + reg_loss + lipschits_loss

        if self.logger is not None:
            self.log_dict(
                {
                    "reg_loss": reg_loss,
                    "l1_loss": chunk_loss,
                    "lipschits_loss": lipschits_loss,
                },
                logger=True
            )

            self.log_dict({"train_loss": training_loss}, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return training_loss

class DeepSDFBalancedGPU(pl.LightningModule):
    def __init__(self, decoder, specs):
        super().__init__()

        self.specs = specs
        
        self.decoder: Decoder = decoder
        
        # self.lr_schedules = get_learning_rate_schedules()

        self.lr_weights = specs.get("lr_weights", 0.001)
        self.lr_latents = specs.get("lr_latents", 0.0005)

        self.use_lr_scheduler = specs.get("use_lr_scheduler", False)
        if self.use_lr_scheduler:
            self.lr_weights_final = specs.get("lr_weights_final", 0.001)
            self.lr_latents_final = specs.get("lr_latents_final", 0.0005)

        self.latent_size = decoder.latent_size
        
        self.code_reg_lambda = specs.get("code_reg_lambda", 1e-2)

        self.use_lipreg_loss = specs.get("use_lipreg_loss", True)

        self.use_loss = specs.get("use_loss", "L1")

        self.lat_vecs = {"trainable": nn.Embedding}

        if self.use_loss == "L1":
            self.loss_fn = torch.nn.L1Loss(reduction="sum")  #! was reduction="sum"
        elif self.use_loss == "MSE":
            self.loss_fn = torch.nn.MSELoss(reduction="sum")
        else:
            self.loss_fn = torch.nn.L1Loss(reduction="sum") 

        # self.loss_fn = torch.nn.MSELoss(reduction="sum")  #! was reduction="sum"
        # self.loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
        # self.loss_cos = torch.nn.CosineSimilarity(dim=-1)

        self.use_positional_encoding = decoder.use_positional_encoding

        self.pos_enc_dim = decoder.pos_enc_dim

        if self.use_positional_encoding:
            self.register_buffer("freqs", 2.0 ** torch.arange(self.pos_enc_dim))    # This removes millions of sin/cos ops per step.

        self.enforce_minmax = specs.get("enforce_minmax", False)

        self.clamp_distance = specs.get("clamp_distance", 0.1)

        self.Cs = specs.get("scale_spatial_inputs_by", 100)

        self.lipschits_regularization = self.decoder.lipschits_layers[0] != -1
        self.lipschits_alpha = specs.get("lipschitz_alpha", 2e-6)


    def set_embedding(self, num_scenes = None, embedding=None):
        if num_scenes is None:
            raise ValueError("Must define number of scenes to create correct number of embeddings.")
        
        # if num_scenes != (len(self.sdf_train) + len(self.sdf_val)):
        #     raise ValueError("Setting number of embeddings different than total number of loaded scenes (train + validation datasets) !")
        
        self.num_scenes = num_scenes

        self.lat_vecs["trainable"] = nn.Embedding(
            self.num_scenes,
            self.latent_size,
            dtype=torch.float32
        )

        if embedding is not None:
            self.lat_vecs["trainable"].weight.data = torch.as_tensor(
                embedding, dtype=torch.float32
            )
        else:
            torch.nn.init.normal_(
                self.lat_vecs["trainable"].weight.data,
                0.0,
                1.0 / math.sqrt(self.latent_size),
            )

    def positional_encoding(self, xyz):
        freqs = self.freqs
        xyz_expanded = xyz[..., None, :] * freqs[:, None] * torch.pi / self.Cs #--> remove spatial scaling for sin and cos computation
        encoded = torch.cat([torch.sin(xyz_expanded), torch.cos(xyz_expanded)], dim=-1)
        return torch.cat([xyz, encoded.reshape(*xyz.shape[:-1], -1)], dim=-1)
    
    def configure_optimizers(self):
        if "trainable" in self.lat_vecs.keys():
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.decoder.parameters(),
                        "lr": self.lr_weights,
                        # "lr": self.lr_schedules[0].get_learning_rate(0),
                    },
                    {
                        "params": self.lat_vecs["trainable"].parameters(),
                        "lr": self.lr_latents #0.0005,  # was 0.0005
                        # "lr": self.lr_schedules[1].get_learning_rate(0),
                    },
                ]
            )
            if self.use_lr_scheduler:
                T_max = 100000

                def cosine_lambda(epoch, base_lr, eta_min):
                    return eta_min / base_lr + 0.5 * (1 - eta_min / base_lr) * (1 + math.cos(math.pi * epoch / T_max))

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=[
                        lambda epoch: cosine_lambda(epoch, self.lr_weights, self.lr_weights_final),
                        lambda epoch: cosine_lambda(epoch, self.lr_latents, self.lr_latents_final)
                    ]
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1
                    }
                }
        else:
            optimizer = torch.optim.Adam(params=self.decoder.parameters(), lr=self.lr_weights)

            if self.use_lr_scheduler:
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=10, eta_min=self.lr_weights_final
                    ),
                    "interval": "epoch",  # step per epoch
                    "frequency": 1,
                    "name": "lr_decoder"
                }

                return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    def on_fit_start(self):
        # move manually latents on the device to be sure it's on the same device as data when fitting the model
        self.lat_vecs["trainable"].to(self.device)
        return super().on_fit_start()
    
    def on_train_start(self):
        # d = flatten_dict_for_logging(self.specs)
        # self.logger.log_hyperparams(d)
        # self.logger.log_hyperparams({"hparams_json": json.dumps(self.specs)}) # --> only for mlflow
        if self.logger is not None:
            json_path = Path(self.logger.log_dir) / "hparams.json"
            with open(json_path, "w") as f:
                json.dump(self.specs, f, indent=4)

    def balance_batch(self, batch_indices):
        batch_coords = []
        batch_sdf = []
        for idx in batch_indices:
            coords_full = self.trainer.datamodule.coords_gpu[idx]  
            sdf_full = self.trainer.datamodule.sdf_gpu[idx]
            # retrieve also pos and neg indices for this scene
            pos_idxs = self.trainer.datamodule.pos_idx_gpu[idx]
            neg_idxs = self.trainer.datamodule.neg_idx_gpu[idx]  
 
            per_surface = self.trainer.datamodule.num_samples_per_scene // self.decoder.out_dim
            half = per_surface // 2

            idxs_total = []

            if self.trainer.datamodule.sampling_method in ["random", "random_seed"]:
                if self.trainer.datamodule.sampling_method == "random_seed":
                    torch.manual_seed(69)

                for s in range(self.decoder.out_dim):
                    # now stored in each entry of pos_idxs and neg_idxs (one entry per surface epi/la/ra) are stored torch tensors already on GPU (see setup of SDFBalancedSamplesGPU)
                    pos = pos_idxs[s]
                    neg = neg_idxs[s]

                    pos_sel = pos[torch.randint(0, len(pos), (half,), device=coords_full.device)]
                    neg_sel = neg[torch.randint(0, len(neg), (half,), device=coords_full.device)]

                    idxs_total.append(pos_sel)
                    idxs_total.append(neg_sel)

                idxs_total = torch.cat(idxs_total, dim=0)

                deficit = self.trainer.datamodule.num_samples_per_scene - idxs_total.shape[0]
                if deficit > 0:
                    extra = idxs_total[torch.randint(0, len(idxs_total), (deficit,), device=coords_full.device)]
                    idxs_total = torch.cat([idxs_total, extra], dim=0)
            elif self.trainer.datamodule.sampling_method == "all":
                idxs_total = torch.arange(coords_full.shape[0], device=coords_full.device)
            else:
                raise ValueError(f"Unknown sampling method: {self.trainer.datamodule.sampling_method}")

            batch_coords.append(coords_full[idxs_total])
            batch_sdf.append(sdf_full[idxs_total])
        
        return batch_coords, batch_sdf

    def training_step(self, batch, batch_idx):

        # batch contains only scene indices, all data is already loaded on GPU
        indices = batch  # shape: (batch_size,)

        batch_coords, batch_sdf = self.balance_batch(indices) # each batch has (roughly, accounting for batch size constraint) 50/50% pos/neg sdf samples for each surface represented

        # Stack into final batch for decoder
        coords = torch.stack(batch_coords, dim=0)  # (batch_size, num_samples_per_scene, 3)
        sdf_gt = torch.stack(batch_sdf, dim=0)     # (batch_size, num_samples_per_scene, sdf_dim)
        if self.enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, min = -self.clamp_distance, max=self.clamp_distance)

        num_samp_per_scene = coords.shape[1]

        xyz = coords.reshape(-1, 3) * self.Cs
        if self.use_positional_encoding:
            xyz = self.positional_encoding(xyz)

        sdf_gt = sdf_gt.reshape(-1, self.decoder.out_dim)

        num_sdf_samples = sdf_gt.shape[0]

        indices.requires_grad = False
        xyz.requires_grad = False
        sdf_gt.requires_grad = False

        indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)

        batch_vecs = self.lat_vecs["trainable"](indices)

        input_ = torch.cat([batch_vecs, xyz], dim=1)

        # NN optimization
        prediction = self.decoder(input_)
        if self.enforce_minmax:
            prediction = torch.clamp(prediction, min = -self.clamp_distance, max=self.clamp_distance)
        
        # L1 LOSS, clamped optionally
        chunk_loss = self.loss_fn(prediction, sdf_gt) / num_sdf_samples

        # REG LOSS
        reg_loss = self.code_reg_lambda * torch.sum( torch.linalg.norm(batch_vecs, dim=1) ) / num_sdf_samples

        # LIPLOSS
        if self.use_lipreg_loss:
            lipschits_loss = 1.0
            if self.lipschits_regularization:
                for layer in self.decoder.lipschits_layers:
                    weight = self.decoder.__getattr__("lin" + str(layer)).weight
                    norm = torch.linalg.matrix_norm(weight, ord=float("inf")) # TODO: bound this so it doesn't explode
                    lipschits_loss *= F.softplus(norm)
                lipschits_loss = self.lipschits_alpha * lipschits_loss
        else:
            lipschits_loss = 0.0

        training_loss = chunk_loss + reg_loss + lipschits_loss

        if self.logger is not None:
            self.log_dict(
                {
                    "reg_loss": reg_loss,
                    "l1_loss": chunk_loss,
                    "lipschits_loss": lipschits_loss,
                },
                logger=True
            )

            self.log_dict({"train_loss": training_loss}, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return training_loss


if __name__ == "__main__":

    pass

    specs_file = "specs_files/specs_deepsdfatria.json"
    
    specs = json.load( open(specs_file) )

    decoder = Decoder(**specs["Network_specs"])

    print( decoder.description() )

    # Create LIGHTNING MODULE
    model = DeepSDF(decoder, specs = specs)