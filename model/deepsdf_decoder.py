import torch
try:
    import lightning as pl # pyright: ignore[reportMissingImports]
except ImportError:
    import pytorch_lightning as pl
import json
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from deel.torchlip.modules.linear import SpectralLinear
import numpy as np

act_fn = {
    "ReLU": nn.ReLU(), "Tanh" : nn.Tanh(), 
    "Softplus": nn.Softplus(), 
    "SiLU": nn.SiLU(),
    "GELU" : nn.GELU(), 
    "GELUApprox": nn.GELU(approximate="tanh")
}

# ===== Lipschitz scaled layer, as in learning smooth neural functions via lip reg paper ===== #
class LipschitzNormLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, init_c=0.0):
        super().__init__(in_features, out_features, bias=bias)

        self.c = nn.Parameter(torch.ones(1) * init_c)  # Learnable Lipschitz scaling parameter (scalar per layer)

    def forward(self, x):
        # Compute row-wise absolute sum of the weight
        absrowsum = self.weight.abs().sum(dim=1, keepdim=True) + 1e-12

        # Compute row-wise scaling factor
        scale = torch.clamp(F.softplus(self.c) / absrowsum, max=1.0)

        # Scale the weight row-wise
        W_scaled = self.weight * scale

        # Linear forward (like nn.Linear)
        return F.linear(x, W_scaled, self.bias)

    def lipschitz_bound(self): # this is so I can just get it when computing the loss in training
        return F.softplus(self.c)
    
class Decoder(nn.Module):

    def __init__(self, **specs): # unpacks given keyword args into specs dictionary, so I can use .get() and have default instead of errors if the key isn't there
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

        self.lipschitz_layers = specs.get("lipschitz_layers", [-1]) # these are SpectralLinear layers, enforced to be lipschitz

        self.use_lipschitz_normalized_layers = specs.get("use_lipschitz_normalized_layers", False) # these are weight normalized layers, with learnable lipschitz bounds (...)

        self.hidden_dims = specs.get("dims", [])
        
        # self.check_decoder_specs_validity()
            
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
            elif layer in self.lipschitz_layers and not self.use_lipschitz_normalized_layers:
                lin = SpectralLinear(self.dims[layer], out_dim_).float()
            elif self.use_lipschitz_normalized_layers:
                lin = LipschitzNormLinear( self.dims[layer], out_dim_ ).float()
            else:
                lin = nn.Linear(self.dims[layer], out_dim_).float()

            setattr(self, f"lin{layer}", lin)

        self.act = act_fn.get(self.activation, "Softplus")

        self.last_tanh = specs.get("last_tanh", False)

        # TODO: check specs and decoder validity

        # for i in range(self.decoder.num_layers - 1):
        #     layer = getattr(self.decoder, f"lin{i}")
        #     if self.use_lipschitz_normalized_layers:
        #         assert isinstance(layer, LipschitzNormLinear), "Requested to use lipschitz normalized layers, but they have not been initialized correctly"
        #     elif i in self.lipschitz_layers:
        #         assert isinstance(layer, SpectralLinear)

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
        if self.lipschitz_layers != [-1]:
            f = f + f"\n Using Lipschitz constrained linear layers in {[i+1 for i in self.lipschitz_layers]}, layers type = SpectralLinear"
        if self.use_lipschitz_normalized_layers:
            f = f + f"\n Using Lipschitz normalized linear layers: layers type = LipschitzNormLinear"
        if self.use_positional_encoding:
            f = f + f"\n Using positional encoding of dimension {self.pos_enc_dim} on input."
        if self.latent_in != [-1]:
            f = f + f"\n Shortcut connection of input to layer {self.latent_in[0]}"
        f = f + f"\n Using latent dimension {self.latent_size}."
        f = f + f"\n Activations: {self.activation}"
        if self.last_tanh:
            f = f + ", tanh on output."

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            print(f"\n layer {layer}: {type(lin)}, weights = {lin.weights.shape}")
        
        return desc + f

class DeepSDF(pl.LightningModule):

    def __init__(self, decoder : Decoder, specs : dict):
        super().__init__()

        self.specs = specs
        
        self.decoder = decoder

        self.lr_weights = specs.get("lr_weights", 0.001)
        self.lr_latents = specs.get("lr_latents", 0.0005)
        self.lr_decay_T_max = specs.get("lr_decay_time_max", 100000)

        self.use_lr_scheduler = specs.get("use_lr_scheduler", False)
        if self.use_lr_scheduler:
            self.lr_weights_final = specs.get("lr_weights_final", 0.001)
            self.lr_latents_final = specs.get("lr_latents_final", 0.0005)

        self.latent_size = decoder.latent_size

        self.lat_vecs = {"trainable": nn.Embedding}

        self.code_reg_lambda = specs.get("code_reg_lambda", 1e-4)

        self.anneal_reg_loss = specs.get("anneal_reg_loss", False)

        self.anneal_warmup_epochs = specs.get("code_reg_anneal_warmup_epochs", 1000)

        self.use_loss = specs.get("use_loss", "SmoothL1")

        if self.use_loss == "L1":
            self.loss_fn = torch.nn.L1Loss(reduction="sum")
        elif self.use_loss == "MSE":
            self.loss_fn = torch.nn.MSELoss(reduction="sum")
        elif self.use_loss == "SmoothL1":
            self.loss_fn = torch.nn.SmoothL1Loss(reduction="sum") 

        self.use_lipreg_loss = self.decoder.use_lipschitz_normalized_layers

        self.lipschitz_alpha = specs.get("lipschitz_alpha", 2e-6)

        self.enforce_minmax = specs.get("enforce_minmax", False)

        self.clamp_distance = specs.get("clamp_distance", 0.1)

        self.Cs = specs.get("scale_spatial_inputs_by", 100)

        self.log_every_n_epochs = specs.get("log_every_n_epochs", 1)

    def set_embedding(self, num_scenes = None, embedding=None):
        if num_scenes is None:
            raise ValueError("Must define number of scenes to create correct number of embeddings.")
                
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

    def configure_optimizers(self):
        if "trainable" in self.lat_vecs.keys():
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.decoder.parameters(),
                        "lr": self.lr_weights,
                    },
                    {
                        "params": self.lat_vecs["trainable"].parameters(),
                        "lr": self.lr_latents 
                    },
                ]
            )
            if self.use_lr_scheduler:
                T_max =  self.lr_decay_T_max

                # maybe explore OneCycleR schedule: the LR first increases from a low start (lr_start) 
                # to a maximum (max_lr) over pct_start fraction of total steps 
                # Then decreases down to a final LR (lr_final) over the remaining steps (careful ! STEPS, not epochs !!)

                # CUSTOM SCHEDULER USING LambdaLR scheduler, fully custom
                def linear_lambda(epoch, lr_start, lr_final):
                    return max(lr_final / lr_start, 1.0 - (epoch / T_max) * (1.0 - lr_final / lr_start))

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    # lr_lambda function does not return the absolute LR, it returns a multiplicative factor for it
                    lr_lambda=[
                        lambda epoch: linear_lambda(epoch, self.lr_weights, self.lr_weights_final),
                        lambda epoch: linear_lambda(epoch, self.lr_latents, self.lr_latents_final)
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

        return optimizer

    def anneal_latent_reg(self):
        if "trainable" in self.lat_vecs.keys():
            # linear annealing for now self.current_epoch, self.anneal_warmup_epochs
            return self.code_reg_lambda * min(  self.current_epoch /  self.anneal_warmup_epochs, 1.0 )   

    def on_fit_start(self):
        # move manually latents on the device to be sure it's on the same device as data when fitting the model
        self.lat_vecs["trainable"].to(self.device)
        return super().on_fit_start()
    
    def on_train_start(self):
        # d = flatten_dict_for_logging(self.specs)
        # self.logger.log_hyperparams(d)
        # self.logger.log_hyperparams({"hparams_json": json.dumps(self.specs)}) # --> only for mlflow
        if self.logger is not None: # manually save the specs
            json_path = Path(self.logger.log_dir) / "hparams.json"
            with open(json_path, "w") as f:
                json.dump(self.specs, f, indent=4)

    def on_train_end(self):
        # for now, save as numpy data
        if self.logger is not None:
            npy_path = Path(self.logger.log_dir) / "latents.npy"
            embeddings = self.lat_vecs["trainable"].weight.data.cpu().numpy()
            np.save(npy_path, embeddings)

    def training_step(self, batch, batch_idx):

        data = batch[0]
        indices = batch[1]

        coords = data["coords"].to(self.device)
        sdf_gt = data["sdf"].to(self.device)

        num_samp_per_scene = data["coords"].shape[1]

        xyz = coords.reshape(-1, 3) * self.Cs

        sdf_gt = sdf_gt.reshape(-1, self.decoder.out_dim)
        if self.enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, min = -self.clamp_distance, max=self.clamp_distance)

        num_sdf_samples = sdf_gt.shape[0]

        indices.requires_grad = False
        xyz.requires_grad = False
        sdf_gt.requires_grad = False

        indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)

        batch_vecs = self.lat_vecs["trainable"](indices) # repeat latents for input

        input_ = torch.cat([batch_vecs, xyz], dim=1)

        # NN optimization
        prediction = self.decoder(input_)
        if self.enforce_minmax:
            prediction = torch.clamp(prediction, min = -self.clamp_distance, max=self.clamp_distance)
        
        # REGRESSION LOSS
        chunk_loss = self.loss_fn(prediction, sdf_gt) / (num_sdf_samples * self.decoder.out_dim) # divide by N only --> total error per sample, divide by N * out_dim --> average per scalar, most unit-free choice

        # REGULARIZATION LOSS  # was: reg_loss = torch.sum( torch.linalg.norm(batch_vecs, dim=1) ) / num_sdf_samples
        # don't waste computation on batch_vecs, in there are repeated latents !! was reg_loss = torch.sum( torch.linalg.norm(batch_vecs, dim=1) ** 2 ) / num_sdf_samples
        latents = self.lat_vecs["trainable"]( torch.unique(indices) )
        reg_loss = torch.mean( torch.linalg.norm(latents, dim=1) ** 2 )

        if self.anneal_reg_loss: # self.global_step is the total optimizer steps so far (across all epochs), self.current_epoch is the actual epoch
            code_reg_lambda = self.anneal_latent_reg()
        else:
            code_reg_lambda =  self.code_reg_lambda

        lipschitz_loss = 0.0
        if self.use_lipreg_loss:
            for i in range(self.decoder.num_layers - 1):
                layer = getattr(self.decoder, f"lin{i}")
                softplus_ci = layer.lipschitz_bound()
                lipschitz_loss += torch.log( softplus_ci ) 
            lipschitz_loss = torch.exp(lipschitz_loss)

        training_loss = chunk_loss + code_reg_lambda * reg_loss + self.lipschitz_alpha * lipschitz_loss


        if self.logger is not None and (self.current_epoch + 1) % self.log_every_n_epochs == 0:
            
            # optimizer = self.optimizers()
            # # optimizer.param_groups is a list of dicts — each dict corresponds to one group defined in torch.optim.Adam([ ... ])
            # current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            # lr_weights, lr_latents = current_lrs

            self.log_dict(
                {
                    "latents_mean_L2_squared": reg_loss.detach().cpu(),
                    "lipschitz_penalty": lipschitz_loss.detach().cpu() if self.use_lipreg_loss else torch.tensor(0.0, device="cpu"),
                    "regression_loss": chunk_loss.detach().cpu(),
                    "train_loss" : training_loss.detach().cpu(),
                    "code_reg_lambda" : code_reg_lambda,
                    # "lr_weights" : lr_weights,
                    # "lr_latents" : lr_latents
                },
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=False
            )

        return training_loss






if __name__ == "__main__":

    pass

""" --> old setup
        # # REGULARIZATION LOSS  # was: reg_loss = torch.sum( torch.linalg.norm(batch_vecs, dim=1) ) / num_sdf_samples
        # reg_loss = torch.sum( torch.linalg.norm(batch_vecs, dim=1) ** 2 ) / num_sdf_samples

        # # if self.normalize_reg_loss:
        # #     pass


        # # # LIPSCHITZ PENALTY
        # # if self.use_lipreg_loss:
        # #     lipschitz_loss = 1.0
        # #     for layer in range(self.decoder.num_layers):
        # #         weight = self.decoder.__getattr__("lin" + str(layer)).weight
        # #         norm = torch.linalg.matrix_norm(weight, ord=float("inf")) # TODO: bound this so it doesn't explode
        # #         lipschitz_loss *= F.softplus(norm)
        # #     lipschitz_loss = self.lipschitz_alpha * lipschitz_loss
        # # else:
        # #     lipschitz_loss = 0.0

        # # # LIPSCHITZ PENALTY
        # lipschitz_loss = 0.0
        # if self.use_lipreg_loss: # compute product of spectral norms for all layers !
        #     Ws = torch.stack([getattr(self.decoder, f"lin{i}").weight for i in range(self.decoder.num_layers)])
        #     # Compute norms for all layers at once
        #     norms = torch.linalg.matrix_norm(Ws, ord=float('inf'), dim=(1,2))  # shape: (num_layers,)
        #     softplus_norms = F.softplus(norms)
        #     # compute product of spectral norms as sum in log space for stability
        #     log_prod = torch.sum(torch.log(softplus_norms + 1e-12))  # add eps for stability
        #     # lipschitz_loss = torch.exp(log_prod) # normalized by depth:  torch.exp(log_prod / self.decoder.num_layers)
        #     # do not exponentiate for stabilty, for training it is unnecessary ( still get the same minimizer)
        #     lipschitz_loss = log_prod 

        # training_loss = chunk_loss + self.code_reg_lambda * reg_loss + self.lipschitz_alpha * lipschitz_loss

        # if self.logger is not None and (self.current_epoch + 1) % self.log_every_n_epochs == 0:

        #     self.log_dict(
        #         {
        #             "latent_reg_loss": reg_loss,
        #             "prediction_loss": chunk_loss,
        #         },
        #         logger=True
        #     )

        #     self.log(
        #         "train_loss",
        #         training_loss,
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         logger=True
        #     )
"""