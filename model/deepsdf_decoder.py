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

# # ---- to reconstruct surface for validation ---- #
# from skimage import measure
# import pyvista as pv

# def isosurface_from_sdf(x, y, z, sdf_pred, level, box_lim = 105):
    
#     D = sdf_pred.reshape((len(x), len(y), len(z)))

#     D = np.transpose(D, (1, 0, 2))

#     # Run marching cubes
#     verts, faces, normals, values = measure.marching_cubes(
#         D,
#         level=level,
#         spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0])
#     )

#     # Adjust vertices
#     # my volume is in [-105,105] cube but marching cubes assumes a vertex is in (0,0,0), so I need to traslate it back to my real coordinates
#     verts = verts - box_lim  

#     # Convert faces for PyVista
#     faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)

#     # Create PyVista mesh
#     mesh = pv.PolyData(verts, faces_pv)

#     return mesh



act_fn = {
    "ReLU": nn.ReLU(), "Tanh" : nn.Tanh(), 
    "Softplus": nn.Softplus(), 
    "SiLU": nn.SiLU(),
    "GELU" : nn.GELU(), 
    "GELUApprox": nn.GELU(approximate="tanh")
}




class LipschitzNormLinear(nn.Linear):
    """
        Lipschitz scaled layer, as in learning smooth neural functions via lip reg paper
    """
    def __init__(self, in_features, out_features, bias=True, init_c=1.0): #, init_c=0.0):
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

        self.actual_concatenation = specs.get("actual_skip_concatenation", False)
        
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
                if self.actual_concatenation:
                    self.dims[layer + 1] += self.dims[0]
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
        # - if lipschitz_layers is not -1 but also use_lipschitz_normalized_layers is True, warn that the latter will override SpectraLinear layers !!
        # - check dimensions requested can be build with wanted skip connection, + pos enc

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

        f = f" \n {self.num_layers} layers with channels {self.dims} :"

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            f += f"\n  layer {layer}:  {lin.__class__.__name__}, shape {lin.weight.shape[0], lin.weight.shape[1]}"

        # if self.lipschitz_layers != [-1]:
        #     f = f + f"\n Using Lipschitz constrained linear layers in {[i+1 for i in self.lipschitz_layers]}"

        # if self.use_lipschitz_normalized_layers:
        #     f = f + f"\n Using Lipschitz normalized linear layers"

        if self.use_positional_encoding:
            f = f + f"\n Using positional encoding of dimension {self.pos_enc_dim} on input."

        if self.latent_in != [-1]:
            f = f + f"\n Shortcut connection of input to layer {self.latent_in}"

        f = f + f"\n Using latent dimension {self.latent_size}."

        f = f + f"\n Activations: {self.activation}"

        if self.last_tanh:
            f = f + ", tanh on output. \n"

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

        # EIKONAL terms
        # self.use_eikonal = specs.get("use_eikonal_loss", False)
        self.use_eikonal_loss = False

        # self.eikonal_weight = specs.get("eikonal_weight", 1e-2)
        self.eikonal_weight = 1e-3
        
        # self.ekional_frac = specs.get("eikonal_frac", 0.25)
        self.eikonal_frac = 1


        self.enforce_minmax = specs.get("enforce_minmax", False)

        self.clamp_distance = specs.get("clamp_distance", 0.1)

        self.Cs = specs.get("scale_spatial_inputs_by", 1.00)

        self.log_every_n_epochs = specs.get("log_every_n_epochs", 1000)

        self.log_val_every_n_epochs = specs.get("log_val_every_n_epochs", None)

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

    def regression_error(self, pred, gt):
        if self.use_loss == "L1":
            return F.l1_loss(pred, gt, reduction="none")

        elif self.use_loss == "MSE":
            return F.mse_loss(pred, gt, reduction="none")

        elif self.use_loss == "SmoothL1":
            return F.smooth_l1_loss(pred, gt, reduction="none")

        else:
            raise ValueError(f"Unknown loss: {self.use_loss}")
    
    def training_step2(self, batch, batch_idx):

        data = batch[0]
        indices = batch[1]

        coords = data["coords"].to(self.device)
        sdf_gt = data["sdf"].to(self.device)

        mask = data["mask"].to(self.device)

        num_samp_per_scene = data["coords"].shape[1]

        xyz = coords.reshape(-1, 3) * self.Cs

        if getattr(self, "use_eikonal_loss", False):
            xyz = xyz.clone().detach().requires_grad_(True)

        # if we '* self.Cs' is uncommented in the definition of xyz above, then uncomment it also in the definition of sdf_gt below.
        sdf_gt = sdf_gt.reshape(-1, self.decoder.out_dim)

        mask = mask.reshape(-1, self.decoder.out_dim)

        if self.enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, min = -self.clamp_distance, max=self.clamp_distance)

        num_sdf_samples = sdf_gt.shape[0]

        indices.requires_grad = False
        # xyz.requires_grad = False
        sdf_gt.requires_grad = False

        indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)

        batch_vecs = self.lat_vecs["trainable"](indices) # repeat latents for input

        input_ = torch.cat([batch_vecs, xyz], dim=1)

        # NN optimization
        prediction = self.decoder(input_)
        if self.enforce_minmax:
            prediction = torch.clamp(prediction, min = -self.clamp_distance, max=self.clamp_distance)
        
        # REGRESSION LOSS
        # chunk_loss = self.loss_fn(prediction, sdf_gt) / (num_sdf_samples * self.decoder.out_dim) # divide by N only --> total error per sample, divide by N * out_dim --> average per scalar, most unit-free choice

        err = self.regression_error(
            prediction,
            sdf_gt,
        )

        chunk_loss = 0.0

        for j in range(self.decoder.out_dim):
            denom = mask[:, j].sum() + 1e-8
            loss_j = (err[:, j] * mask[:, j]).sum() / denom
            chunk_loss = chunk_loss + loss_j

        chunk_loss = chunk_loss / self.decoder.out_dim


        # REGULARIZATION LOSS  # was: reg_loss = torch.sum( torch.linalg.norm(batch_vecs, dim=1) ) / num_sdf_samples
        # don't waste computation on batch_vecs, in there are repeated latents !! was reg_loss = torch.sum( torch.linalg.norm(batch_vecs, dim=1) ** 2 ) / num_sdf_samples
        latents = self.lat_vecs["trainable"]( torch.unique(indices) )
        reg_loss = torch.mean( torch.linalg.norm(latents, dim=1) ** 2 )

        if self.anneal_reg_loss: # self.global_step is the total optimizer steps so far (across all epochs), self.current_epoch is the actual epoch
            code_reg_lambda = self.anneal_latent_reg()
        else:
            code_reg_lambda =  self.code_reg_lambda

        # LIPSCHTIZ LOSS
        lipschitz_loss = 0.0
        if self.use_lipreg_loss:
            for i in range(self.decoder.num_layers - 1):
                layer = getattr(self.decoder, f"lin{i}")
                softplus_ci = layer.lipschitz_bound()
                lipschitz_loss += torch.log( softplus_ci ) 
            lipschitz_loss = torch.exp(lipschitz_loss)

        # EIKONAL LOSS
        # qua computiamo il termine eikonale (grad(S) - 1)
        # eikonal_loss = torch.tensor(0.0, device=self.device)
        eikonal_loss = 0.0
        #debug
        eikonal_loss = torch.tensor(0.0, device=self.device)
        # fine debug
        eikonal_alpha = self.eikonal_weight


        if getattr(self, "use_eikonal_loss", False):
            # old eikonal way (wrong)
            # N = xyz.shape[0]

            # frac = float(getattr(self, "eikonal_frac", 0.25))
            # frac = max(0.0, min(1.0, frac))

            # M = max(1, int(frac * N))

            # eik_idx = torch.randperm(N, device=xyz.device)[:M]

            # xyz_eik = xyz[eik_idx].detach().clone().requires_grad_(True)

            # batch_vecs_eik = batch_vecs[eik_idx].detach()

            # input_eik = torch.cat([batch_vecs_eik, xyz_eik], dim=1)
            # pred_eik = self.decoder(input_eik)

            # if self.decoder.out_dim > 1:
            #     pred_eik_scalar = pred_eik.sum(dim=1, keepdim=True)
            # else:
            #     pred_eik_scalar = pred_eik
            
            # grads = torch.autograd.grad(
            #     outputs=pred_eik_scalar,
            #     inputs=xyz_eik,
            #     grad_outputs=torch.ones_like(pred_eik_scalar),
            #     create_graph=True,
            #     retain_graph=True,
            #     only_inputs=True        
            #     )[0]
            
            # grad_norm = grads.norm(2, dim=1)
            # eikonal_loss = (grad_norm - 1.0).abs().mean() 
            # eikonal_loss = ((grad_norm - 1.0)**2).mean() 
            # fine old eikonal

            # new eikonal
            grads = []

            # we compute the gradients of the points in the unitary sphere
            xyz = xyz / self.Cs

            for k in range(prediction.shape[1]):
                grad_k = torch.autograd.grad(
                    outputs=prediction[:,k].sum(),
                    inputs=xyz,
                    create_graph=True
                )[0]
                grads.append(grad_k.norm(dim=1))
            
            grads = torch.stack(grads, dim=1)

        
            eikonal_alpha = self.eikonal_weight

            # target = 1.0 / self.Cs
            target = torch.tensor(1.0, device=self.device)

            # eikonal_loss = ((grads - target)**2).mean()

            # masked eikonal
            eikonal_error = (grads - 1.0) ** 2

            eikonal_loss = torch.zeros(
                (),
                device=self.device,
                dtype=eikonal_error.dtype,
            )

            n_valid_surfaces = 0

            for k in range(self.decoder.out_dim):
                valid = mask[:, k] > 0.5

                if valid.any():
                    eikonal_loss = (
                        eikonal_loss
                        + eikonal_error[valid, k].mean()
                    )
                    n_valid_surfaces += 1

            if n_valid_surfaces > 0:
                eikonal_loss = eikonal_loss / n_valid_surfaces

            #end masked eikonal

        # qua dovremo aggiungere il termine eikonale 
        training_loss = (chunk_loss 
                        + code_reg_lambda * reg_loss
                        + self.lipschitz_alpha * lipschitz_loss
                        + eikonal_alpha * eikonal_loss)


        if self.logger is not None and (self.current_epoch + 1) % self.log_every_n_epochs == 0:
            
            # optimizer = self.optimizers()
            # # optimizer.param_groups is a list of dicts — each dict corresponds to one group defined in torch.optim.Adam([ ... ])
            # current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            # lr_weights, lr_latents = current_lrs

            logs = {
                "latents_mean_L2_squared": reg_loss.detach(),
                "lipschitz_penalty": lipschitz_loss.detach() if self.use_lipreg_loss else torch.tensor(0.0, device=self.device),
                "prediction_loss": chunk_loss.detach(),
                "training_loss": training_loss.detach(),
                "code_reg_factor": torch.tensor(code_reg_lambda, device=self.device),
                "eikonal_loss": eikonal_loss.detach(),
            }
                        
            if self.use_eikonal_loss:
                logs.update({
                    "eikonal_loss_signed": (grads - target).mean().detach(),
                    "eikonal_loss_abs": (grads - target).abs().mean().detach(),
                    "grad_norm_mean": grads.mean().detach(),
                    "grad_norm_min": grads.min().detach(),
                    "grad_norm_max": grads.max().detach(),
                })

            self.log_dict(
                logs,
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=False
            )

        return training_loss
    
    # nuova funzione di training_step, la rpecedente cambiata in trainin_Step2
    def training_step(self, batch, batch_idx):

        data = batch[0]
        indices = batch[1]

        coords = data["coords"].to(self.device)
        sdf_gt = data["sdf"].to(self.device)

        mask = data["mask"].to(self.device)

        num_samp_per_scene = coords.shape[1]

        # Coordinate originali
        xyz_raw = coords.reshape(-1, 3)

        # Se uso eikonal, voglio derivare rispetto a xyz_raw
        if getattr(self, "use_eikonal_loss", False):
            xyz_raw = xyz_raw.clone().detach().requires_grad_(True)

        # Coordinate scalate date al decoder
        xyz = xyz_raw * self.Cs

        sdf_gt = sdf_gt.reshape(-1, self.decoder.out_dim)
        mask = mask.reshape(-1, self.decoder.out_dim)

        if self.enforce_minmax:
            sdf_gt = torch.clamp(
                sdf_gt,
                min=-self.clamp_distance,
                max=self.clamp_distance
            )

        num_sdf_samples = sdf_gt.shape[0]

        indices.requires_grad = False
        sdf_gt.requires_grad = False

        indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)

        batch_vecs = self.lat_vecs["trainable"](indices)

        input_ = torch.cat([batch_vecs, xyz], dim=1)

        prediction = self.decoder(input_)

        if self.enforce_minmax:
            prediction = torch.clamp(
                prediction,
                min=-self.clamp_distance,
                max=self.clamp_distance
            )

        # chunk_loss = self.loss_fn(prediction, sdf_gt) / (
        #     num_sdf_samples * self.decoder.out_dim
        # )

        err = self.regression_error(
            prediction,
            sdf_gt,
        )

        chunk_loss = 0.0

        for j in range(self.decoder.out_dim):
            denom = mask[:, j].sum() + 1e-8
            loss_j = (err[:, j] * mask[:, j]).sum() / denom
            chunk_loss = chunk_loss + loss_j

        chunk_loss = chunk_loss / self.decoder.out_dim


        latents = self.lat_vecs["trainable"](torch.unique(indices))
        reg_loss = torch.mean(torch.linalg.norm(latents, dim=1) ** 2)

        if self.anneal_reg_loss:
            code_reg_lambda = self.anneal_latent_reg()
        else:
            code_reg_lambda = self.code_reg_lambda

        lipschitz_loss = torch.tensor(0.0, device=self.device)

        if self.use_lipreg_loss:
            for i in range(self.decoder.num_layers - 1):
                layer = getattr(self.decoder, f"lin{i}")
                softplus_ci = layer.lipschitz_bound()
                lipschitz_loss = lipschitz_loss + torch.log(softplus_ci)

            lipschitz_loss = torch.exp(lipschitz_loss)

        eikonal_loss = torch.tensor(0.0, device=self.device)
        eikonal_alpha = self.eikonal_weight

        if getattr(self, "use_eikonal_loss", False):
            grads = []

            for k in range(prediction.shape[1]):
                grad_k = torch.autograd.grad(
                    outputs=prediction[:, k].sum(),
                    inputs=xyz_raw,
                    create_graph=True,
                    retain_graph=True,
                )[0]

                grads.append(grad_k.norm(dim=1))

            grads = torch.stack(grads, dim=1)

            # target = torch.tensor(1.0, device=self.device)
            # eikonal_loss = ((grads - target) ** 2).mean()

            target = torch.tensor(
                1.0,
                device=self.device,
                dtype=grads.dtype,
            )

            eikonal_error = (grads - target) ** 2

            eikonal_loss = torch.zeros(
                (),
                device=self.device,
                dtype=eikonal_error.dtype,
            )

            n_valid_surfaces = 0

            for k in range(self.decoder.out_dim):
                valid = mask[:, k] > 0.5

                if valid.any():
                    eikonal_loss = (
                        eikonal_loss
                        + eikonal_error[valid, k].mean()
                    )
                    n_valid_surfaces += 1

            if n_valid_surfaces > 0:
                eikonal_loss = (
                    eikonal_loss / n_valid_surfaces
                )


        training_loss = (
            chunk_loss
            + code_reg_lambda * reg_loss
            + self.lipschitz_alpha * lipschitz_loss
            + eikonal_alpha * eikonal_loss
        )

        if self.logger is not None and (self.current_epoch + 1) % self.log_every_n_epochs == 0:

            logs = {
                "latents_mean_L2_squared": reg_loss.detach(),
                "lipschitz_penalty": lipschitz_loss.detach(),
                "prediction_loss": chunk_loss.detach(),
                "training_loss": training_loss.detach(),
                "code_reg_factor": torch.tensor(code_reg_lambda, device=self.device),
                "eikonal_loss": eikonal_loss.detach(),
            }

            if getattr(self, "use_eikonal_loss", False):
                logs.update({
                    "eikonal_loss_signed": (grads - target).mean().detach(),
                    "eikonal_loss_abs": (grads - target).abs().mean().detach(),
                    "grad_norm_mean": grads.mean().detach(),
                    "grad_norm_min": grads.min().detach(),
                    "grad_norm_max": grads.max().detach(),
                })

            self.log_dict(
                logs,
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=False
            )

        return training_loss

    def validation_step(self, batch, batch_idx):

        data = batch[0]

        batch_size = data["coords"].shape[0]  # batch size
        num_samp_per_scene = data["coords"].shape[1]  # points per scene

        xyz_gt = data["coords"].to(self.device)
        sdf_gt = data["sdf"].to(self.device)

        mask = data["mask"].to(self.device)

        xyz_gt = xyz_gt.reshape(-1, 3) * self.Cs

        # if we '* self.Cs' is uncommented in the definition of xyz_gt above, then uncomment it also in the definition of sdf_gt below.
        sdf_gt = sdf_gt.reshape(-1, self.decoder.out_dim) #* self.cs
        mask = mask.reshape(-1, self.decoder.out_dim)

        if self.enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, min=-self.clamp_distance, max=self.clamp_distance)

        num_sdf_samples = sdf_gt.shape[0]

        xyz_gt.requires_grad = False
        sdf_gt.requires_grad = False

        # starting point for optimization: zero
        # I could also save initial vectors when training and start with empirical mean and covariance,
        # sampling a latent using MultivariateNormal and rsample()
        # TODO: add option to start from somewhere else (from mean of loaded latents, random sample, ...)

        latents = torch.zeros(batch_size, self.latent_size, device=self.device, requires_grad=True)
            
        optimizer = torch.optim.Adam(params=[latents], lr=0.005)
        
        # ==================================================== #
        # region fit latent
        # ==================================================== #
        with torch.enable_grad(): # I want them during validation, they are disabled automatically
            
            for i in range(250):
                
                self.decoder.eval()
                
                optimizer.zero_grad()

                batch_vecs = latents.unsqueeze(1).expand(batch_size, num_samp_per_scene, self.latent_size).reshape(batch_size*num_samp_per_scene, self.latent_size)
                input_ = torch.cat([batch_vecs, xyz_gt], dim=1)

                sdf_pred = self.decoder(input_)
                if self.enforce_minmax:
                    sdf_pred = torch.clamp(sdf_pred, min = -self.clamp_distance, max=self.clamp_distance)
                    
                # mahalanobis to train codes distribution

                # vanilla loss : same loss as in training
                reg_loss = torch.sum(latents ** 2, dim=1).mean() 

                # chunk_loss = self.loss_fn(sdf_pred, sdf_gt) / (num_sdf_samples *  self.decoder.out_dim)
                
                err = self.regression_error(
                    sdf_pred,
                    sdf_gt,
                )

                chunk_loss = 0.0

                for j in range(self.decoder.out_dim):
                    denom = mask[:, j].sum() + 1e-8
                    loss_j = (err[:, j] * mask[:, j]).sum() / denom
                    chunk_loss = chunk_loss + loss_j

                chunk_loss = chunk_loss / self.decoder.out_dim


                loss = chunk_loss + self.code_reg_lambda * reg_loss

                loss.backward()

                optimizer.step()

        regression_loss = chunk_loss

        # # ==================================================== #
        # # region reconstruct surface from predicted sdf
        # # ==================================================== #
        # latent.requires_grad = False    
        # resolution = 128
        # box_lim = self.Cs * 1.05

        # with torch.no_grad():
            
        #     self.decoder.eval()

        #     #region SDF ON GRID FOR RECONSTRUCTION
        #     x = np.linspace(-box_lim, box_lim, resolution)
        #     y = np.linspace(-box_lim, box_lim, resolution)
        #     z = np.linspace(-box_lim, box_lim, resolution)
        #     xx, yy, zz = np.meshgrid(x, y, z)

        #     grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        #     xyz_raw = grid

        #     n_points = 500000
        #     n_batches = len(xyz_raw) // n_points

        #     sdf_preds = []

        #     for i in range(n_batches + 1):
        #         if i < n_batches:
        #             xyz = torch.from_numpy(xyz_raw[n_points * i : n_points * (i + 1)]).to(self.device)
        #         else:
        #             xyz = torch.from_numpy(xyz_raw[n_points * i :]).to(self.device)

        #         batch_vecs = latent.expand(xyz.shape[0], -1)

        #         input_ = torch.cat([batch_vecs, xyz], dim=1)

        #         sdf_pred_batch = self.decoder(input_.to(torch.float32)).cpu().data.numpy() 

        #         sdf_preds.append(sdf_pred_batch)
            
        # sdf_pred = np.concatenate(sdf_preds, axis=0)

        # # ==================================================== #
        # # region compute some metric to track
        # # ==================================================== #

        # sdfs_pred = {
        #     "epicardium": sdf_pred[:, 0],
        #     "la_endo": sdf_pred[:, 1],
        #     "ra_endo": sdf_pred[:, 2]
        # }

        # organs_to_process = ["epicardium", "la_endo", "ra_endo"]

        # for i,organ in enumerate(organs_to_process):

        #     threshold = 0.0 
                                    
        #     try:
        #         mesh_reconstructed = isosurface_from_sdf( x, y, z, sdf_pred=sdfs_pred[organ], level = threshold, box_lim = box_lim )
        #     except:
        #         logger.warning( f"Version {version}: skipping {organ} isosurface extraction: not found for current isovalue")
        #         if i == len(organs_to_process)-1:
        #             return
        #         else:
        #             continue

        self.log_dict(
            {
                "sdf_regression_loss_on_test_shapes": regression_loss.detach().cpu()
            },
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )

        return loss




if __name__ == "__main__":

    # specs = {
    #     "latent_size" : 64,
    #     "out_dim" : 3,
    #     "dims" : [256,256,256,256,256],
    #     "latent_in" : [3],
    #     "actual_skip_concatenation" : True,
    #     "positional_encoding" : False,
    #     "pos_enc_dim" : 4,   
    #     "lipschitz_layers" : [1,3],
    #     "use_lipschitz_normalized_layers" : False,
    #     "activation" : "SiLU",
    #     "last_tanh" : False,
    #     "norm_layers" : [-1],
    #     "batch_norm" : False,
    #     "dropout_prob" : 0.2,
    #     "dropout_layers" : [-1]       
    # }

    # decoder = Decoder(**specs)

    # print(decoder.description())

    pass
