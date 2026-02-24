# DeepSDF for bi-atrial geometries
(ctrl+shift+v)
## File organization (as of now)

#### Setup
The various python scripts rely on a specific organizational structure which is completely defined and customizable in the `config.py` script. 

This sets paths for directories containing meshes files for each patient, training data, and results directories, as well as specific informations on locations of atrial regions used to process the original meshes from the specific dataset used (see specifics in How to use).


#### Experiments runs
Each training run is organized under an "experiment" and "version" directory. Training only results in the creation of `.pth` files storing the trained model weights, and an `hparams.json` file recording the specifics used for that run. Model checkpoints during training can be also saved.

The only file that is required to start training (or testing) is a `.json` under the specification files directory path, which sets the hyperparameters, network architecture, and data files to be used for the experiment.

Example of a specs file:
```python
{
    "TrainSplit" : "train/data_fnames_train.json",

    "TestSplit" : "test/data_fnames_test.json",

    "Network_specs" : {
        "latent_size" : 64,
        "out_dim" : 3,
        "dims" : [256,256,256,256,256],
        "latent_in" : [2],
        "positional_encoding" : false,
        "pos_enc_dim" : 4,   
        "lipschitz_layers" : [2,3,4],
        "regularize_layers" : [-1],
        "activation" : "SiLU",
        "last_tanh" : false,
        "norm_layers" : [-1],
        "batch_norm" : false,
        "dropout_prob" : 0.2,
        "dropout_layers" : [-1]       
    },
    
    "NumEpochs": 50000,
    "lr_weights": 0.001,
    "lr_latents": 0.005,
    "num_samp_per_scene": 4096,
    "sampling_scene_method": "random",
    "balance_pos_neg" : false,
    "batch_size": 16,
    "scale_spatial_inputs_by": 100,
    "enforce_minmax": false,
    "clamp_distance": 0.2,
    "use_loss": "SmoothL1",
    "use_lipreg_loss": false,
    "lipschitz_alpha": 2e-06,
    "code_reg_lambda": 2e-05,
    "use_lr_scheduler": true,
    "lr_weights_final": 1e-05,
    "lr_latents_final": 5e-05,
    "lr_decay_time_max": 100000,
    "log_every_n_epochs" : 100,
    "checkpoint_every_n_epochs": 20001,
    "resume_training_from_version": "-"
}
```

## How to use

### Processing of bi-atrial volumetric meshes

In order to use mesh data for training a DeepSDF model, the mesh will need to be pre-processed. This can be done with the `atria_data_preparing.py` executable. This script allows to:

- create `.vtu` files of volumetric meshes of the atria starting from `carp_bin` files defining the mesh, using meshtool.
- extract **watertight** epicardium and left / right endocardium surfaces from volumetric meshes of the atria in `.vtu` format
- create `.npy` files containing arrays of sampled points with their sdf values for a given anatomy

The surface extraction is done using a specific dictionary of TAGS identifying various parts of the anatomies. For now this is tailored *precisely* to be used with the dataset supplied by Elena, if new geometries are added they have to respect these exact tags!

All these steps can be done separately or alltogheter, over one or multiple patients. See docstrings of functions to understand what directories or files are expected, and how and where results are saved.

Possible TODO: make it executable from command like passing options as input, actually split into several executable scripts that each do a step, then explain here the intended sequence in which to use them if starting from scratch.

### Training 

After setting data and results directory in `config.py` script, the only thing needed to start training is a `.json` file specifying a dictionary holding all training hyperparameters, network architecture, and actual data file path to use.

#### data files
Data will be loaded in the dataloader from `.json` files containing a list of `.npy` file names, one for each patient/anatomy. The full path will be constructed relative to the `PATIENT_NUMPY_DATA_DIR` specified in the `config.py` file, so the `.json` just needs to specify each single `.npy` file to use, assumed they are then found under the same directory `PATIENT_NUMPY_DATA_DIR`.

example: `data_fnames.json` contains `["patient1.npy", "patient2.npy"]`, then full paths are assumed to be `PATIENT_NUMPY_DATA_DIR/patient1.npy`, `PATIENT_NUMPY_DATA_DIR/patient2.npy`

#### train script
Then the script `train.py` can be executed combining optional features:

- `--experiment_name`, `-e` *(str)*  
  Experiment identifier. Becomes the directory name under which the corresponding `version_x` folder is created for the training run.  

- `--specs_file_path`, `-s` *(str)*  
  Path to the `.json` specs file defining training hyperparameters, network architecture, and data paths.

<!-- - `--train_mode` *(str)*  
  Training mode selector (default: `use_specs_file`). If passed as `compose_specs_from_options` then overwrites fields in the original specs files with new ones  

- `--override_specs` *(str)*  
  Path to an alternative specs file. Overrides values defined in the default specs file. It can also just contain the specific fields to override, with the same names as in the original specs. -->

- `--show_progress`  
  Optionally display training progress bar.

<!-- A folder `experiment` will be created as a directory under the `EXPERIMENTS_DIR` path specified in `config.py`. The training creates a `.pth` file storing the model weights, records the specs used in a `hparams.json` file, and records the trainer logs in an `events.out` type file readable with tensorboard. Each run with the same experiment name will be saved under `version_x` folders under the same experiment directory.  -->


### Testing and Results

A trained model from a specific experiment and version can be loaded from just the `.pth` file storing the model weights, and the `hparams.json` specification file defining the architecture. 

The script `test.py` can be executed with additional flags specifying the trained model to use, data, and what to do:

- `--experiment_name`, `-e` *(str)*  and `--version`, `-v` *(str)*
  Run identifiers, specify which run to load decoder weights and parameters from 

- `--override_with_test_dataset`, , `-od` *(str)*  
  path to `.json` file indicating which anatomies to process, overrides the one specified in specs `TestSplit`. This again is a file storing paths to `.npy` files storing coords and sdf for each anatomy.

- `--mode`, `-m` *(int, {1,2})*  
  `1`: fit latent codes to reconstruct surfaces, optionally visualize / compute metrics / save  
  `2`: fit and save latent codes only

- `--num_epochs`, `-N` and `--lr` and `--latent_reg_factor`, `-lreg` 
  Number of epochs to fit latent code, learning rate, and factor of latent regularization in the loss

Then one can specify further options if wanted: 
- `--save_latent_codes`, `-sc`  
  Save fitted latent codes

- `--interactive_images`, `-i`  
  Show interactive reconstruction images

- `--save_images`, `-si`  
  Save reconstruction images, images can be generated off screen

- `--save_reconstructed_meshes`, `-sm`  
  Save reconstructed meshes to `.vtp` format

- `--compute_metrics`, `-cm`  
  Compute chamfer, haussdorff and LDDMM metrics, results are always saved to `.parquet` files if computed

This will process all the anatomies (patients) specified in the test split.

**Example** : reconstruct and save surfaces, then save screenshot of plots off screen and computed chamfer distance values
```bash
python test.py \
  --experiment_name deepsdf_atria_training \
  --version version_114 \
  --mode 1 \
  --save_images \
  --save_reconstructed_meshes \
  --compute_metrics
```

<!-- Generally, test SDF sampling strategy and regularization could affect the quality of the test reconstructions. For example, sampling aggressively near the surface could provide accurate surface details but might leave under-sampled space unconstrained, and using high L2 regularization coefficient could result in perceptually better but quantitatively worse test reconstructions. -->