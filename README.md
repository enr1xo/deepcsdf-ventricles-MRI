# DeepSDF for bi-atrial geometries

## File organization

#### setuppo
The various python scripts rely on a specific organizational structure which is completely defined and customizable in the `config.py` script. 

This sets paths for directories containing meshes files for each patient, training data, and results directories, as well as specific informations on locations of atrial regions used to process the original meshes from the specific dataset used (see specifics in How to use).

#### experiments runs
Each training run is organized under an "experiment" and "version" directory. Training only results in the creation of `.pth` files storing the trained model weights, and an `hparams.json`file recording the specifics used for that run. The only file that is required to start training (or testing) is a '.json' under the specification files directory path, which sets the hyperparameters, network architecture, and data files to be used for the experiment.

## How to use

### Processing of bi-atrial volumetric meshes

In order to use mesh data for training a DeepSDF model, the mesh will need to be pre-processed. This can be done with the `atria_data_preparing.py` executable. This script allows to:

- create `.vtu` files of volumetric meshes of the atria starting from `carp_bin` files defining the mesh
- extract epicardium and left / right endocardium surfaces from volumetric meshes of the atria in `.vtu`format

### Training and testing 