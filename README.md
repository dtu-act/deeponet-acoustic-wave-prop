# DEEPONET FOR ACUSTIC WAVE PROPAGATIONS

This text explains how to setup and train a DeepONet specifically for reproducing the results from the paper "Sound propagation in realistic interactive 3D scenes with parameterized sources using deep neural operators", [Nikolas Borrel-Jensen](mailto:nikolasborrel@gmail.com), Somdatta Goswami, Allan P. Engsig-Karup, George Em Karniadakis, and [Cheol-Ho Jeong](chje@dtu.dk) (PNAS).

The training, validation and testing data in 3D are generated with the Discontinuous Gallerkin Finite Element method (DG-FEM) method. For more information please refer to the [instructions](https://github.com/dtu-act/libparanumal). All scripts for generating data for reproducing the 3D results from the paper are available. The data (< 500 GB) can also be provided by contacting [Cheol-Ho Jeong](mailto:chje@dtu.dk) or [Finn T. Agerkvist](mailto:ftag@dtu.dk).

The data in 2D are generated with a Matlab [implementation](https://github.com/dtu-act/numerical-pde-solvers/tree/main/SEMSolvers), and can be downloaded from the PNAS Supplementary Information.

## INSTALLATION
The results from the paper was done using Python 3.10+, Jax 0.4.10+ and Flax 0.6.10+:

```
pip install --user --upgrade optax tensorboard tensorboard_plugin_profile pytorch_lightning  matplotlib smt pydot graphviz h5py tqdm meshio "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax orbax-checkpoint
```
You can also take a look at the `./requirements.txt` file located at the root directory for specific versions used.

## RUN TRAINING
To start training a DeepONet model on either 1D/2D or 3D, run 

```bash
> main1D2D_train.py --path_settings <path-to-settings-file>
> main3D_train.py --path_settings <path-to-settings-file>` 
```

The parameters `<path-to-settings-file>` should be the path to the setup settings file explained in the next section.

## SETUP SETTINGS FILE
To train a DeepONet model, a setup settings file is required indicating what data to use, the neural network type and parameters to use, whether transfer learning should be applied, and more. The setup file is a JSON file and looks like

```json
{
    "id": "<id>",

    "input_dir": "/path/to/input/data/",
    "output_dir": "/path/to/output/",

    "training_data_dir": "<training-dir-name>",
    "testing_data_dir": "<validation-dir-name>",

    "tmax": 17.0,
    
    "f0_feat": [1.458,0.729,0.486],
    "normalize_data": true,

    "iterations": 80000,
    "use_adaptive_weights": true,
    "decay_steps": 2000,
    "decay_rate": 0.90,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    
    "__comment1__": "total batch_size is a multiple of branch and coordinate sizes",
    "batch_size_branch": 64,
    "batch_size_coord": 1000,

    "branch_net": {
        "architecture": "mod-mlp",
        "activation": "sin",
        "num_hidden_layers": 5,
        "num_hidden_neurons": 2048
    },
    "trunk_net": {
        "architecture": "mod-mlp",
        "activation": "sin",
        "num_hidden_layers": 5,
        "num_hidden_neurons": 2048
    },
    "num_output_neurons": 100
}
```
Most of the parameters should be self-explanatory. 

Supported `architecture` are 
* `mod-mlp`: the modified MLP architecture (using 4 uplifting networks) from [N. Borrel-Jensen, S. Goswami, A. P. Engsig-Karup, G. E. Karniadakis, C.-H. Jeong](https://doi.org/10.1073/pnas.2312159120).
* `mlp`: the traditional MLP architecture. *NOTE* that the original scripts for reproducing the results from [N. Borrel-Jensen, S. Goswami, A. P. Engsig-Karup, G. E. Karniadakis, C.-H. Jeong](https://doi.org/10.1073/pnas.2312159120) mapped the `mlp` keyword to the now `mod-mlp` type, so you would have to update the scripts.
* `resnet`: the residual neural network implemented as a series of convolutional neural net with skip connections as explained in [N. Borrel-Jensen, A. P. Engsig-Karup, Cheol-Ho Jeong](http://dx.doi.org/10.61782/fa.2023.0930)

Activation functions can be either `sin`, `tanh`, or `relu`. The key `f0_feat` is given the normalized frequencies for the positional encoding for the Fourier expansions and `tmax` is the normalized physical simulation time. The data in the training and validation folders should be data in `HDF5` data format as generated with the MATLAB (1D and 2D) and libParanumal DG-FEM (3D) solvers.

It is possible to continue training a model by adding adding the following to the JSON script:

```json
{
    "transfer_learning": {
        "resume_learning": true,
        "transfer_model_path": "/path/to/the/deeponet/model/directory/"
    },
}
```

For transfer learning, one might to freeze certain layers done as follows:

```json
{
    "transfer_learning": {
        "resume_learning": false,
        "transfer_model_path": "/path/to/the/deeponet/model/directory/",
        "freeze_layers": {
            "bn": [0,1],
            "tn": [0],
            "bn_transformer": true,
            "tn_transformer": true
        }
    },
}
```
The indexes for the branch net `bn` and `tn` is indicating the layers to freeze and the boolean for `bn_transformer` and `tn_transformer` determines if the encoder/transformer layers for the modified MLP network should be frozen or not.

## RUN EVALUATION
For evaluating a trained model, the following Python scripts can be used (see also section "RE-CREATING RESULTS FROM THE PAPER"):

    .
       ├── main1D2D_eval.py       
       ├── eval3D.py
       ├── main3D_eval.py
       ├── main3D_eval_speed.py
       ├── scripts
       │   ├── evaluate
       │   │   ├── evaluate_speed3D.sh

A `JSON` script is used to specify how to evaluate the model and an example is given below:

```JSON
{
    "model_dir": "/path/to/trained/model/dir",    
    "validation_data_dir": "/path/to/validation/data/dir",

    "tmax": 17,

    "write_full_wave_field": true,

    "snap_to_grid": true,
    "write_ir_plots": true,
    "write_ir_animations": false,
    "write_ir_wav": true,        

    "receiver_positions": [
        [[1.66256893, 1.61655235, 1.64047122]],
        [[1.52937829, 1.57425201, 1.57134283]],
        [[1.53937507, 1.50955164, 1.48763454]],
        [[0.33143353, 0.33566815, 0.36886978]],
        [[0.42988288, 0.43229115, 0.43867755]]
    ]
}
```
If many identical receivers for different source positions are to be evaluated, the `JSON` script can also be setup as

```JSON
{
    "model_dir": "/path/to/trained/model/dir",    
    "validation_data_dir": "/path/to/validation/data/dir",

    "tmax": 17,

    "write_full_wave_field": true,

    "snap_to_grid": true,
    "write_ir_plots": true,
    "write_ir_animations": false,
    "write_ir_wav": true,        

    "receiver_pos_0": [
        [[1.66256893, 1.61655235, 1.64047122]]
    ],

    "receiver_positions": [
        "receiver_pos_0",
        "receiver_pos_0",
        "receiver_pos_0",
        "receiver_pos_0",
        "receiver_pos_0"
    ]
}
where the entries in `receiver_positions` are keys to another entry in the `JSON` script.
```

* `model_dir`: the path to the model checkpoint to load.
* `validation_data_dir`: the path to a `HDH5` test file containing the source functions for which to evaluate the model as well as reference solution data.
* `tmax`: the length of the impulse response predictions (normalized in seconds). 
* `receiver_positions`: the receiver positions where the impulse responses should be evaluated, where the first dimension should correspond to the number of sources in the `HDF5`; the second dimension determines the number of receiver positions to evaluate for the given source. 
* `write_full_wave_field`: whether the full predicted wave field should be written to disk (in the `XDMF` format for visualizations in e.g. ParaView). The wave field will be predicted in a grid determined by the validation data. Note that predicting the full wave field can be time consuming.
* `snap_to_grid`: whether the predicted impulse response positions should be adjusted to the nearest grid point determined by the validation data. To compare with the reference solution, this field should be set to `True`.
* `write_ir_plots`: whether impulse responses should be plotted and written to disk.
* `write_ir_animations`: whether the impulse responses should be written as gif animations over time.
* `write_ir_wav`: whether the impulse responses should be written as `WAV` files.

## DATA CONVERTERS
2D Data generated with MATLAB can be assembled (needed if generated using multiple threads in parallel) and downsampled to specific resolutions using the 2D scripts below:
    
    .
       ├── convertH5
       │   ├── assembly2D.py
       │   ├── convert2D_resolutions.py
       │   ├── main2D_assembly.py
       │   ├── main2D_convert_resolutions.py
       ├── scripts
       │   ├── converters
       │   │   ├── run2D_H5_assemble.sh
       │   │   ├── run2D_H5_convert_resolutions.sh

Data generated with the libParanumal DG-FEM solver are generated with the resolution needed, but scripts for converting from float 32-bit to float 16-bit and scripts for extracting data for domain decomposition (i.e. for the dome geometry) can be done with the scripts below:

    .
       ├── convertH5
       │   ├── main3D_convert_dtype.py
       │   ├── main3D_DD.py
       ├── scripts
       │   ├── converters
       │   │   ├── run3D_H5_convert_dome_DD.sh
       │   │   └── run3D_H5_convert_dtype.sh

More information can be found inside these scripts.

## RE-CREATING RESULTS FROM THE PAPER
The scripts below can be used to train and evaluate all DeepONet models from the paper (adjust the data paths to match your local system).

    .
    │   ├── threeD
    │   │   ├── evaluate3D_cube.sh
    │   │   ├── evaluate3D_dome_quarter.sh
    │   │   ├── evaluate3D_dome.sh
    │   │   ├── evaluate3D_furnished.sh
    │   │   ├── evaluate3D_Lshape.sh
    │   │   ├── evaluate3D_speed.sh
    │   │   ├── setups
    │   │   │   ├── cube_eval.json
    │   │   │   ├── cube.json
    │   │   │   ├── dome_1stquad_eval.json
    │   │   │   ├── dome_1stquad.json
    │   │   │   ├── dome_eval.json
    │   │   │   ├── dome.json
    │   │   │   ├── furnished_eval.json
    │   │   │   ├── furnished.json
    │   │   │   ├── Lshape_eval.json
    │   │   │   ├── Lshape.json
    │   │   │   └── settings.json
    │   │   ├── train3D_cube.sh
    │   │   ├── train3D_dome_quarter.sh
    │   │   ├── train3D_dome.sh
    │   │   ├── train3D_furnished.sh
    │   │   ├── train3D_Lshape.sh
    │   └── twoD_transfer_learning
    │       ├── furnished
    │       │   ├── rect3x3_furn_bs600_reference.json
    │       │   ├── rect3x3_furn_srcpos_3ppw_bs600_tar.json
    │       │   ├── rect3x3_furn_srcpos_5ppw_bs600_tar.json
    │       │   └── rect3x3_source.json
    │       ├── Lshape
    │       │   ├── Lshape2_5x2_5_reference.json
    │       │   ├── Lshape2_5x2_5_srcpos_3ppw_bs600_tar.json
    │       │   ├── Lshape2_5x2_5_srcpos_5ppw_bs600_tar.json
    │       │   └── Lshape3x3_source.json
    │       ├── rect
    │       │   ├── rect2x2_reference.json
    │       │   ├── rect2x2_srcpos_3ppw_target.json
    │       │   └── rect3x3_source.json
    │       ├── train2D_furnished_reference.sh
    │       ├── train2D_furnished_source.sh
    │       ├── train2D_furnished_target.sh
    │       ├── train2D_Lshape_reference.sh
    │       ├── train2D_Lshape_source.sh
    │       ├── train2D_Lshape_target.sh
    │       ├── train2D_rect_reference.sh
    │       ├── train2D_rect_source.sh
    │       └── train2D_rect_target.sh
    ├── main1D2D_eval.py -- for evaluating 2D transfer learning (modify manually to point to the model of interest)

E.g., for training a DeepONet model for the cube geometry (using IBM Spectrum LSF), run
```bash
> bsub < scripts/threeD/train3D_cube.sh
```
and to evaluate the accuracy generating plots comparing against a reference solution, run
```bash
> bsub < scripts/threeD/evaluate3D_cube.sh
```

The training, validation and test data can be generated with the scripts described [here](https://github.com/dtu-act/libparanumal/tree/master/solvers/acoustics/simulationSetups/deeponet) or be provided by contacting the authors (> 500 GB).