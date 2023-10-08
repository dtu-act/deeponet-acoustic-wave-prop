# DEEPONET FOR ACUSTIC WAVE PROPAGATIONS

This text explains how to setup and train a DeepONet specifically for reproducing the results from the paper "Sound propagation in realistic interactive 3D scenes with parameterized sources using deep neural operators", [Nikolas Borrel-Jensen](mailto:nikolasborrel@gmail.com), Somdatta Goswami, Allan P. Engsig-Karup, George Em Karniadakis, and [Cheol-Ho Jeong](chje@dtu.dk) (PNAS).

The training, validation and testing data in 3D are generated with the Discontinuous Gallerkin Finite Element method (DG-FEM) method. For more information please refer to the [instructions](https://github.com/dtu-act/libparanumal). All scripts for generating data for reproducing the 3D results from the paper are available. The data (< 500 GB) can also be provided by contacting [Cheol-Ho Jeong](mailto:chje@dtu.dk) or [Finn T. Agerkvist](mailto:ftag@dtu.dk).

The data in 2D are generated with a Matlab [implementation](https://github.com/dtu-act/numerical-pde-solvers/tree/main/SEMSolvers), and can be downloaded from the PNAS Supplementary Information.

## INSTALLATION
Python 3.10+, Jax 0.4.10+ and Flax 0.6.10+ are used among other dependencies listed inside `scripts/install/install_packages.sh`. The code has only been tested on MacOS X.

## RUN TRAINING
To start training a DeepONet model on either 1D/2D or 3D, run 

```bash
> main1D2D_train.py --path_settings <path-to-settings-file>
> main3D_train.py --path_settings <path-to-settings-file>` 
```

The parameters `<path-to-settings-file>` should be the path to the setup settings file.

## SETUP SETTINGS FILE
To train and evaluate a DeepONet model, a setup settings file is required indicating what data to use, the neural network type and parameters to use, whether transfer learning should be applied, and more. The setup file is a JSON file and looks like

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
        "architecture": "mlp",
        "activation": "sin",
        "num_hidden_layers": 5,
        "num_hidden_neurons": 2048
    },
    "trunk_net": {
        "architecture": "mlp",
        "activation": "sin",
        "num_hidden_layers": 5,
        "num_hidden_neurons": 2048
    },
    "num_output_neurons": 100
}
```

Most of the parameters should be self-explanatory. Supported architecures for the branch net for the key `architecture` are `mlp` and `cnn`, activation functions can be either `sin`, `tanh`, or `relu`. The key `f0_feat` is given the normalized frequencies for the positional encoding for the Fourier expansions and `tmax` is the normalized physical simulation time. The data in the training and validation folders should be data in `HDF5` data format as generated with the MATLAB (1D and 2D) and libParanumal DG-FEM (3D) solvers.

It is possible to continue training a model by adding adding the following to the JSON script:

```json
{
    ...
    "transfer_learning": {
        "resume_learning": true,
        "transfer_model_path": "/path/to/the/deeponet/model/directory/"
    },
    ...
}
```

For transfer learning, one might to freeze certain layers done as follows:

```json
{
    ...
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
    ...
}
```
The indexes for the branch net `bn` and `tn` is indicating the layers to freeze and the boolean for `bn_transformer` and `tn_transformer` determines if the encoder/transformer layers for the modified MLP network should be frozen or not.

## RUN EVALUATION
For evaluating a trained model, the following Python scripts can be used:

    deeponet_wave_propagation
        ├── main1D2D_evaluate_accuracy.py
        ├── main3D_evaluate_accuracy.py
        ├── main3D_evaluate_speed.py
        ├── scripts
        │   ├── evaluate
        │   │   ├── evaluate_accuracy3D.sh
        │   │   ├── evaluate_speed3D.sh

The `*accuracy.py` scripts are used for plotting and creating `XDMF` files for visualizations in e.g. ParaView.

## DATA CONVERTERS
2D Data generated with MATLAB can be assembled (needed if generated using multiple threads in parallel) and downsampled to specific resolutions using the 2D scripts below:
    
    deeponet_wave_propagation
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

    deeponet_wave_propagation
        ├── convertH5
        │   ├── main3D_convert_dtype.py
        │   ├── main3D_DD.py
        ├── scripts
        │   ├── converters
        │   │   ├── run3D_H5_convert_dome_DD.sh
        │   │   └── run3D_H5_convert_dtype.sh

More information can be found inside these scripts.

## RE-CREATING RESULTS FROM THE PAPER
The scripts below can be used to train all DeepONet models from the paper. The training, validation and test data The data can be generated with the scripts described [here](https://github.com/dtu-act/libparanumal/tree/master/solvers/acoustics/simulationSetups/deeponet) or be provided by contacting the authors (< 500 GB).

    deeponet_wave_propagation
        │   ├── threeD
        │   │   ├── cube.json
        │   │   ├── dome_1stquad.json
        │   │   ├── dome.json
        │   │   ├── furnished.json
        │   │   ├── Lshape.json
        │   │   ├── run3D_cube.sh
        │   │   ├── run3D_dome_quarter.sh
        │   │   ├── run3D_dome.sh
        │   │   ├── run3D_furnished.sh
        │   │   ├── run3D_Lshape.sh
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
        │       ├── run_furnished_reference.sh
        │       ├── run_furnished_source.sh
        │       ├── run_furnished_target.sh
        │       ├── run_Lshape_reference.sh
        │       ├── run_Lshape_source.sh
        │       ├── run_Lshape_target.sh
        │       ├── run_rect_reference.sh
        │       ├── run_rect_source.sh
        │       └── run_rect_target.sh