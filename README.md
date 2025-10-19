# DeepONet for Acoustic Wave Propagation

Train and deploy Deep Operator Networks (DeepONets) for simulating sound propagation in 3D scenes. Based on the PNAS paper: ["Sound Propagation in Realistic Interactive 3D Scenes with Parameterized Sources Using Deep Neural Operators"](https://doi.org/10.1073/pnas.2312159120) by Borrel-Jensen et al.

## Getting started

```bash
# clone the repository
git clone git@github.com:dtu-act/deeponet-acoustic-wave-prop.git
cd deeponet-acoustic-wave-prop

# option 1 - development installation via uv (recommended)
uv sync
pre-commit install
# run ruff for formatting code
uv run ruff check . --fix && uv run ruff format .

# option 2 - development installation via pip
pip install -e .
pre-commit install
# run ruff for formatting code
uv run ruff check . --fix && uv run ruff format .
```
*Tested with Python 3.11, JAX 0.7.2, Flax 0.10.7*.

**Training:**
```bash
uv run deeponet-train --path_settings <settings.json>
```

**Inference:**
```bash
uv run deeponet-infer --path_settings <train_settings.json> --path_eval_settings <eval_settings.json>
```

## Configuration

### Training Settings (`settings.json`)

**Required fields:**
```json
{
    "id": "model_name",
    "input_dir": "/path/to/data/",
    "output_dir": "/path/to/output/",
    "training_data_dir": "train/",
    "testing_data_dir": "validation/",
    
    "tmax": 17.0,
    "f0_feat": [1.458, 0.729, 0.486],
    "normalize_data": true,
    
    "iterations": 80000,
    "learning_rate": 1e-3,
    "optimizer": "adam",
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

**Optional training parameters:**
```json
{
    "use_adaptive_weights": true,
    "decay_steps": 2000,
    "decay_rate": 0.90
}
```

**Architecture options:**
- `mod-mlp`: Modified MLP with 4 uplifting networks (paper default)
- `mlp`: Traditional MLP
- `resnet`: Residual network (smaller model, similar accuracy)

**Activation functions:** `sin`, `tanh`, `relu`

**Key parameters:**
- `f0_feat`: Normalized frequencies for positional encoding in Fourier expansions
- `tmax`: Normalized physical simulation time
- `batch_size_branch` × `batch_size_coord`: Total batch size

### Evaluation Settings

```json
{
    "model_dir": "/path/to/trained/model/",
    "validation_data_dir": "/path/to/validation/data/",
    "tmax": 17,
    
    "snap_to_grid": true,
    "write_ir_plots": true,
    "write_ir_wav": true,
    "write_full_wave_field": false,
    
    "receiver_pos_0": [[[1.66, 1.62, 1.64]]],
    "receiver_positions": ["receiver_pos_0"]
}
```

**Output options:**
- `snap_to_grid`: Align predictions to validation grid (required for comparison with reference)
- `write_ir_plots`: Generate impulse response plots
- `write_ir_animations`: Export impulse responses as GIF animations
- `write_ir_wav`: Export WAV files
- `write_full_wave_field`: Export full field in XDMF format for ParaView (time-intensive)

**Receiver positions:**
- `receiver_positions`: Array matching number of sources in HDF5 file
- Each entry references a key (e.g., `receiver_pos_0`) defining receiver coordinates
- Second dimension specifies number of evaluation points per source

### Transfer Learning

Resume training:
```json
{
    "transfer_learning": {
        "resume_learning": true,
        "transfer_model_path": "/path/to/model/"
    }
}
```

Freeze specific layers:
```json
{
    "transfer_learning": {
        "resume_learning": false,
        "transfer_model_path": "/path/to/model/",
        "freeze_layers": {
            "bn": [0, 1],
            "tn": [0],
            "bn_transformer": true,
            "tn_transformer": true
        }
    }
}
```

**Layer freezing:**
- `bn`/`tn`: Layer indices to freeze in branch/trunk networks
- `bn_transformer`/`tn_transformer`: Freeze encoder/transformer layers in modified MLP

## Data

**Training data:**
- **3D:** Generated with [DG-FEM solver](https://github.com/dtu-act/libparanumal) (~500 GB, contact [Cheol-Ho Jeong](mailto:chje@dtu.dk))
- **2D:** Generated with [MATLAB solvers](https://github.com/dtu-act/numerical-pde-solvers/tree/main/SEMSolvers) (download from PNAS Supplementary Information)

**Format:** HDF5 files containing source functions and reference solutions

### Data Preprocessing

**2D data processing:**
```
scripts/convertH5/
├── assemble_2D_sources.py        # Assemble parallel-generated data
├── downsample_2D_resolution.py   # Downsample to specific resolutions
└── split_2D_by_source.py         # Split multi-source files to single-source format

shell_scripts/converters/
├── run2D_H5_assembly.sh
└── run2D_H5_convert_resolutions.sh
```

**3D data processing:**
```
scripts/convertH5/
├── convert_3D_dtype_compact.py         # Convert float32 → float16
└── extract_3D_domain_decomposition.py  # Domain decomposition (dome geometry)

shell_scripts/converters/
├── run3D_H5_convert_dtype.sh
└── run3D_H5_convert_dome_DD.sh
```

## Reproducing Paper Results

Contact authors for training data (~500 GB) or generate using [these scripts](https://github.com/dtu-act/libparanumal/tree/master/solvers/acoustics/simulationSetups/deeponet).

Pre-configured settings available in `json_setups/`:

**3D geometries:**
```
json_setups/threeD/
├── cube.json / cube_eval.json
├── dome.json / dome_eval.json
├── dome_1stquad.json / dome_1stquad_eval.json
├── furnished.json / furnished_eval.json
└── Lshape.json / Lshape_eval.json
```

**2D transfer learning:**
```
json_setups/twoD_transfer_learning/
├── rect/
│   ├── rect3x3_source.json
│   ├── rect2x2_reference.json
│   └── rect2x2_srcpos_3ppw_target.json
├── Lshape/
│   ├── Lshape3x3_source.json
│   ├── Lshape2_5x2_5_reference.json
│   ├── Lshape2_5x2_5_srcpos_3ppw_bs600_tar.json
│   └── Lshape2_5x2_5_srcpos_5ppw_bs600_tar.json
└── furnished/
    ├── rect3x3_source.json
    ├── rect3x3_furn_bs600_reference.json
    ├── rect3x3_furn_srcpos_3ppw_bs600_tar.json
    └── rect3x3_furn_srcpos_5ppw_bs600_tar.json
```

**Example (using IBM Spectrum LSF):**
```bash
# Train cube geometry model
bsub < json_setups/threeD/train3D_cube.sh

# Evaluate with reference comparison
bsub < json_setups/threeD/evaluate3D_cube.sh
```

## Citation

If you use this code, please cite:

```bibtex
@article{borrel-jensen2023sound,
  title={Sound propagation in realistic interactive 3D scenes with parameterized sources using deep neural operators},
  author={Borrel-Jensen, Nikolas and Goswami, Somdatta and Engsig-Karup, Allan P and Karniadakis, George Em and Jeong, Cheol-Ho},
  journal={Proceedings of the National Academy of Sciences},
  volume={120},
  number={48},
  pages={e2312159120},
  year={2023},
  publisher={National Academy of Sciences}
}
```

## Contact

- **Data:** [Cheol-Ho Jeong](mailto:chje@dtu.dk)
- **Implementation/Questions:** [Nikolas Borrel-Jensen](mailto:nikolasborrel@proton.me)