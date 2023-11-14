#!/bin/bash

#BSUB -W 02:00
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -J deeponet2D

### -- Notify me by email when execution begins --
#BSUB -B
### -- Notify me by email when execution ends   --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o "/work3/nibor/data/logs/deeponet2D_%J.out"
#BSUB -e "/work3/nibor/data/logs/deeponet2D_%J.err"

export PYTHONPATH="${PYTHONPATH}:/zhome/00/4/50173/.local/bin"

module load python3/3.10.7
module load cuda/12.1.1
module load cudnn/v8.9.1.23-prod-cuda-12.X
module load tensorrt/8.6.1.6-cuda-12.X 

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$CUDA_ROOT/extras/CUPTI/lib64/"

python3 main1D2D_train.py --path_settings="scripts/twoD/settings.json"
