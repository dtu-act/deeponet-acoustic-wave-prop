#!/bin/bash

#BSUB -W 24:00
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process:gmem=30GB"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "select[localssd0]"
#BSUB -J Lshape3D

### -- Notify me by email when execution begins --
#BSUB -B
### -- Notify me by email when execution ends   --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o "/work3/nibor/data/logs/deeponet_Lshape3D_%J.out"
#BSUB -e "/work3/nibor/data/logs/deeponet_Lshape3D_%J.err"

export PYTHONPATH="${PYTHONPATH}:/zhome/00/4/50173/.local/bin"

module load python3/3.10.7
module load cuda/12.1.1
module load cudnn/v8.9.1.23-prod-cuda-12.X
module load tensorrt/8.6.1.6-cuda-12.X 

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$CUDA_ROOT/extras/CUPTI/lib64/"

mkdir -p /localssd0/nibor/
rsync -a --delete --ignore-existing /work3/nibor/1TB/input3D/Lshape_1000hz_p6_6ppw_srcs5165_train /localssd0/nibor/ || { echo 'copying training data failed' ; exit 1; }
rsync -a --delete --ignore-existing /work3/nibor/1TB/input3D/Lshape_1000hz_p4_5ppw_srcs180_val /localssd0/nibor/ || { echo 'copying validation data failed' ; exit 1; }
echo ls /localssd0/nibor/

python3 main3D_train.py --path_settings="scripts/threeD/setups/Lshape.json" || rm -rf /localssd0/nibor/

rm -rf /localssd0/nibor/ || { echo 'removing data failed failed' ; exit 1; }