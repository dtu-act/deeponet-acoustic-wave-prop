#!/bin/bash

#BSUB -W 02:00
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=128GB]"
#BSUB -J convertH5

### -- Notify me by email when execution begins --
#BSUB -B
### -- Notify me by email when execution ends   --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o "/work3/nibor/data/logs/convertH5_%J.out"
#BSUB -e "/work3/nibor/data/logs/convertH5_%J.err"

export PYTHONPATH="${PYTHONPATH}:/zhome/00/4/50173/.local/bin"
module load python3/3.10.7

# 2D: converting from high spatial/temporal resolutions to lower resolutions (data generated with MATLAB)
python3 convertH5/main2D_convert_resolutions_H5.py # used for 2D sensibility analysis paper