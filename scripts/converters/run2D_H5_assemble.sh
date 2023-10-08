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

# 2D: when running MATLAB in parallel with multiple threads, each source position is written to separate files.
#     We need to assemble the data into one file for the 2D Python code to process the data. Call this script 
#     before converting resolutions
python3 convertH5/main2D_assembly_H5.py # for 2D Matlab data only