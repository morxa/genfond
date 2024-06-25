#!/bin/bash
#
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --cpus-per-task=32
#SBATCH --mem=248000
#SBATCH --output=%x-%j.out

echo "Running: $@"
apptainer run --bind $PWD genfond_env.sif "$@"
