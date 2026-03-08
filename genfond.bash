#!/bin/bash
#
#SBATCH --partition=rleap_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

echo "Running: $@"
apptainer run --bind $PWD genfond_env.sif "$@"
