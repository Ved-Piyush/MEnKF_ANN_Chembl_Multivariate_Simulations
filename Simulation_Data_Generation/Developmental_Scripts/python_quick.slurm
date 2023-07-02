#!/bin/bash
#SBATCH --time=44:00:00
#SBATCH --mem=760gb
#SBATCH --partition=guest
#SBATCH --job-name=enkf_4
#SBATCH --error=enkf_4.%J.err
#SBATCH --output=enkf_4.%J.out
#SBATCH --nodes=1
##SBATCH --gres=gpu:1
##SBATCH --constraint=gpu_80gb
pwd
source activate enkf
python get_sim_rep_results.py