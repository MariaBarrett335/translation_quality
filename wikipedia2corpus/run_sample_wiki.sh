#!/bin/bash
#SBATCH --job-name=sample_wiki  # Job name
#SBATCH --output=.logs/%j.out # Name of stdout output file
#SBATCH --error=.logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G 
#SBATCH --time=1:00:00       # Run time (d-hh:mm:ss) # 4 hours for everything
#SBATCH --account=project_462000963  # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch
source ../.venv/bin/activate

mkdir -p .logs

python sample_wiki.py