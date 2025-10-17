#!/bin/bash
#SBATCH --job-name=pairwise  # Job name
#SBATCH --output=.logs/%j.out # Name of stdout output file
#SBATCH --error=.logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G 
#SBATCH --time=2:00:00       # Run time (d-hh:mm:ss) # 4 hours for everything
#SBATCH --account=project_462000615  # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch
source .venv/bin/activate

mkdir -p .logs

source ~/.bashrc
#huggingface-cli login --token $HF_TOKEN --add-to-git-credential

python fluency_eval.py --model_path /scratch/project_462000353/maribarr/alignment-handbook/data/llama-3-8b-fluency_all-20k-epochs-5-batch-64/checkpoint-264 --dataset_path /scratch/project_462000353/maribarr/translation_quality/data/sft_format/fluency_all/test.jsonl
#python fluency_eval.py --model_path meta-llama/Llama-3.1-8B-Instruct --dataset_path /scratch/project_462000353/maribarr/translation_quality/data/sft_format/fluency_all/test.jsonl