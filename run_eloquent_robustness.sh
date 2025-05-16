#!/bin/bash
#SBATCH --job-name=pairwise  # Job name
#SBATCH --output=.logs/%j.out # Name of stdout output file
#SBATCH --error=.logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G 
#SBATCH --time=5:00:00       # Run time (d-hh:mm:ss) # 4 hours for everything
#SBATCH --account=project_462000615  # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch
source .venv/bin/activate


#pip install --upgrade transformers
#pip install sacrebleu unbabel-comet deepl litellm bertscore

#install google-cloud-sdk and make sure you are signed in with the right account
# make sure to use the right project associated with the API key
#gcloud config set project 880367206890
# On the first day, it didn't work. I got 429 ressource exhaused after only one prompt, sometimes two. 
#The next day, it worked without code changes.


mkdir -p .logs

source ~/.bashrc
#huggingface-cli login --token $HF_TOKEN --add-to-git-credential


OUTPUT_DIR="eloquent"
mkdir -p $OUTPUT_DIR

MODELS=("/scratch/project_462000353/zosaelai2/models/viking-33b-synthetic-magpie-oasst2-epochs-2-batch-128-packed")
MODELS=("/scratch/project_462000353/zosaelai2/models/finnish-llama-3-8b-rated-dpo-helpsteer2-eng-fin-epochs-5-batch-64")
MODELS=(""/scratch/project_462000353/zosaelai2/models/finnish-llama-3-70b-rated-dpo-helpsteer2-eng-fin-epochs-3-batch-64"")
COT="False"
TEMP=0.3
LANGUAGES=("fi" "en")
export SSL_CERT_FILE="" # litellm does not run with this environment variable set. The value was /etc/ssl/ca-bundle.pem I restore it immediately after running the script
cot_lower=$(echo "$COT" | tr '[:upper:]' '[:lower:]')

for MODEL in "${MODELS[@]}"; do
    for LANG in "${LANGUAGES[@]}"; do
    # Extract the model name correctly using command substitution
    MODEL_NAME=$(echo $MODEL | rev | cut -d/ -f1 | rev)

    # Add _cot to the filename only if COT is enabled
    if [[ "$cot_lower" == "true" || "$cot_lower" == "1" || "$cot_lower" == "yes" || "$cot_lower" == "y" ]]; then
        OUTPUT_SUFFIX="_cot.jsonl"
    else
        OUTPUT_SUFFIX=".jsonl"
    fi

    python translation_scripts/eloquent_robustness.py --model $MODEL \
                                                --cot $COT \
                                                --output_file "${OUTPUT_DIR}/${MODEL_NAME}_${LANG}_${TEMP}${OUTPUT_SUFFIX}" \
                                                --max_tokens 1000 \
                                                --lang $LANG \
                                                --temp $TEMP

    done
done

export SSL_CERT_FILE=/etc/ssl/ca-bundle.pem # restore the SSL_CERT_FILE environment variable