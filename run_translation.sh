#!/bin/bash
#SBATCH --job-name=translate  # Job name
#SBATCH --output=.logs/%j.out # Name of stdout output file
#SBATCH --error=.logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G 
#SBATCH --time=01:00:00       # Run time (d-hh:mm:ss) # 4 hours for everything
#SBATCH --account=project_462000615  # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch
source .venv/bin/activate
 
#pip install sacrebleu unbabel-comet deepl litellm

#install google-cloud-sdk and make sure you are signed in with the right account
# make sure to use the right project associated with the API key
#gcloud config set project 880367206890
# On the first day, it didn't work. I got 429 ressource exhaused after only one prompt, sometimes two. 
#The next day, it worked without code changes.

mkdir -p .logs

export OPENAI_API_KEY=xxx
export HF_HOME=/scratch/project_462000353/cache

# Define the arguments
#SOURCE_FILE="/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
#ENCH_NAME="mt_bench"
#TRANSLATE_COL='turns'

SOURCE_FILE="translations/back_translation/fr.jsonl"
BENCH_NAME='back_translation'
TRANSLATE_COL='source_sentence_mt-translated'

#SOURCE_FILE="hf://datasets/CohereForAI/m-ArenaHard/en/test-00000-of-00001.parquet"
#ENCH_NAME='arenahard'
#RANSLATE_COL='prompt'

#SOURCE_FILE="hf://datasets/basicv8vc/SimpleQA/simple_qa_test_set.csv"
#BENCH_NAME='simpleqa'
#RANSLATE_COL='problem'
#TRANSLATE_COL2='answer' 

OUTPUT_DIR="translations/${BENCH_NAME}"
mkdir -p $OUTPUT_DIR
LANGUAGES=("da" "de" "fi" "el" "fr" "pl")
LANGUAGES=("da" "fi")
export SSL_CERT_FILE="" # litellm does not run with this environment variable set. The value was /etc/ssl/ca-bundle.pem I restore it immediately after running the script

# Iterate over each language and run the translation script
for TGT_LANG in "${LANGUAGES[@]}"; do
    echo "Translating to $TGT_LANG..."
    python translation_scripts/translate_list_str.py --source-file $SOURCE_FILE \
                                --output-dir $OUTPUT_DIR \
                                --tgt-lang $TGT_LANG \
                                --source-file  $SOURCE_FILE \
                                --translate-col $TRANSLATE_COL \
                                --src_lang fr
                                #--max-samples 5 \
      
    
    echo "---"
done

export SSL_CERT_FILE=/etc/ssl/ca-bundle.pem # restore the SSL_CERT_FILE environment variable