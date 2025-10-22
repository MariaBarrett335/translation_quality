# salloc --account=project_462000963 --partition=dev-g --ntasks=1 --gres=gpu:mi250:8 --time=3:00:00 --mem=0
# srun --account=project_462000963 --partition=dev-g --ntasks=1 --gres=gpu:mi250:8 --time=3:00:00 --mem=0 --pty bash
# srun --account=project_462000353 --partition=dev-g --ntasks=1 --gpus-per-node=8 --nodes=1 --time=00:30:00 --mem=480G --pty bash
# srun --pty bash
# reconnect:
# srun --jobid=<your_allocation_jobid> --pty bash
# use this
# srun --pty --overlap --jobid YOUR-JOBID bash 


#MODEL=${1:-"/scratch/project_462000353/zosaelai2/models/Llama-3.3-70B-Instruct"}
MODEL=${1:-"/scratch/project_462000963/users/maribarr/alignment-handbook/data/llama3-70b_pairwise-fluency/checkpoint-200"}
TENSOR_PARALLEL_SIZE=${2:-4}
MAX_MODEL_LEN=16384

# In your vLLM server script, add these environment variables:
export TRITON_CACHE_DIR="/scratch/project_462000353/maribarr/.triton_cache"
export XDG_CACHE_HOME="/scratch/project_462000353/maribarr/.cache"

# Create the directories
mkdir -p $TRITON_CACHE_DIR
mkdir -p $XDG_CACHE_HOME

module use /appl/local/csc/modulefiles
module load pytorch/2.5
export SSL_CERT_FILE=$(python -m certifi)

mkdir -p logs

nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    > "logs/vllm_server_sft.log" 2>&1 &

echo "Check tail -f logs/vllm_server.log"