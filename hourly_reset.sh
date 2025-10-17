# Configuration variables
OUTPUT_FILE="output_pairwise_sample.jsonl"
INPUT_FILE="data/pairwise_sample.jsonl"
PROMPT_FILE="prompts/empty.md"

#!/bin/bash
while true; do
    # Kill existing process more gracefully
    #pkill -TERM -f batched_script.py
    pkill -TERM -f deepseek_batch_inference.py
    sleep 5  # Give process time to finish current operations
    #pkill -KILL -f batched_script.py  # Force kill if still running
    pkill -KILL -f deepseek_batch_inference.py  # Force kill if still running
    
    # Clean null entries
    #grep -v '"fluency_score": null' output.jsonl > tmp && mv tmp output.jsonl
    grep -v '"output": null' "$OUTPUT_FILE" | grep -v '^null$' > tmp && mv tmp "$OUTPUT_FILE"

    # Restart inference
     nohup python deepseek_batch_inference.py \
        "$INPUT_FILE" \
        "$PROMPT_FILE" \
        "$OUTPUT_FILE" \
        --temperature 0.01 \
        --max-concurrent 25 \
        --max-tokens 1500 \
        >> inference.log 2>&1 &
    
    #nohup python batched_script.py \
        #data/pairwise_sample.jsonl \
        #output_pairwise_sample.jsonl \
        #--prompt-column prompt \
        #--id-column id \
        #--temperature 0.01 \
        #--max-concurrent 25 \
        #--max-tokens 1500 \
        #> inference.log 2>&1 &
    
    
    # Wait 0.5 hour (1800 seconds)
    sleep 1800
done