#!/bin/bash
while true; do
    # Kill existing process
    pkill -f batched_script.py
    
    # Clean null entries
    grep -v '"fluency_score": null' output.jsonl > tmp && mv tmp output.jsonl
    
    # Restart inference
    nohup python batched_script.py \
        data/output_wiki_minp-0-08_topp-1-0_temp-0-1.jsonl \
        output.jsonl \
        --temperature 0.01 \
        --max-concurrent 20 \
        --max-tokens 1500 \
        >> inference.log 2>&1 &
    
    # Wait 0.5 hour (1800 seconds)
    sleep 1800
done