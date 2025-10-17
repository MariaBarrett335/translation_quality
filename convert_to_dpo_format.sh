#!/bin/bash
#SBATCH --job-name=convert_dpo
#SBATCH --output=.logs/%j.out
#SBATCH --error=.logs/%j.err
#SBATCH --partition=standard-g
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --account=project_462000353

# Usage: sbatch convert_to_dpo_format.sh input_file.jsonl output_directory [sentence_column] [score_column] [min_score_diff] [translation_mode]

INPUT_FILE=$1
OUTPUT_DIR=$2
SENTENCE_COLUMN=${3:-"sentence"}
SCORE_COLUMN=${4:-"fluency_score"}
MIN_SCORE_DIFF=${5:-1}
TRANSLATION_MODE=${6:-"false"}

export INPUT_FILE OUTPUT_DIR SENTENCE_COLUMN SCORE_COLUMN MIN_SCORE_DIFF TRANSLATION_MODE

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch convert_to_dpo_format.sh input_file.jsonl output_directory [sentence_column] [score_column] [min_score_diff] [translation_mode]"
    exit 1
fi

echo "Starting DPO conversion: $INPUT_FILE -> $OUTPUT_DIR"
echo "Columns: $SENTENCE_COLUMN, $SCORE_COLUMN, min_diff: $MIN_SCORE_DIFF, translation: $TRANSLATION_MODE"

python3 << 'EOF'
import json
import os
import sys
from collections import defaultdict
import itertools
import random

def validate_sentence(sentence):
    return sentence and isinstance(sentence, str) and len(sentence.strip()) >= 3

def validate_dpo_entry(entry):
    try:
        required = ["prompt", "chosen", "rejected"]
        if not all(k in entry for k in required):
            return False, "Missing fields"
        if not isinstance(entry["chosen"], list) or len(entry["chosen"]) != 2:
            return False, "Invalid chosen format"
        if not isinstance(entry["rejected"], list) or len(entry["rejected"]) != 2:
            return False, "Invalid rejected format"
        return True, "Valid"
    except:
        return False, "Validation error"

def convert_to_dpo_format(input_file, output_dir, sentence_col, score_col, min_score_diff, train_split=0.8):
    is_translation = os.environ.get('TRANSLATION_MODE', 'false').lower() == 'true'
    
    print(f"Converting to DPO format - Translation mode: {is_translation}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Read and group data by first ID part (x in x-y-z)
    print("Reading and grouping data...")
    data_by_group = defaultdict(list)
    errors = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                
                # Validate required fields
                if not all(k in item for k in ['id', sentence_col, score_col]):
                    errors += 1
                    continue
                
                if is_translation and 'original_paragraph' not in item:
                    errors += 1
                    continue
                
                # Extract group ID (x part of x-y-z)
                group_id = item['id'].split('-')[0]
                
                # Validate sentence and score
                sentence = item[sentence_col]
                if not validate_sentence(sentence):
                    errors += 1
                    continue
                
                score = float(item[score_col])
                english_sentence = item.get('original_paragraph', '').strip() if is_translation else None
                
                data_by_group[group_id].append({
                    'id': item['id'],
                    'sentence': sentence.strip(),
                    'english_sentence': english_sentence,
                    'score': score
                })
                        
            except Exception as e:
                errors += 1
                
            if line_num % 5000 == 0:
                print(f"Processed {line_num} lines...")
    
    total_items = sum(len(items) for items in data_by_group.values())
    print(f"Loaded {total_items} items in {len(data_by_group)} groups. Errors: {errors}")
    
    if total_items == 0:
        print("No valid data found.")
        sys.exit(1)
    
    # Create DPO pairs
    print("Creating DPO pairs...")
    pairs_by_group = defaultdict(list)
    pairs_created = 0
    
    # Prompt templates
    if is_translation:
        templates = [
            "Which translation is more fluent for: \"{english_sentence}\"\n\nA: {sentence_a}\nB: {sentence_b}\n\nAnswer A or B:",
            "Better Danish translation of \"{english_sentence}\":\n\n1. {sentence_a}\n2. {sentence_b}\n\nAnswer 1 or 2:"
        ]
    else:
        templates = [
            "Which Danish sentence is more fluent?\n\nA: {sentence_a}\nB: {sentence_b}\n\nAnswer A or B:",
            "Choose the more fluent sentence:\n\n1. {sentence_a}\n2. {sentence_b}\n\nAnswer 1 or 2:"
        ]
    
    for group_id, items in data_by_group.items():
        if len(items) < 2:
            continue
        
        for item1, item2 in itertools.combinations(items, 2):
            score_diff = abs(item1['score'] - item2['score'])
            if score_diff < min_score_diff or item1['score'] == item2['score']:
                continue
            
            # Determine better/worse sentences
            better = item1 if item1['score'] > item2['score'] else item2
            worse = item2 if item1['score'] > item2['score'] else item1
            
            if not validate_sentence(better["sentence"]) or not validate_sentence(worse["sentence"]):
                continue
            
            # For translations, ensure same English source
            if is_translation:
                if (not better.get("english_sentence") or 
                    not worse.get("english_sentence") or
                    better["english_sentence"] != worse["english_sentence"]):
                    continue
                english_ref = better["english_sentence"]
            
            # Random position assignment
            if random.random() < 0.5:
                sentence_a, sentence_b = better["sentence"], worse["sentence"]
                correct, wrong = "A", "B"
            else:
                sentence_a, sentence_b = worse["sentence"], better["sentence"]
                correct, wrong = "B", "A"
            
            template = random.choice(templates)
            
            # Handle 1/2 format
            if "1." in template:
                correct = "1" if correct == "A" else "2"
                wrong = "2" if wrong == "A" else "1"
            
            # Create prompt
            if is_translation:
                prompt = template.format(english_sentence=english_ref, sentence_a=sentence_a, sentence_b=sentence_b)
            else:
                prompt = template.format(sentence_a=sentence_a, sentence_b=sentence_b)
            
            if len(prompt.strip()) < 50:
                continue
            
            # Create DPO entry
            dpo_entry = {
                "prompt": prompt.strip(),
                "chosen": [
                    {"role": "user", "content": prompt.strip()},
                    {"role": "assistant", "content": correct}
                ],
                "rejected": [
                    {"role": "user", "content": prompt.strip()},
                    {"role": "assistant", "content": wrong}
                ]
            }
            
            if validate_dpo_entry(dpo_entry)[0]:
                pairs_by_group[group_id].append(dpo_entry)
                pairs_created += 1
    
    print(f"Created {pairs_created} DPO pairs from {len([g for g in pairs_by_group.keys()])} groups")
    
    if pairs_created == 0:
        print("No pairs created. Check minimum score difference.")
        sys.exit(1)
    
    # GROUP-BASED TRAIN/TEST SPLIT
    print("Creating group-based train/test split...")
    group_ids = [gid for gid in pairs_by_group.keys() if pairs_by_group[gid]]
    random.shuffle(group_ids)
    
    split_idx = int(train_split * len(group_ids))
    train_groups = set(group_ids[:split_idx])
    test_groups = set(group_ids[split_idx:])
    
    train_pairs = []
    test_pairs = []
    
    for group_id in group_ids:
        pairs = pairs_by_group[group_id]
        if group_id in train_groups:
            train_pairs.extend(pairs)
        else:
            test_pairs.extend(pairs)
    
    random.shuffle(train_pairs)
    random.shuffle(test_pairs)
    
    print(f"Split: {len(train_groups)} train groups ({len(train_pairs)} pairs), {len(test_groups)} test groups ({len(test_pairs)} pairs)")
    
    # Write files
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    test_file = os.path.join(output_dir, 'test.jsonl')
    with open(test_file, 'w', encoding='utf-8') as f:
        for pair in test_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"Files written: {train_file}, {test_file}")
    print("Group-based splitting prevents data leakage between train/test")
    
    return pairs_created

# Main execution
if __name__ == "__main__":
    random.seed(42)
    
    input_file = os.environ.get('INPUT_FILE')
    output_dir = os.environ.get('OUTPUT_DIR')
    sentence_col = os.environ.get('SENTENCE_COLUMN')
    score_col = os.environ.get('SCORE_COLUMN')
    min_score_diff = int(os.environ.get('MIN_SCORE_DIFF'))
    
    try:
        total_pairs = convert_to_dpo_format(input_file, output_dir, sentence_col, score_col, min_score_diff)
        print(f"SUCCESS: {total_pairs} preference pairs created")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

EOF

echo "Job completed"