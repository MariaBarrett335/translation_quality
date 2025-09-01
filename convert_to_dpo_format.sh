#!/bin/bash
#SBATCH --job-name=convert_dpo
#SBATCH --output=.logs/%j.out # Name of stdout output file
#SBATCH --error=.logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --time=01:00:00
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --account=project_462000353  # Project for billing

# Usage: sbatch convert_to_dpo_format.sh input_file.jsonl output_directory [sentence_column] [score_column] [min_score_diff] [translation_mode]

# Get command line arguments
INPUT_FILE=$1
OUTPUT_DIR=$2
SENTENCE_COLUMN=${3:-"sentence"}
SCORE_COLUMN=${4:-"fluency_score"}
MIN_SCORE_DIFF=${5:-1}
TRANSLATION_MODE=${6:-"false"}

export INPUT_FILE
export OUTPUT_DIR  
export SENTENCE_COLUMN
export SCORE_COLUMN
export MIN_SCORE_DIFF
export TRANSLATION_MODE

# Validate required arguments
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch convert_to_dpo_format.sh input_file.jsonl output_directory [sentence_column] [score_column] [min_score_diff] [translation_mode]"
    echo ""
    echo "Arguments:"
    echo "  input_file.jsonl  - Input JSONL file with fluency ratings"
    echo "  output_directory  - Output directory for train.jsonl and test.jsonl"
    echo "  sentence_column   - Column name for sentence text (default: 'sentence')"
    echo "  score_column      - Column name for fluency score (default: 'fluency_score')"
    echo "  min_score_diff    - Minimum score difference for comparisons (default: 1)"
    echo "  translation_mode  - Set to 'true' for translation task, 'false' for fluency (default: 'false')"
    echo ""
    echo "Examples:"
    echo "  # Basic fluency comparison"
    echo "  sbatch convert_to_dpo_format.sh fluency_data.jsonl output_dir"
    echo ""
    echo "  # Translation fluency comparison (requires 'english_sentence' column)"
    echo "  sbatch convert_to_dpo_format.sh translation_data.jsonl output_dir sentence fluency_score 1 true"
    echo ""
    echo "  # Custom settings for fluency task"
    echo "  sbatch convert_to_dpo_format.sh data.jsonl output_dir text score 2 false"
    exit 1
fi

echo "Starting DPO format conversion..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Sentence column: $SENTENCE_COLUMN"
echo "Score column: $SCORE_COLUMN"
echo "Minimum score difference: $MIN_SCORE_DIFF"
echo "Translation mode: $TRANSLATION_MODE"
echo ""

# Run the Python conversion script
python3 << 'EOF'
import json
import os
import sys
from datetime import datetime
from collections import defaultdict
import itertools
import random

def validate_sentence(sentence):
    """Validate that a sentence is not empty and has meaningful content."""
    if not sentence:
        return False
    if not isinstance(sentence, str):
        return False
    # Check if sentence has meaningful content (not just whitespace)
    cleaned = sentence.strip()
    if not cleaned:
        return False
    # Check if it's too short to be meaningful
    if len(cleaned) < 3:
        return False
    return True

def validate_dpo_entry(entry):
    """Validate that a DPO entry has all required fields and valid content."""
    try:
        # Check basic structure
        if not isinstance(entry, dict):
            return False, "Entry is not a dictionary"
        
        # Check prompt field
        if "prompt" not in entry or not entry["prompt"]:
            return False, "Missing or empty prompt field"
        
        if not isinstance(entry["prompt"], str) or not entry["prompt"].strip():
            return False, "Prompt field is empty or not string"
        
        # Check chosen field
        if "chosen" not in entry or not entry["chosen"]:
            return False, "Missing or empty chosen field"
        
        if not isinstance(entry["chosen"], list) or len(entry["chosen"]) != 2:
            return False, "Chosen field is not a valid list or doesn't have 2 messages"
        
        # Check chosen messages
        user_msg = entry["chosen"][0]
        assistant_msg = entry["chosen"][1]
        
        if not isinstance(user_msg, dict) or user_msg.get("role") != "user":
            return False, "First chosen message is not user role"
        
        if not isinstance(assistant_msg, dict) or assistant_msg.get("role") != "assistant":
            return False, "Second chosen message is not assistant role"
        
        if not user_msg.get("content") or not assistant_msg.get("content"):
            return False, "Chosen messages missing content"
        
        # Check rejected field
        if "rejected" not in entry or not entry["rejected"]:
            return False, "Missing or empty rejected field"
        
        if not isinstance(entry["rejected"], list) or len(entry["rejected"]) != 2:
            return False, "Rejected field is not a valid list or doesn't have 2 messages"
        
        # Check rejected messages
        user_msg = entry["rejected"][0]
        assistant_msg = entry["rejected"][1]
        
        if not isinstance(user_msg, dict) or user_msg.get("role") != "user":
            return False, "First rejected message is not user role"
        
        if not isinstance(assistant_msg, dict) or assistant_msg.get("role") != "assistant":
            return False, "Second rejected message is not assistant role"
        
        if not user_msg.get("content") or not assistant_msg.get("content"):
            return False, "Rejected messages missing content"
        
        return True, "Valid entry"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def convert_to_dpo_format(input_file, output_dir, sentence_col, score_col, min_score_diff, train_split=0.8):
    """
    Convert fluency rating data to DPO format where model picks the most fluent sentence
    """
    # Read translation mode from environment variable
    translation_mode = os.environ.get('TRANSLATION_MODE', 'false')
    is_translation = translation_mode.lower() == 'true'
    
    print("=" * 70)
    print("CONVERTING TO DPO FORMAT - FLUENCY COMPARISON TASK")
    print("=" * 70)
    print(f"Task: Model picks the most fluent sentence from two options")
    print(f"Sentence column: '{sentence_col}'")
    print(f"Score column: '{score_col}'")
    print(f"Minimum score difference: {min_score_diff}")
    print(f"Translation mode: {is_translation}")
    print("=" * 70)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Read input data and group by x (first part of id x-y-z)
    print(f"📖 Reading data from: {input_file}")
    print(f"📋 Grouping sentences by first ID part (x in x-y-z format)")
    data_by_group = defaultdict(list)
    errors = 0
    missing_sentence_col = 0
    missing_score_col = 0
    missing_id = 0
    missing_english_col = 0
    invalid_sentences = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    
                    # Check if required columns exist
                    if 'id' not in item:
                        missing_id += 1
                        if missing_id <= 5:
                            print(f"⚠️  Line {line_num}: Missing 'id' field")
                        continue
                        
                    if sentence_col not in item:
                        missing_sentence_col += 1
                        if missing_sentence_col <= 5:
                            print(f"⚠️  Line {line_num}: Missing sentence column '{sentence_col}'")
                        continue
                        
                    if score_col not in item:
                        missing_score_col += 1
                        if missing_score_col <= 5:
                            print(f"⚠️  Line {line_num}: Missing score column '{score_col}'")
                        continue
                    
                    # Check for English sentence if in translation mode
                    english_sentence = None
                    if is_translation:
                        if 'english_sentence' not in item:
                            missing_english_col += 1
                            if missing_english_col <= 5:
                                print(f"⚠️  Line {line_num}: Missing 'english_sentence' column (required for translation mode)")
                            continue
                        english_sentence = item['english_sentence']
                        if not validate_sentence(english_sentence):
                            invalid_sentences += 1
                            if invalid_sentences <= 5:
                                print(f"⚠️  Line {line_num}: Invalid English sentence: '{english_sentence}'")
                            continue
                    
                    # Extract group ID (x part of x-y-z format)
                    # Only sentences with same x will be compared
                    try:
                        id_parts = item['id'].split('-')
                        if len(id_parts) < 1:
                            print(f"⚠️  Line {line_num}: ID too short: {item.get('id', 'N/A')}")
                            errors += 1
                            continue
                        group_id = id_parts[0]  # Get the 'x' part from 'x-y-z'
                    except (IndexError, AttributeError):
                        print(f"⚠️  Line {line_num}: Invalid ID format (expected x-y-z): {item.get('id', 'N/A')}")
                        errors += 1
                        continue
                    
                    # Validate sentence content
                    sentence = item[sentence_col]
                    if not validate_sentence(sentence):
                        invalid_sentences += 1
                        if invalid_sentences <= 5:
                            print(f"⚠️  Line {line_num}: Invalid sentence: '{sentence}'")
                        continue
                    
                    # Validate score
                    try:
                        score = float(item[score_col])
                    except (ValueError, TypeError):
                        print(f"⚠️  Line {line_num}: Invalid score value: {item.get(score_col, 'N/A')}")
                        errors += 1
                        continue
                    
                    # Add to group
                    data_by_group[group_id].append({
                        'id': item['id'],
                        'sentence': sentence.strip(),  # Ensure sentence is cleaned
                        'english_sentence': english_sentence.strip() if english_sentence else None,
                        'score': score,
                        'original_item': item
                    })
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON error on line {line_num}: {e}")
                    errors += 1
                except Exception as e:
                    print(f"❌ Error processing line {line_num}: {e}")
                    errors += 1
                    
                if line_num % 5000 == 0:
                    print(f"   Processed {line_num} lines...")
                    
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        sys.exit(1)
    
    total_items = sum(len(items) for items in data_by_group.values())
    print(f"✅ Successfully loaded {total_items} valid examples in {len(data_by_group)} groups")
    if errors > 0:
        print(f"⚠️  Skipped {errors} entries with missing/invalid data")
    if missing_id > 0:
        print(f"⚠️  {missing_id} entries missing 'id' field")
    if missing_sentence_col > 0:
        print(f"⚠️  {missing_sentence_col} entries missing sentence column '{sentence_col}'")
    if missing_score_col > 0:
        print(f"⚠️  {missing_score_col} entries missing score column '{score_col}'")
    if missing_english_col > 0:
        print(f"⚠️  {missing_english_col} entries missing 'english_sentence' column")
    if invalid_sentences > 0:
        print(f"⚠️  {invalid_sentences} entries with invalid sentences")
    
    if total_items == 0:
        print("❌ No valid data found. Exiting.")
        sys.exit(1)
    
    # Show group statistics
    group_sizes = [len(items) for items in data_by_group.values()]
    print(f"📊 Group statistics (by first ID part 'x'):")
    print(f"   Total groups (unique x values): {len(data_by_group)}")
    print(f"   Groups with 1 item: {sum(1 for size in group_sizes if size == 1)}")
    print(f"   Groups with 2+ items: {sum(1 for size in group_sizes if size >= 2)}")
    print(f"   Average items per group: {sum(group_sizes) / len(group_sizes):.1f}")
    print(f"   Max items in a group: {max(group_sizes)}")
    
    # Show some example groups
    print(f"📋 Example groups:")
    sample_groups = list(data_by_group.items())[:3]
    for group_id, items in sample_groups:
        print(f"   Group '{group_id}': {len(items)} sentences")
        for item in items[:2]:  # Show first 2 IDs in each group
            print(f"     - ID: {item['id']}")
        if len(items) > 2:
            print(f"     - ... and {len(items)-2} more")
    # Create DPO preference pairs for fluency selection task
    print(f"🔄 Creating DPO fluency selection pairs...")
    print(f"📋 Rule: Only comparing sentences within same group (same x in x-y-z)")
    print(f"📋 Task: Given two sentences, model should select the more fluent one")
    dpo_pairs = []
    groups_processed = 0
    pairs_created = 0
    validation_failed = 0
    skipped_low_diff = 0
    
    # Track statistics separately since we're not storing metadata
    score_differences = []
    correct_answers_stats = []
    
    # Prompt templates for fluency selection task
    if is_translation:
        prompt_templates = [
            "Which of these sentences is a more fluent translation of this English sentence: \"{english_sentence}\"\n\nA: {sentence_a}\nB: {sentence_b}\n\nAnswer only with the letter A or B:",
            "Given the English sentence: \"{english_sentence}\"\n\nWhich Danish translation is more fluent?\n\nA: {sentence_a}\nB: {sentence_b}\n\nRespond with only A or B:",
            "Compare these two Danish translations of \"{english_sentence}\":\n\nA: {sentence_a}\nB: {sentence_b}\n\nWhich is more fluent? Answer A or B only:",
            "English: \"{english_sentence}\"\n\nWhich Danish translation sounds more natural?\n\nOption A: {sentence_a}\nOption B: {sentence_b}\n\nAnswer: (A or B only)",
            "For the English sentence \"{english_sentence}\", choose the better Danish translation:\n\n1. {sentence_a}\n2. {sentence_b}\n\nAnswer with only the number (1 or 2):",
        ]
    else:
        prompt_templates = [
            "Which of these two Danish sentences is more fluent?\n\nA: {sentence_a}\nB: {sentence_b}\n\nAnswer with only the letter (A or B):",
            "Compare the fluency of these Danish sentences and choose the more fluent one:\n\nSentence A: {sentence_a}\nSentence B: {sentence_b}\n\nRespond with only A or B:",
            "Between these two Danish sentences, which one sounds more fluent?\n\nOption A: {sentence_a}\nOption B: {sentence_b}\n\nAnswer: (A or B only)",
            "Please select the more fluent Danish sentence:\n\nA) {sentence_a}\nB) {sentence_b}\n\nYour answer (letter only):",
            "Evaluate these two Danish sentences for fluency and choose the better one:\n\n1. {sentence_a}\n2. {sentence_b}\n\nAnswer with only the number (1 or 2):",
        ]
    
    for group_id, items in data_by_group.items():
        if len(items) < 2:
            continue  # Need at least 2 items to create pairs within the same group
            
        groups_processed += 1
        
        # Create all possible pairs within this specific group (same x)
        # This ensures we only compare sentences that share the same first ID part
        for item1, item2 in itertools.combinations(items, 2):
            score_diff = abs(item1['score'] - item2['score'])
            
            # Only create pairs if score difference meets minimum threshold
            if score_diff < min_score_diff:
                skipped_low_diff += 1
                continue
                
            # Determine which is more fluent (higher score) and less fluent (lower score)
            if item1['score'] > item2['score']:
                more_fluent_item = item1
                less_fluent_item = item2
            elif item2['score'] > item1['score']:
                more_fluent_item = item2
                less_fluent_item = item1
            else:
                continue  # Skip if scores are equal
            
            # Validate sentences before creating entry
            more_fluent_sentence = more_fluent_item["sentence"]
            less_fluent_sentence = less_fluent_item["sentence"]
            
            if not validate_sentence(more_fluent_sentence) or not validate_sentence(less_fluent_sentence):
                validation_failed += 1
                continue
            
            # For translation mode, ensure both items have the same English sentence
            if is_translation:
                more_fluent_english = more_fluent_item.get("english_sentence")
                less_fluent_english = less_fluent_item.get("english_sentence")
                
                if not more_fluent_english or not less_fluent_english:
                    validation_failed += 1
                    continue
                
                if more_fluent_english.strip() != less_fluent_english.strip():
                    validation_failed += 1
                    continue  # Skip if English sentences don't match
                
                english_reference = more_fluent_english.strip()
            
            # Randomly assign sentences to A/B positions to avoid position bias
            # This ensures the more fluent sentence appears equally in A and B positions
            position_random = random.random()
            if position_random < 0.5:
                # More fluent sentence goes to position A
                sentence_a = more_fluent_sentence
                sentence_b = less_fluent_sentence
                correct_answer = "A"
                wrong_answer = "B"
            else:
                # More fluent sentence goes to position B  
                sentence_a = less_fluent_sentence
                sentence_b = more_fluent_sentence
                correct_answer = "B"
                wrong_answer = "A"
            
            # Use random prompt template for diversity
            prompt_template = random.choice(prompt_templates)
            
            # Handle different answer formats (A/B vs 1/2)
            if "1." in prompt_template and "2." in prompt_template:
                if correct_answer == "A":
                    correct_answer = "1"
                    wrong_answer = "2"
                else:
                    correct_answer = "2"
                    wrong_answer = "1"
            
            # Create prompt content
            if is_translation:
                prompt_content = prompt_template.format(
                    english_sentence=english_reference,
                    sentence_a=sentence_a,
                    sentence_b=sentence_b
                )
            else:
                prompt_content = prompt_template.format(
                    sentence_a=sentence_a,
                    sentence_b=sentence_b
                )
            
            # Additional validation for prompt content
            if not prompt_content or len(prompt_content.strip()) < 50:
                validation_failed += 1
                continue
            
            # Create response content:
            # - Chosen response: correct answer (points to more fluent sentence)
            # - Rejected response: wrong answer (points to less fluent sentence)
            chosen_response = correct_answer
            rejected_response = wrong_answer
            
            # Validate responses
            if not chosen_response or not rejected_response:
                validation_failed += 1
                continue
            
            # Create DPO format entry (without metadata to match expected format)
            dpo_entry = {
                "prompt": prompt_content.strip(),
                "chosen": [
                    {
                        "role": "user",
                        "content": prompt_content.strip()
                    },
                    {
                        "role": "assistant", 
                        "content": chosen_response
                    }
                ],
                "rejected": [
                    {
                        "role": "user",
                        "content": prompt_content.strip()
                    },
                    {
                        "role": "assistant",
                        "content": rejected_response
                    }
                ]
            }
            
            # Store metadata separately for logging purposes only
            metadata = {
                "more_fluent_score": more_fluent_item['score'],
                "less_fluent_score": less_fluent_item['score'],
                "score_difference": score_diff,
                "group_id": group_id,
                "more_fluent_id": more_fluent_item['id'],
                "less_fluent_id": less_fluent_item['id'],
                "more_fluent_sentence": more_fluent_sentence,
                "less_fluent_sentence": less_fluent_sentence,
                "english_sentence": english_reference if is_translation else None,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "correct_answer": correct_answer,
                "prompt_used": prompt_content.strip(),
                "task_type": "translation" if is_translation else "fluency"
            }
            
            # Final validation of the complete DPO entry
            is_valid, error_msg = validate_dpo_entry(dpo_entry)
            if not is_valid:
                validation_failed += 1
                if validation_failed <= 5:
                    print(f"⚠️  Validation failed for group {group_id}: {error_msg}")
                continue
            
            dpo_pairs.append(dpo_entry)
            pairs_created += 1
            
            # Track statistics for reporting
            score_differences.append(score_diff)
            correct_answers_stats.append(correct_answer)
        
        if groups_processed % 100 == 0:
            print(f"   Processed {groups_processed} groups (unique x values), created {pairs_created} pairs...")
    
    print(f"✅ Created {len(dpo_pairs)} DPO fluency selection pairs from {groups_processed} groups")
    print(f"📋 Each pair: model chooses between two sentences (A/B or 1/2)")
    if validation_failed > 0:
        print(f"⚠️  {validation_failed} entries failed validation and were skipped")
    if skipped_low_diff > 0:
        print(f"⚠️  {skipped_low_diff} pairs skipped due to low score difference (< {min_score_diff})")
    
    if len(dpo_pairs) == 0:
        print("❌ No preference pairs could be created. Check your minimum score difference setting.")
        sys.exit(1)
    
    # Show score difference distribution (using tracked statistics)
    if score_differences:
        print(f"📊 Score difference distribution:")
        for diff in sorted(set(score_differences)):
            count = score_differences.count(diff)
            percentage = count / len(score_differences) * 100
            print(f"   Difference {diff:.1f}: {count} pairs ({percentage:.1f}%)")
    
    # Show answer distribution to verify balancing
    print(f"📊 Response distribution (verifying position balance):")
    if correct_answers_stats:
        for answer in sorted(set(correct_answers_stats)):
            count = correct_answers_stats.count(answer)
            percentage = count / len(correct_answers_stats) * 100
            print(f"   More fluent sentence in position {answer}: {count} pairs ({percentage:.1f}%)")
        
        # Verify balance (should be roughly 50/50)
        if len(set(correct_answers_stats)) >= 2:
            balance_diff = abs(correct_answers_stats.count('A') - correct_answers_stats.count('B')) if 'A' in correct_answers_stats and 'B' in correct_answers_stats else 0
            if '1' in correct_answers_stats and '2' in correct_answers_stats:
                balance_diff = max(balance_diff, abs(correct_answers_stats.count('1') - correct_answers_stats.count('2')))
            if balance_diff <= len(dpo_pairs) * 0.1:  # Within 10% is good balance
                print(f"   ✅ Good position balance achieved (difference: {balance_diff})")
            else:
                print(f"   ⚠️  Position imbalance detected (difference: {balance_diff})")
    
    # Shuffle data for better distribution
    random.shuffle(dpo_pairs)
    
    # Create train/test split
    print(f"📊 Creating train/test split ({train_split*100:.0f}%/{(1-train_split)*100:.0f}%)...")
    
    split_idx = int(train_split * len(dpo_pairs))
    train_pairs = dpo_pairs[:split_idx]
    test_pairs = dpo_pairs[split_idx:]
    
    print(f"   Train pairs: {len(train_pairs)}")
    print(f"   Test pairs: {len(test_pairs)}")
    
    # Create output directory
    print(f"📁 Creating output directory: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        sys.exit(1)
    
    # Write train split
    train_file = os.path.join(output_dir, 'train.jsonl')
    print(f"💾 Writing training data to: {train_file}")
    try:
        with open(train_file, 'w', encoding='utf-8') as f:
            for pair in train_pairs:
                # Final validation before writing
                is_valid, _ = validate_dpo_entry(pair)
                if is_valid:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                else:
                    print(f"⚠️  Skipping invalid entry during write")
    except Exception as e:
        print(f"❌ Error writing train file: {e}")
        sys.exit(1)
    
    # Write test split
    test_file = os.path.join(output_dir, 'test.jsonl')
    print(f"💾 Writing test data to: {test_file}")
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            for pair in test_pairs:
                # Final validation before writing
                is_valid, _ = validate_dpo_entry(pair)
                if is_valid:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                else:
                    print(f"⚠️  Skipping invalid entry during write")
    except Exception as e:
        print(f"❌ Error writing test file: {e}")
        sys.exit(1)
    
    # Verification
    print(f"🔍 Verifying output files...")
    try:
        train_lines = sum(1 for _ in open(train_file, 'r'))
        test_lines = sum(1 for _ in open(test_file, 'r'))
        
        print(f"   Train file: {train_lines} lines")
        print(f"   Test file: {test_lines} lines")
        
        # Sample validation of written files
        print(f"🔍 Validating written data...")
        sample_valid = 0
        sample_total = 0
        
        with open(train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Check first 10 entries
                    break
                try:
                    entry = json.loads(line.strip())
                    is_valid, error_msg = validate_dpo_entry(entry)
                    if is_valid:
                        sample_valid += 1
                    else:
                        print(f"⚠️  Sample validation failed: {error_msg}")
                    sample_total += 1
                except Exception as e:
                    print(f"⚠️  Sample parse error: {e}")
                    sample_total += 1
        
        print(f"   Sample validation: {sample_valid}/{sample_total} entries valid")
        
        if sample_valid == sample_total and sample_total > 0:
            print("✅ Verification passed!")
        else:
            print("⚠️  Warning: Some validation issues found")
            
    except Exception as e:
        print(f"⚠️  Could not verify files: {e}")
    
    # Show example of correct DPO format
    print(f"\n📝 Example DPO format (Fluency Selection Task):")
    if train_pairs:
        example = train_pairs[0]
        # Create a clean example without metadata for display
        clean_example = {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }
        print(json.dumps(clean_example, ensure_ascii=False, indent=2))
        print(f"\nMetadata for this example:")
        print(f"  More fluent sentence: {example['metadata']['more_fluent_sentence']}")
        print(f"  Less fluent sentence: {example['metadata']['less_fluent_sentence']}")
        if is_translation and example['metadata']['english_sentence']:
            print(f"  English reference: {example['metadata']['english_sentence']}")
        print(f"  Score difference: {example['metadata']['score_difference']:.1f}")
        print(f"  Correct answer: {example['metadata']['correct_answer']}")
        print(f"  Group ID (x): {example['metadata']['group_id']}")
        print(f"  Sentence A: {example['metadata']['sentence_a']}")
        print(f"  Sentence B: {example['metadata']['sentence_b']}")
    
    task_name = "translation fluency" if is_translation else "fluency selection"
    print(f"\n✅ DPO {task_name} conversion completed successfully!")
    print(f"📂 Output directory: {output_dir}")
    print(f"📁 Files created:")
    print(f"   - train.jsonl ({len(train_pairs)} preference pairs)")
    print(f"   - test.jsonl ({len(test_pairs)} preference pairs)")
    if is_translation:
        print(f"\n🔧 Task: Model learns to select the more fluent translation")
        print(f"🔧 Input: English sentence + two Danish translation options")
    else:
        print(f"\n🔧 Task: Model learns to select the more fluent sentence from two options")
    print(f"🔧 Constraint: Only sentences with same first ID part (x) are compared")
    print(f"🔧 Format: prompt/chosen/rejected format compatible with alignment-handbook DPO training")
    
    return len(dpo_pairs)

# Main execution
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"🚀 Process started at: {start_time}")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    input_file = os.environ.get('INPUT_FILE')
    output_dir = os.environ.get('OUTPUT_DIR') 
    sentence_col = os.environ.get('SENTENCE_COLUMN')
    score_col = os.environ.get('SCORE_COLUMN')
    min_score_diff = int(os.environ.get('MIN_SCORE_DIFF'))
    translation_mode = os.environ.get('TRANSLATION_MODE', 'false')
    
    try:
        total_pairs = convert_to_dpo_format(input_file, output_dir, sentence_col, score_col, min_score_diff)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 SUCCESS!")
        print(f"⏰ Completed at: {end_time}")
        print(f"⏱️  Duration: {duration}")
        print(f"📊 Total preference pairs created: {total_pairs}")
        print(f"\n📋 Next steps:")
        print(f"   1. Update your DPO config to point to: {output_dir}")
        print(f"   2. Run DPO training with: scripts/run_dpo.py your_dpo_config.yaml")
        if translation_mode.lower() == 'true':
            print(f"\n🎯 Training objective: Model will learn to select the more fluent translation")
            print(f"🎯 Input format: English sentence + two Danish translation options")
        else:
            print(f"\n🎯 Training objective: Model will learn to select the more fluent sentence")
        print(f"🎯 Comparison scope: Only between sentences with same first ID part (x)")
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

EOF

echo ""
echo "🏁 Job completed at: $(date)"