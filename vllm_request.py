# Run on the same node as the vllm server is running

"""
srun --pty --overlap --jobid YOUR-JOBID bash 

export SSL_CERT_FILE=$(python -m certifi)

module use /appl/local/csc/modulefiles/
module load pytorch
#source .venv/bin/activate
source .handbook_venv/bin/activate
mkdir -p .logs


"""
import json
from openai import OpenAI
import os 
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize OpenAI client to vLLM server
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

input_file = "/scratch/project_462000353/maribarr/translation_quality/data/sft_format/pairwise_all_no_id_leakage/held-out-test.jsonl"
#input_file = "/scratch/project_462000353/maribarr/translation_quality/pairwise_sample.jsonl"
output_file = "/scratch/project_462000353/maribarr/translation_quality/output/llama-3-3-70b_SFT_full_test.jsonl"
#output_file = "/scratch/project_462000353/maribarr/translation_quality/output/llama-3-3-70b_SFT_test_sample.jsonl"

MODEL = "/scratch/project_462000353/zosaelai2/models/Llama-3.3-70B-Instruct"
MODEL = "/scratch/project_462000963/users/maribarr/alignment-handbook/data/llama3-70b_pairwise-fluency/checkpoint-200"

@dataclass
class PairwiseEvaluationResults:
    """Container for pairwise evaluation metrics"""
    accuracy: float
    total_samples: int
    correct_predictions: int
    tie_rate: float
    ties_detected: int
    a_predictions: int
    b_predictions: int
    a_rate: float
    b_rate: float
    predictions: List[str]
    true_labels: List[str]
    ties: List[bool]
    confusion_matrix: np.ndarray
    classification_report: str

def predict_pairwise_choice_from_logprobs(top_logprobs, debug=False):
    """
    Predict A/B choice using log probabilities with same method as fluency_eval.py
    
    Args:
        top_logprobs: Dictionary of token: log_probability pairs
        debug: Whether to print debug information
        
    Returns:
        Tuple of (predicted_choice, is_tie)
    """
    try:
        # Define candidate tokens (same as fluency_eval.py)
        a_candidates = ["A", " A", "\nA", "_A"]
        b_candidates = ["B", " B", "\nB", "_B"]
        
        # Find probabilities for A and B tokens
        a_probs = []
        b_probs = []
        
        for token, logprob in top_logprobs.items():
            # Strip token before matching
            stripped_token = token.strip()
            
            # Check A candidates
            if stripped_token in a_candidates or token in a_candidates:
                a_probs.append(logprob)
                if debug:
                    print(f"Found A candidate: '{token}' (stripped: '{stripped_token}') -> {logprob}")
            
            # Check B candidates  
            if stripped_token in b_candidates or token in b_candidates:
                b_probs.append(logprob)
                if debug:
                    print(f"Found B candidate: '{token}' (stripped: '{stripped_token}') -> {logprob}")
        
        # Get best probability for each choice
        best_a_prob = max(a_probs) if a_probs else float('-inf')
        best_b_prob = max(b_probs) if b_probs else float('-inf')
        
        if debug:
            print(f"Best A probability: {best_a_prob}")
            print(f"Best B probability: {best_b_prob}")
        
        # Handle ties and missing tokens
        if best_a_prob == float('-inf') and best_b_prob == float('-inf'):
            # Neither A nor B found in logprobs
            if debug:
                print("Neither A nor B found in logprobs - defaulting to tie")
            return "A", True  # Default with tie flag
        elif best_a_prob == float('-inf'):
            # Only B found
            return "B", False
        elif best_b_prob == float('-inf'):
            # Only A found  
            return "A", False
        else:
            # Both found - compare probabilities
            prob_diff = abs(best_a_prob - best_b_prob)
            is_tie = prob_diff < 0.1  # Small difference threshold for ties
            
            if best_a_prob > best_b_prob:
                return "A", is_tie
            else:
                return "B", is_tie
                
    except Exception as e:
        if debug:
            print(f"Error in logprob prediction: {e}")
        return "A", True  # Default with tie flag on error

def evaluate_results(output_file: str):
    """Evaluate the results from the output file"""
    print("\n" + "="*60)
    print("RUNNING EVALUATION ON OUTPUT FILE")
    print("="*60)
    
    # Load samples from output file
    samples = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num + 1}: {e}")
                    continue
    
    print(f"Loaded {len(samples)} samples from {output_file}")
    
    if not samples:
        print("No samples found for evaluation!")
        return
    
    # Process samples
    predictions = []
    true_labels = []
    is_tie_list = []
    
    processed_count = 0
    skipped_count = 0
    
    for i, sample in enumerate(samples):
        # Extract from logprobs
        if "top_logprobs" not in sample or not sample["top_logprobs"]:
            skipped_count += 1
            continue
        
        # Get the first token's logprobs
        top_logprobs = sample["top_logprobs"][0] if isinstance(sample["top_logprobs"], list) else sample["top_logprobs"]
        
        # Predict using logprobs
        predicted_choice, is_tie = predict_pairwise_choice_from_logprobs(
            top_logprobs, 
            debug=(i < 5)  # Debug first 5 samples
        )
        
        # Get ground truth
        correct_answer = sample.get("correct_answer", None)
        if correct_answer is None:
            skipped_count += 1
            continue
        
        predictions.append(predicted_choice)
        true_labels.append(correct_answer)
        is_tie_list.append(is_tie)
        processed_count += 1
    
    print(f"Processed {processed_count} samples for evaluation, skipped {skipped_count} samples")
    
    if not predictions:
        print("No valid samples found for evaluation!")
        return
    
    # Calculate metrics
    pred_array = np.array(predictions)
    true_array = np.array(true_labels)
    tie_array = np.array(is_tie_list)
    
    # Basic metrics
    total_samples = len(predictions)
    correct_predictions = np.sum(pred_array == true_array)
    accuracy = correct_predictions / total_samples
    
    # Tie statistics
    ties_detected = np.sum(tie_array)
    tie_rate = ties_detected / total_samples
    
    # Prediction distribution
    a_predictions = np.sum(pred_array == "A")
    b_predictions = np.sum(pred_array == "B")
    a_rate = a_predictions / total_samples
    b_rate = b_predictions / total_samples
    
    # Confusion matrix and classification report
    labels = ["A", "B"]
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    cr = classification_report(true_labels, predictions, labels=labels, target_names=labels)
    
    # Print results
    print("\n" + "="*60)
    print("PAIRWISE FLUENCY JUDGMENT EVALUATION RESULTS")
    print("="*60)
    
    print(f"Dataset size: {total_samples}")
    print()
    
    print("ACCURACY METRICS:")
    print(f"  Overall Accuracy:     {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Correct Predictions:  {correct_predictions}/{total_samples}")
    print()
    
    print("TIE DETECTION:")
    print(f"  Ties Detected:        {ties_detected} ({tie_rate*100:.1f}%)")
    print(f"  Tie Rate:             {tie_rate:.4f}")
    print()
    
    print("PREDICTION DISTRIBUTION:")
    print(f"  A Predictions:        {a_predictions} ({a_rate*100:.1f}%)")
    print(f"  B Predictions:        {b_predictions} ({b_rate*100:.1f}%)")
    print()
    
    print("CONFUSION MATRIX:")
    print(f"               Predicted")
    print(f"               A     B")
    print(f"    Actual A   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"           B   {cm[1,0]:4d}  {cm[1,1]:4d}")
    print()
    
    print("CLASSIFICATION REPORT:")
    print(cr)
    
    # Save evaluation results
    results_file = output_file.replace('.jsonl', '_evaluation_results.json')
    evaluation_summary = {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "tie_rate": tie_rate,
        "ties_detected": ties_detected,
        "a_predictions": a_predictions,
        "b_predictions": b_predictions,
        "a_rate": a_rate,
        "b_rate": b_rate,
        "confusion_matrix": cm.tolist(),
        "classification_report": cr
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Evaluation results saved to: {results_file}")

# Main processing code
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load already processed IDs from output file
processed_ids = set()
if os.path.exists(output_file):
    print(f"📄 Checking existing output file for processed IDs...")
    with open(output_file, "r", encoding="utf-8") as existing:
        for line in existing:
            try:
                data = json.loads(line.strip())
                if 'id' in data:
                    processed_ids.add(data['id'])
                elif 'sample_idx' in data:  # Fallback for index-based IDs
                    processed_ids.add(data['sample_idx'])
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    print(f"✅ Found {len(processed_ids)} already processed IDs")

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "a", encoding="utf-8") as fout:
    total_samples = 0
    skipped_samples = 0
    processed_samples = 0
    
    for idx, line in enumerate(fin):
        sample = json.loads(line)
        total_samples += 1
        
        # Handle different ID formats
        if 'id' in sample:
            sample_id = sample['id']
        else:
            sample_id = idx  # Use line index as ID
            
        # Skip if already processed
        if sample_id in processed_ids:
            print(f"⏭️  Skipping {sample_id} (already processed)")
            skipped_samples += 1
            continue
            
        print(f"🔄 Processing {sample_id}")
        
        # Handle different message formats
        if "messages" in sample and len(sample["messages"]) > 0:
            user_message = sample["messages"][0]["content"]
        elif "prompt" in sample:
            user_message = sample["prompt"]
        else:
            print(f"❌ No valid message format found for sample {sample_id}")
            continue

        try:
            # Request completion with logprobs
            response = client.completions.create(
                model=MODEL,
                prompt=user_message,
                max_tokens=1,
                temperature=0.1,
                logprobs=20,
            )
            
            choice = response.choices[0]

            # Save relevant info
            output_data = {
                "prompt": user_message,
                "completion": choice.text,
                "tokens": choice.logprobs.tokens,
                "token_logprobs": choice.logprobs.token_logprobs,
                "top_logprobs": choice.logprobs.top_logprobs,
                "id": sample_id,  # Use the determined ID
                "sample_idx": idx,  # Also save the index
                "correct_answer": sample.get('correct_answer', None)  # Use .get() to avoid KeyError
            }

            fout.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            fout.flush()  # Force write to disk immediately
            processed_samples += 1
            
            # Add to processed set to avoid duplicates in this run
            processed_ids.add(sample_id)
            
            # Print progress every 10 samples
            if processed_samples % 10 == 0:
                print(f"✅ Processed {processed_samples} new samples ({skipped_samples} skipped)")
                
        except Exception as e:
            print(f"❌ Error processing {sample_id}: {e}")
            continue

print(f"🎉 Processing complete!")
print(f"📊 Summary:")
print(f"   - Total samples in input: {total_samples}")
print(f"   - Already processed (skipped): {skipped_samples}")
print(f"   - Newly processed: {processed_samples}")
print(f"   - Output saved to: {output_file}")

# Run evaluation on the output file
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    evaluate_results(output_file)
else:
    print("⚠️  No output file found or file is empty - skipping evaluation")