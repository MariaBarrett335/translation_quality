#!/usr/bin/env python3
"""
Evaluation script for pairwise fluency judgment results from vLLM output.
Analyzes the logprob-based A/B predictions and calculates metrics.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse


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


class PairwiseEvaluator:
    def __init__(self, debug: bool = False):
        """Initialize the pairwise evaluator"""
        self.debug = debug
    
    def load_vllm_output(self, output_file: str) -> List[Dict]:
        """
        Load vLLM output JSONL file.
        
        Args:
            output_file: Path to the vLLM output JSONL file
            
        Returns:
            List of processed samples
        """
        samples = []
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        if self.debug:
                            print(f"Warning: Could not parse line {line_num + 1}: {e}")
                        continue
        
        print(f"Loaded {len(samples)} samples from {output_file}")
        return samples
    
    def process_samples(self, samples: List[Dict]) -> Tuple[List[str], List[str], List[bool]]:
        """
        Process samples to extract predictions and ground truth.
        
        Args:
            samples: List of vLLM output samples
            
        Returns:
            Tuple of (predictions, true_labels, is_tie_list)
        """
        predictions = []
        true_labels = []
        is_tie_list = []
        
        processed_count = 0
        skipped_count = 0
        
        for i, sample in enumerate(samples):
            # Check if we already have a prediction in the sample
            if "predicted_choice" in sample and "is_tie" in sample:
                # Use existing prediction
                predicted_choice = sample["predicted_choice"]
                is_tie = sample["is_tie"]
            else:
                # Extract from logprobs
                if "top_logprobs" not in sample or not sample["top_logprobs"]:
                    if self.debug:
                        print(f"Warning: No top_logprobs found for sample {i}")
                    skipped_count += 1
                    continue
                
                # Get the first token's logprobs
                top_logprobs = sample["top_logprobs"][0] if isinstance(sample["top_logprobs"], list) else sample["top_logprobs"]
                
                # Predict using logprobs
                predicted_choice, is_tie = predict_pairwise_choice_from_logprobs(
                    top_logprobs, 
                    debug=(self.debug and i < 5)  # Debug first 5 samples
                )
            
            # Get ground truth
            correct_answer = sample.get("correct_answer", None)
            if correct_answer is None:
                if self.debug:
                    print(f"Warning: No correct_answer found for sample {i}")
                skipped_count += 1
                continue
            
            predictions.append(predicted_choice)
            true_labels.append(correct_answer)
            is_tie_list.append(is_tie)
            processed_count += 1
        
        print(f"Processed {processed_count} samples, skipped {skipped_count} samples")
        return predictions, true_labels, is_tie_list
    
    def calculate_metrics(self, predictions: List[str], true_labels: List[str], is_tie_list: List[bool]) -> PairwiseEvaluationResults:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: List of predicted choices ("A" or "B")
            true_labels: List of correct choices ("A" or "B") 
            is_tie_list: List of tie flags
            
        Returns:
            PairwiseEvaluationResults with all metrics
        """
        # Convert to numpy arrays
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
        
        return PairwiseEvaluationResults(
            accuracy=accuracy,
            total_samples=total_samples,
            correct_predictions=correct_predictions,
            tie_rate=tie_rate,
            ties_detected=ties_detected,
            a_predictions=a_predictions,
            b_predictions=b_predictions,
            a_rate=a_rate,
            b_rate=b_rate,
            predictions=predictions,
            true_labels=true_labels,
            ties=is_tie_list,
            confusion_matrix=cm,
            classification_report=cr
        )
    
    def print_results(self, results: PairwiseEvaluationResults):
        """Print evaluation results in a nice format"""
        print("\n" + "="*60)
        print("PAIRWISE FLUENCY JUDGMENT EVALUATION RESULTS")
        print("="*60)
        
        print(f"Dataset size: {results.total_samples}")
        print()
        
        print("ACCURACY METRICS:")
        print(f"  Overall Accuracy:     {results.accuracy:.4f} ({results.accuracy*100:.1f}%)")
        print(f"  Correct Predictions:  {results.correct_predictions}/{results.total_samples}")
        print()
        
        print("TIE DETECTION:")
        print(f"  Ties Detected:        {results.ties_detected} ({results.tie_rate*100:.1f}%)")
        print(f"  Tie Rate:             {results.tie_rate:.4f}")
        print()
        
        print("PREDICTION DISTRIBUTION:")
        print(f"  A Predictions:        {results.a_predictions} ({results.a_rate*100:.1f}%)")
        print(f"  B Predictions:        {results.b_predictions} ({results.b_rate*100:.1f}%)")
        print()
        
        print("CONFUSION MATRIX:")
        print(f"               Predicted")
        print(f"               A     B")
        print(f"    Actual A   {results.confusion_matrix[0,0]:4d}  {results.confusion_matrix[0,1]:4d}")
        print(f"           B   {results.confusion_matrix[1,0]:4d}  {results.confusion_matrix[1,1]:4d}")
        print()
        
        print("CLASSIFICATION REPORT:")
        print(results.classification_report)
    
    def save_detailed_results(self, results: PairwiseEvaluationResults, output_dir: str):
        """Save detailed results including predictions"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save predictions to CSV
        df = pd.DataFrame({
            'predicted': results.predictions,
            'true_label': results.true_labels,
            'is_tie': results.ties,
            'is_correct': [p == t for p, t in zip(results.predictions, results.true_labels)]
        })
        df.to_csv(output_path / 'pairwise_predictions.csv', index=False)
        
        # Save metrics summary - convert NumPy types to Python native types
        metrics_summary = {
            "accuracy": float(results.accuracy),
            "total_samples": int(results.total_samples),
            "correct_predictions": int(results.correct_predictions),
            "tie_rate": float(results.tie_rate),
            "ties_detected": int(results.ties_detected),
            "a_predictions": int(results.a_predictions),
            "b_predictions": int(results.b_predictions),
            "a_rate": float(results.a_rate),
            "b_rate": float(results.b_rate),
            "confusion_matrix": results.confusion_matrix.tolist(),  # This already converts to list
            "classification_report": results.classification_report
        }
        
        with open(output_path / 'metrics_summary.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate pairwise fluency judgment predictions from vLLM output")
    parser.add_argument("--output_file", required=True, help="Path to the vLLM output JSONL file")
    parser.add_argument("--results_dir", default="./pairwise_evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PairwiseEvaluator(debug=args.debug)
    
    # Load vLLM output
    samples = evaluator.load_vllm_output(args.output_file)
    
    if not samples:
        print("No samples found in output file!")
        return
    
    # Process samples to get predictions and ground truth
    predictions, true_labels, is_tie_list = evaluator.process_samples(samples)
    
    if not predictions:
        print("No valid samples found for evaluation!")
        return
    
    # Calculate metrics
    results = evaluator.calculate_metrics(predictions, true_labels, is_tie_list)
    
    # Print results
    evaluator.print_results(results)
    
    # Save detailed results
    evaluator.save_detailed_results(results, args.results_dir)


if __name__ == "__main__":
    main()