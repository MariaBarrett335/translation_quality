#!/usr/bin/env python3
"""
Evaluation script for Danish pairwise fluency judgment models.
Uses only log probabilities to compare A vs B choices.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


@dataclass
class PairwiseEvaluationResults:
    """Container for pairwise evaluation metrics"""
    accuracy: float
    total_examples: int
    correct_predictions: int
    tie_count: int
    non_tie_examples: int
    predictions: List[str]
    true_answers: List[str]
    user_prompts: List[str]
    is_tie_list: List[bool]


class DanishPairwiseEvaluator:
    def __init__(self, model_path: str, device: str = "auto", debug: bool = False):
        """
        Initialize the evaluator with a HuggingFace instruction-tuned model.
        
        Args:
            model_path: Path to the HuggingFace model
            device: Device to run inference on ("auto", "cuda", "cpu")
            debug: Whether to print debug information during evaluation
        """
        self.device = self._setup_device(device)
        self.debug = debug
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory optimization (version that worked)
        if self.device.type == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_buffers=True,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)
            
        self.model.eval()
        print("Model loaded successfully!")
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"Model config: {self.model.config.model_type if hasattr(self.model.config, 'model_type') else 'Unknown'}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup the appropriate device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load JSONL dataset with pairwise comparison format.
        
        Args:
            dataset_path: Path to the JSONL file
            
        Returns:
            List of parsed conversation examples
        """
        examples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        print(f"Loaded {len(examples)} examples from {dataset_path}")
        return examples

    def predict_pairwise_choice(self, user_prompt: str) -> Tuple[str, bool]:
        """
        Predict which choice (A or B) the model would select using only log probabilities.
        
        Args:
            user_prompt: The user's prompt asking for a choice between A and B
            
        Returns:
            Tuple of (predicted_choice, is_tie)
        """
        try:
            # Format exactly like the training data: messages with assistant turn started
            messages = [{"role": "user", "content": user_prompt}]
            
            # Use the chat template to format properly, with generation prompt
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Tokenize the formatted prompt
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2000)
            
            # Handle device placement for inputs based on model loading method
            if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                # Model uses device_map, send inputs to the first device in the map
                first_device = next(iter(self.model.hf_device_map.values()))
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
            else:
                # Model is on a single device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if self.debug:
                print(f"Formatted prompt: {formatted_prompt[-200:]}")  # Last 200 chars
            
            # Get token IDs for A and B - try multiple variations
            a_tokens = {}
            b_tokens = {}
            
            a_candidates = ["A", " A", "\nA", "A)", "(A)", "A.", "A:"]
            b_candidates = ["B", " B", "\nB", "B)", "(B)", "B.", "B:"]
            
            for candidate in a_candidates:
                try:
                    token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
                    if len(token_ids) == 1:  # Single token preferred
                        a_tokens[candidate] = token_ids[0]
                        if self.debug:
                            print(f"Found A token for '{candidate}': {token_ids[0]}")
                        break
                except:
                    continue
            
            for candidate in b_candidates:
                try:
                    token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
                    if len(token_ids) == 1:  # Single token preferred
                        b_tokens[candidate] = token_ids[0]
                        break
                except:
                    continue
            
            # Fallback if no single tokens found
            if not a_tokens:
                token_ids = self.tokenizer.encode("A", add_special_tokens=False)
                a_tokens["A"] = token_ids[0]
            if not b_tokens:
                token_ids = self.tokenizer.encode("B", add_special_tokens=False)
                b_tokens["B"] = token_ids[0]
            
            # Compute logits for next token
            with torch.no_grad():
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[0, -1, :].clone()
                
                # Mask out end-of-turn tokens to force consideration of A/B
                eot_tokens_to_mask = [
                    "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", 
                    "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>"
                ]
                
                for token_str in eot_tokens_to_mask:
                    try:
                        token_id = self.tokenizer.encode(token_str, add_special_tokens=False)[0]
                        next_token_logits[token_id] = float('-inf')
                        if self.debug:
                            print(f"Masked token: {token_str}")
                    except:
                        continue
            
            # Get probabilities for A and B tokens
            a_prob = 0.0
            b_prob = 0.0
            
            for candidate, token_id in a_tokens.items():
                prob = torch.softmax(next_token_logits, dim=-1)[token_id].item()
                if prob > a_prob:
                    a_prob = prob
            
            for candidate, token_id in b_tokens.items():
                prob = torch.softmax(next_token_logits, dim=-1)[token_id].item()
                if prob > b_prob:
                    b_prob = prob
            
            # Check for ties
            is_tie = abs(a_prob - b_prob) < 1e-8
            
            if self.debug:
                print(f"A probability: {a_prob:.10f}")
                print(f"B probability: {b_prob:.10f}")
                if is_tie:
                    print("TIE DETECTED - using fallback to A")
                
                # Show top 10 tokens for debugging
                top_probs, top_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), 10)
            
            # Return prediction and tie status
            if is_tie:
                return "A", True
            else:
                return ("A" if a_prob > b_prob else "B"), False
            
        except Exception as e:
            if self.debug:
                print(f"Prediction failed: {e}")
                import traceback
                traceback.print_exc()
            return "A", False  # fallback

    def evaluate_dataset(self, dataset_path: str) -> PairwiseEvaluationResults:
        """
        Evaluate the model on the entire pairwise comparison dataset.
        
        Args:
            dataset_path: Path to the evaluation dataset
            
        Returns:
            PairwiseEvaluationResults object with all metrics
        """
        examples = self.load_dataset(dataset_path)
        
        predictions = []
        true_answers = []
        user_prompts = []
        is_tie_list = []
        tie_count = 0
        
        print("Running pairwise predictions...")
        for i, example in enumerate(examples):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(examples)}")
            
            messages = example["messages"]
            user_prompt = messages[0]["content"]
            true_answer = messages[1]["content"].strip().upper()
            
            predicted_choice, is_tie = self.predict_pairwise_choice(user_prompt)
            
            if is_tie:
                tie_count += 1
            
            predictions.append(predicted_choice)
            true_answers.append(true_answer)
            user_prompts.append(user_prompt)
            is_tie_list.append(is_tie)
            
            if self.debug:
                print(f"\nExample {i}:")
                print(f"True answer: {true_answer}")
                print(f"Predicted: {predicted_choice}")
                print(f"Tie: {is_tie}")
                is_correct = predicted_choice == true_answer
                print(f"Correct: {is_correct}")
                if is_correct:
                    print("✓ CORRECT PREDICTION")
                else:
                    print("✗ INCORRECT PREDICTION")
        
        print("Calculating metrics...")
        
        # Calculate accuracy excluding ties
        non_tie_correct = 0
        non_tie_total = 0
        
        for pred, true, is_tie in zip(predictions, true_answers, is_tie_list):
            if not is_tie:
                non_tie_total += 1
                if pred == true:
                    non_tie_correct += 1
        
        # Accuracy calculated only on non-tie examples
        accuracy = non_tie_correct / non_tie_total if non_tie_total > 0 else 0.0
        
        return PairwiseEvaluationResults(
            accuracy=accuracy,
            total_examples=len(predictions),
            correct_predictions=non_tie_correct,
            tie_count=tie_count,
            non_tie_examples=non_tie_total,
            predictions=predictions,
            true_answers=true_answers,
            user_prompts=user_prompts,
            is_tie_list=is_tie_list
        )

    def print_results(self, results: PairwiseEvaluationResults):
        """Print evaluation results in a nice format"""
        print("\n" + "="*50)
        print("PAIRWISE EVALUATION RESULTS")
        print("="*50)
        
        print(f"Dataset size: {results.total_examples}")
        print(f"Non-tie examples: {results.non_tie_examples}")
        print(f"Ties: {results.tie_count} ({results.tie_count/results.total_examples*100:.1f}%)")
        print(f"Correct predictions (non-tie only): {results.correct_predictions}")
        print(f"Accuracy (excluding ties): {results.accuracy:.4f} ({results.accuracy*100:.1f}%)")
        print()
        
        # Show choice distribution for non-tie examples only
        pred_a_count = sum(1 for pred, is_tie in zip(results.predictions, results.is_tie_list) if pred == "A" and not is_tie)
        pred_b_count = sum(1 for pred, is_tie in zip(results.predictions, results.is_tie_list) if pred == "B" and not is_tie)
        true_a_count = sum(1 for true, is_tie in zip(results.true_answers, results.is_tie_list) if true == "A" and not is_tie)
        true_b_count = sum(1 for true, is_tie in zip(results.true_answers, results.is_tie_list) if true == "B" and not is_tie)
        
        print("CHOICE DISTRIBUTION (non-tie examples only):")
        print(f"  Predicted A: {pred_a_count} ({pred_a_count/results.non_tie_examples*100:.1f}%)")
        print(f"  Predicted B: {pred_b_count} ({pred_b_count/results.non_tie_examples*100:.1f}%)")
        print(f"  True A: {true_a_count} ({true_a_count/results.non_tie_examples*100:.1f}%)")
        print(f"  True B: {true_b_count} ({true_b_count/results.non_tie_examples*100:.1f}%)")

    def save_detailed_results(self, results: PairwiseEvaluationResults, output_dir: str):
        """Save detailed results including predictions and visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save predictions to CSV including tie information
        df = pd.DataFrame({
            'true_answer': results.true_answers,
            'predicted_answer': results.predictions,
            'is_tie': results.is_tie_list,
            'correct': [pred == true and not is_tie for pred, true, is_tie in zip(results.predictions, results.true_answers, results.is_tie_list)],
            'user_prompt': results.user_prompts
        })
        df.to_csv(output_path / 'pairwise_predictions.csv', index=False)
        
        # Create confusion matrix (2x2 for A/B, excluding ties)
        plt.figure(figsize=(6, 5))
        
        # Count combinations for non-tie examples only
        aa = sum(1 for pred, true, is_tie in zip(results.predictions, results.true_answers, results.is_tie_list) 
                 if pred == "A" and true == "A" and not is_tie)
        ab = sum(1 for pred, true, is_tie in zip(results.predictions, results.true_answers, results.is_tie_list) 
                 if pred == "A" and true == "B" and not is_tie)
        ba = sum(1 for pred, true, is_tie in zip(results.predictions, results.true_answers, results.is_tie_list) 
                 if pred == "B" and true == "A" and not is_tie)
        bb = sum(1 for pred, true, is_tie in zip(results.predictions, results.true_answers, results.is_tie_list) 
                 if pred == "B" and true == "B" and not is_tie)
        
        cm = np.array([[aa, ab], [ba, bb]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=["A", "B"], yticklabels=["A", "B"])
        plt.title('Confusion Matrix (Ties Excluded)')
        plt.ylabel('True Answer')
        plt.xlabel('Predicted Answer')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create accuracy visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Choice distribution (non-tie examples only)
        choices = ["A", "B"]
        pred_counts = [sum(1 for pred, is_tie in zip(results.predictions, results.is_tie_list) if pred == choice and not is_tie) for choice in choices]
        true_counts = [sum(1 for true, is_tie in zip(results.true_answers, results.is_tie_list) if true == choice and not is_tie) for choice in choices]
        
        x = np.arange(len(choices))
        width = 0.35
        
        ax1.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        ax1.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        ax1.set_xlabel('Choice')
        ax1.set_ylabel('Count')
        ax1.set_title('Choice Distribution (Ties Excluded)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(choices)
        ax1.legend()
        
        # Accuracy bar
        ax2.bar(['Overall Accuracy'], [results.accuracy], color='green', alpha=0.7)
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Accuracy (Ties Excluded): {results.accuracy:.1%}')
        ax2.set_ylim(0, 1)
        
        # Add accuracy value on top of bar
        ax2.text(0, results.accuracy + 0.02, f'{results.accuracy:.1%}', 
                ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'accuracy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Danish pairwise fluency judgment model using log probabilities only")
    parser.add_argument("--model_path", required=True, help="Path to the HuggingFace model")
    parser.add_argument("--dataset_path", required=True, help="Path to the evaluation JSONL dataset")
    parser.add_argument("--output_dir", default="./pairwise_evaluation_results", help="Directory to save results")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device for inference")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = DanishPairwiseEvaluator(args.model_path, args.device, debug=args.debug)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(args.dataset_path)
    
    # Print results
    evaluator.print_results(results)
    
    # Save detailed results
    evaluator.save_detailed_results(results, args.output_dir)


if __name__ == "__main__":
    main()