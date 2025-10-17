#!/usr/bin/env python3
"""
Evaluation script for Danish fluency prediction models.
Evaluates a HuggingFace model on fluency prediction task with 1-5 scale using token probabilities only.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


@dataclass
class EvaluationResults:
    """Container for evaluation metrics"""
    mae: float
    rmse: float
    mse: float
    exact_match: float
    within_1: float
    within_2: float
    pearson_r: float
    spearman_r: float
    predictions: List[int]
    true_scores: List[int]
    texts: List[str]


class DanishFluencyEvaluator:
    def __init__(self, model_path: str, device: str = "auto", debug: bool = False):
        """
        Initialize the evaluator with a HuggingFace model.
        
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
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        if self.device.type != "cuda":
            self.model.to(self.device)
        
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
        Load JSONL dataset with conversational format.
        
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

    def extract_text_and_score(self, example: Dict) -> Tuple[str, int]:
        """
        Extract Danish text and fluency score from conversation format.
        
        Args:
            example: Single example in conversational format
            
        Returns:
            Tuple of (danish_text, fluency_score)
        """
        messages = example["messages"]
        user_content = messages[0]["content"]
        true_score = int(messages[1]["content"])
        
        # Extract the Danish sentence from the user prompt
        # Look for text between quotes
        start_quote = user_content.find('"')
        end_quote = user_content.rfind('"')
        
        if start_quote != -1 and end_quote != -1 and start_quote != end_quote:
            danish_text = user_content[start_quote+1:end_quote]
        else:
            # Fallback: assume the text after the colon is the Danish sentence
            colon_idx = user_content.rfind(':')
            if colon_idx != -1:
                danish_text = user_content[colon_idx+1:].strip().strip('"')
            else:
                danish_text = user_content
        
        return danish_text, true_score

    def predict_fluency(self, danish_text: str) -> int:
        """
        Predict fluency score for a Danish text using only token probabilities.
        
        Args:
            danish_text: The Danish sentence to evaluate
            
        Returns:
            Predicted fluency score (1-5)
        """
        return self._predict_via_token_probabilities(danish_text)

    def _predict_via_token_probabilities(self, danish_text: str) -> int:
        """
        Use token probabilities to predict fluency score.
        This computes the likelihood of each score (1-5) given the context.
        """
        try:
            base_prompt = f'Rate the fluency of this Danish sentence on a scale of 1-5 where 1 means either not Danish or very dysfluent and 5 means indistinguishable from text written by a native speaker: "{danish_text}"\n\nDo not say anything else. Only output an integer (1-5).'
            
            # Tokenize the base prompt
            base_inputs = self.tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=450)
            base_inputs = {k: v.to(self.device) for k, v in base_inputs.items()}
            
            if self.debug:
                print(f"Probability prompt: {base_prompt[-100:]}")  # Last 100 chars
            
            # Get token IDs for each score - try multiple variations
            score_tokens = {}
            for score in range(1, 6):
                # Try different token representations
                candidates = [
                    str(score),           # "1"
                    f" {score}",         # " 1" 
                    f"\n{score}",        # "\n1"
                    f"_{score}",         # "_1" (subword)
                ]
                
                for candidate in candidates:
                    token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
                    if len(token_ids) == 1:  # Single token preferred
                        score_tokens[score] = token_ids[0]
                        if self.debug and score == 1:  # Debug first score only
                            print(f"Found token for '{candidate}': {token_ids[0]}")
                        break
                
                # Ultimate fallback
                if score not in score_tokens:
                    token_ids = self.tokenizer.encode(str(score), add_special_tokens=False)
                    score_tokens[score] = token_ids[0]
                    if self.debug and score == 1:
                        print(f"Fallback token for {score}: {token_ids[0]} (from {token_ids})")
            
            # Compute logits for next token
            with torch.no_grad():
                outputs = self.model(**base_inputs)
                next_token_logits = outputs.logits[0, -1, :]  # Last token's logits
            
            # Get probabilities for each score token
            score_probs = {}
            for score, token_id in score_tokens.items():
                score_probs[score] = torch.softmax(next_token_logits, dim=-1)[token_id].item()
            
            if self.debug:
                print(f"Score probabilities: {score_probs}")
                print(f"Token IDs used: {score_tokens}")
            
            # Return the score with highest probability
            predicted_score = max(score_probs.items(), key=lambda x: x[1])[0]
            max_prob = score_probs[predicted_score]
            
            if self.debug:
                print(f"Selected score {predicted_score} with probability {max_prob:.4f}")
            
            return predicted_score
            
        except Exception as e:
            if self.debug:
                print(f"Probability-based prediction failed: {e}")
                import traceback
                traceback.print_exc()
            return 3  # Ultimate fallback

    def evaluate_dataset(self, dataset_path: str) -> EvaluationResults:
        """
        Evaluate the model on the entire dataset.
        
        Args:
            dataset_path: Path to the evaluation dataset
            
        Returns:
            EvaluationResults object with all metrics
        """
        examples = self.load_dataset(dataset_path)
        
        predictions = []
        true_scores = []
        texts = []
        
        print("Running predictions...")
        for i, example in enumerate(examples):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(examples)}")
            
            danish_text, true_score = self.extract_text_and_score(example)
            predicted_score = self.predict_fluency(danish_text)
            
            predictions.append(predicted_score)
            true_scores.append(true_score)
            texts.append(danish_text)
        
        print("Calculating metrics...")
        
        # Convert to numpy arrays
        pred_array = np.array(predictions)
        true_array = np.array(true_scores)
        
        # Calculate metrics
        mae = mean_absolute_error(true_array, pred_array)
        mse = mean_squared_error(true_array, pred_array)
        rmse = np.sqrt(mse)
        
        exact_match = np.mean(pred_array == true_array)
        within_1 = np.mean(np.abs(pred_array - true_array) <= 1)
        within_2 = np.mean(np.abs(pred_array - true_array) <= 2)
        
        pearson_r, _ = pearsonr(true_array, pred_array)
        spearman_r, _ = spearmanr(true_array, pred_array)
        
        return EvaluationResults(
            mae=mae,
            rmse=rmse,
            mse=mse,
            exact_match=exact_match,
            within_1=within_1,
            within_2=within_2,
            pearson_r=pearson_r,
            spearman_r=spearman_r,
            predictions=predictions,
            true_scores=true_scores,
            texts=texts
        )

    def print_results(self, results: EvaluationResults):
        """Print evaluation results in a nice format"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Dataset size: {len(results.predictions)}")
        print(f"Score range: {min(results.true_scores)}-{max(results.true_scores)}")
        print()
        
        print("REGRESSION METRICS:")
        print(f"  Mean Absolute Error (MAE):     {results.mae:.4f}")
        print(f"  Root Mean Square Error (RMSE): {results.rmse:.4f}")
        print(f"  Mean Square Error (MSE):       {results.mse:.4f}")
        print()
        
        print("ACCURACY METRICS:")
        print(f"  Exact Match:          {results.exact_match:.4f} ({results.exact_match*100:.1f}%)")
        print(f"  Within ±1:            {results.within_1:.4f} ({results.within_1*100:.1f}%)")
        print(f"  Within ±2:            {results.within_2:.4f} ({results.within_2*100:.1f}%)")
        print()
        
        print("CORRELATION METRICS:")
        print(f"  Pearson correlation:  {results.pearson_r:.4f}")
        print(f"  Spearman correlation: {results.spearman_r:.4f}")

    def save_detailed_results(self, results: EvaluationResults, output_dir: str):
        """Save detailed results including predictions and visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save predictions to CSV
        df = pd.DataFrame({
            'text': results.texts,
            'true_score': results.true_scores,
            'predicted_score': results.predictions,
            'error': [abs(t - p) for t, p in zip(results.true_scores, results.predictions)]
        })
        df.to_csv(output_path / 'predictions.csv', index=False)
        
        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(results.true_scores, results.predictions, labels=[1, 2, 3, 4, 5])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
        plt.title('Confusion Matrix')
        plt.ylabel('True Score')
        plt.xlabel('Predicted Score')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save score distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True vs Predicted distribution
        ax1.hist(results.true_scores, bins=np.arange(0.5, 6.5, 1), alpha=0.7, label='True', color='blue')
        ax1.hist(results.predictions, bins=np.arange(0.5, 6.5, 1), alpha=0.7, label='Predicted', color='red')
        ax1.set_xlabel('Fluency Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Score Distribution')
        ax1.legend()
        ax1.set_xticks([1, 2, 3, 4, 5])
        
        # Scatter plot
        ax2.scatter(results.true_scores, results.predictions, alpha=0.6)
        ax2.plot([1, 5], [1, 5], 'r--', alpha=0.8)
        ax2.set_xlabel('True Score')
        ax2.set_ylabel('Predicted Score')
        ax2.set_title('True vs Predicted Scores')
        ax2.set_xlim(0.5, 5.5)
        ax2.set_ylim(0.5, 5.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'score_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Danish fluency prediction model")
    parser.add_argument("--model_path", required=True, help="Path to the HuggingFace model")
    parser.add_argument("--dataset_path", required=True, help="Path to the evaluation JSONL dataset")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Directory to save results")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device for inference")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = DanishFluencyEvaluator(args.model_path, args.device, debug=args.debug)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(args.dataset_path)
    
    # Print results
    evaluator.print_results(results)
    
    # Save detailed results
    evaluator.save_detailed_results(results, args.output_dir)


if __name__ == "__main__":
    main()