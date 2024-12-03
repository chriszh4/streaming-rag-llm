import json
from typing import List, Tuple, Dict
import argparse
from pathlib import Path
import numpy as np

def load_json(file_path: str) -> List[str]:
    """Load JSON file containing a list of strings."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_string_metrics(response: str, ground_truth: str, case_sensitive: bool = False) -> Dict[str, float]:
    """
    Compute precision and recall between two strings at word level.
    """
    if not case_sensitive:
        response = response.lower()
        ground_truth = ground_truth.lower()
    
    # Split into words
    response_words = set(response.split())
    ground_truth_words = set(ground_truth.split())
    
    if not ground_truth_words:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not response_words:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Count matches
    matches = len(ground_truth_words.intersection(response_words))
    
    # Calculate metrics
    precision = matches / len(response_words)
    recall = matches / len(ground_truth_words)
    
    # Calculate F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'response_words': len(response_words),
        'ground_truth_words': len(ground_truth_words),
        'matches': matches
    }

def main():
    parser = argparse.ArgumentParser(description='Compute word-level precision and recall between response and ground truth strings')
    parser.add_argument('--response', type=str, required=True, help='Path to response JSON file')
    parser.add_argument('--additional_response', type=str, required=False, help='Path to additional response JSON file')
    parser.add_argument('--ground-truth', type=str, required=True, help='Path to ground truth JSON file')
    parser.add_argument('--case-sensitive', action='store_true', help='Enable case-sensitive matching')
    
    args = parser.parse_args()
    
    # Load the JSON files
    response_data = load_json(args.response)
    ground_truth_data = load_json(args.ground_truth)
    if args.additional_response:
        additional_response_data = load_json(args.additional_response)
        for i in range(len(additional_response_data)):
            response_data[i] = response_data[i] + ' ' + additional_response_data[i]
    
    # Ensure equal lengths
    if len(response_data) != len(ground_truth_data):
        raise ValueError(f"Number of responses ({len(response_data)}) does not match number of ground truth items ({len(ground_truth_data)})")
    
    # Compute metrics for each pair
    all_metrics = []
    total_response_words = 0
    total_ground_truth_words = 0
    total_matches = 0
    
    for i, (response, ground_truth) in enumerate(zip(response_data, ground_truth_data)):
        metrics = compute_string_metrics(response, ground_truth, args.case_sensitive)
        all_metrics.append(metrics)
        
        # Accumulate totals for micro-averaging
        total_response_words += metrics['response_words']
        total_ground_truth_words += metrics['ground_truth_words']
        total_matches += metrics['matches']
        
        # Print individual results if you want to see them
        # print(f"\nPair {i+1}:")
        # print(f"Response: {response}")
        # print(f"Ground truth: {ground_truth}")
        # print(f"Metrics: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    # Calculate micro-average metrics
    micro_precision = total_matches / total_response_words if total_response_words > 0 else 0
    micro_recall = total_matches / total_ground_truth_words if total_ground_truth_words > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Calculate macro-average metrics
    macro_precision = np.mean([m['precision'] for m in all_metrics])
    macro_recall = np.mean([m['recall'] for m in all_metrics])
    macro_f1 = np.mean([m['f1'] for m in all_metrics])
    
    print("\nOverall Results:")
    print("\nMicro-averaged metrics (based on total word counts):")
    print(f"Precision: {micro_precision:.3f}")
    print(f"Recall: {micro_recall:.3f}")
    print(f"F1 Score: {micro_f1:.3f}")
    
    print("\nMacro-averaged metrics (average of individual scores):")
    print(f"Precision: {macro_precision:.3f}")
    print(f"Recall: {macro_recall:.3f}")
    print(f"F1 Score: {macro_f1:.3f}")
    
    print(f"\nTotal pairs processed: {len(all_metrics)}")
    print(f"Total words in responses: {total_response_words}")
    print(f"Total words in ground truth: {total_ground_truth_words}")
    print(f"Total word matches: {total_matches}")

if __name__ == "__main__":
    main()