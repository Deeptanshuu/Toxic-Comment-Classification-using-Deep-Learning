import json
import os
from pathlib import Path

def extract_thresholds(eval_results_path: str, output_path: str = None) -> dict:
    """
    Extract classification thresholds from evaluation results JSON file.
    
    Args:
        eval_results_path (str): Path to the evaluation results JSON file
        output_path (str, optional): Path to save the extracted thresholds. 
                                   If None, will save in the same directory as eval results
    
    Returns:
        dict: Dictionary containing the extracted thresholds per language
    """
    # Read evaluation results
    with open(eval_results_path, 'r') as f:
        results = json.load(f)
    
    # Extract thresholds
    thresholds = results.get('thresholds', {})
    
    # Save to file if output path provided
    if output_path is None:
        # Create thresholds file in same directory as eval results
        eval_dir = os.path.dirname(eval_results_path)
        output_path = os.path.join(eval_dir, 'thresholds.json')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save with nice formatting
    with open(output_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    return thresholds

if __name__ == '__main__':
    # Example usage
    eval_results_path = 'evaluation_results/eval_20250208_161149/evaluation_results.json'
    thresholds = extract_thresholds(eval_results_path)
    print("Thresholds extracted and saved successfully!") 