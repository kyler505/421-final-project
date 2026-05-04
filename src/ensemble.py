"""Ensemble multiple models by voting or averaging probabilities."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import save_submission_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble multiple model predictions")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of prediction CSVs (submission or debug format)")
    parser.add_argument("--output", required=True, help="Output submission CSV path")
    parser.add_argument("--method", choices=["vote", "average"], default="vote", help="Ensemble method")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for average method")
    return parser.parse_args()

def main():
    args = parse_args()
    
    dfs = []
    for i, path in enumerate(args.inputs):
        df = pd.read_csv(path)
        # Ensure row_id is index for alignment
        if "row_id" in df.columns:
            df = df.set_index("row_id")
        else:
            print(f"Warning: {path} has no row_id column. Assuming order matches.")
        dfs.append(df)
        
    if not dfs:
        return

    # Check if we have probabilities for averaging
    has_probs = all("probability" in df.columns for df in dfs)
    
    if args.method == "average":
        if not has_probs:
            print("Error: All input files must have a 'probability' column for 'average' method.")
            # Fallback to vote?
            print("Falling back to 'vote' method.")
            args.method = "vote"
        else:
            print("Averaging probabilities...")
            all_probs = pd.concat([df["probability"] for df in dfs], axis=1)
            mean_probs = all_probs.mean(axis=1)
            final_preds = (mean_probs >= args.threshold).astype(int)
            
    if args.method == "vote":
        print("Majority voting...")
        # Use 'prediction' column
        all_preds = pd.concat([df["prediction"] for df in dfs], axis=1)
        # Majority vote: mean > 0.5
        final_preds = (all_preds.mean(axis=1) > 0.5).astype(int)

    # Save output
    row_ids = final_preds.index.tolist()
    predictions = final_preds.values.tolist()
    
    save_submission_predictions(row_ids, predictions, args.output)
    print(f"Saved ensemble predictions to {args.output}")
    print(f"Positive predictions: {sum(predictions)}/{len(predictions)}")

if __name__ == "__main__":
    main()
