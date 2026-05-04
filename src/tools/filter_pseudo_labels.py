"""Filter pseudo-labels based on confidence thresholds."""

import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Filter pseudo-labels by confidence")
    parser.add_argument("--input", required=True, help="Input debug CSV path (with 'probability' column)")
    parser.add_argument("--output", required=True, help="Output filtered CSV path")
    parser.add_argument("--threshold_high", type=float, default=0.9, help="Keep label=1 if prob > threshold_high")
    parser.add_argument("--threshold_low", type=float, default=0.1, help="Keep label=0 if prob < threshold_low")
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    
    if "probability" not in df.columns:
        print(f"Error: 'probability' column not found in {args.input}")
        return

    print(f"Total samples: {len(df)}")
    
    # Filter based on confidence
    # if prob > 0.9 -> keep (label = 1)
    # if prob < 0.1 -> keep (label = 0)
    
    mask_pos = df["probability"] > args.threshold_high
    mask_neg = df["probability"] < args.threshold_low
    
    df_pos = df[mask_pos].copy()
    df_pos["label"] = 1
    
    df_neg = df[mask_neg].copy()
    df_neg["label"] = 0
    
    filtered_df = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Drop intermediate columns if we want it to look like a training file
    # Training files usually have row_id, text, label
    output_columns = ["row_id", "text", "label"]
    if "row_id" not in filtered_df.columns:
        filtered_df["row_id"] = range(len(filtered_df))
    
    filtered_df = filtered_df[output_columns]
    
    print(f"Kept {len(filtered_df)} samples ({len(df_pos)} positive, {len(df_neg)} negative)")
    print(f"Discarded {len(df) - len(filtered_df)} samples")
    
    filtered_df.to_csv(args.output, index=False)
    print(f"Saved filtered pseudo-labels to {args.output}")

if __name__ == "__main__":
    main()
