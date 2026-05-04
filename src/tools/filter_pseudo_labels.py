"""Filter pseudo-labels based on confidence thresholds."""

import argparse
import pandas as pd
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Filter pseudo-labels by confidence")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output filtered CSV path")
    parser.add_argument("--threshold_high", type=float, default=0.9, help="Keep label=1 if prob > threshold_high")
    parser.add_argument("--threshold_low", type=float, default=0.1, help="Keep label=0 if prob < threshold_low")
    parser.add_argument("--manifest", help="Optional manifest.json to update")
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    
    # Determine probability of class 1
    if "probability" in df.columns:
        probs = df["probability"]
    elif "confidence" in df.columns and "label" in df.columns:
        # if label=1, prob = confidence
        # if label=0, prob = 1 - confidence
        probs = df.apply(lambda r: r["confidence"] if r["label"] == 1 else 1.0 - r["confidence"], axis=1)
    else:
        print(f"Error: Neither 'probability' nor ('confidence' and 'label') found in {args.input}")
        return

    print(f"Total samples: {len(df)}")
    
    mask_pos = probs > args.threshold_high
    mask_neg = probs < args.threshold_low
    
    df_pos = df[mask_pos].copy()
    df_pos["label"] = 1
    
    df_neg = df[mask_neg].copy()
    df_neg["label"] = 0
    
    filtered_df = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Ensure standard training columns
    output_columns = ["row_id", "text", "label", "confidence"]
    for col in output_columns:
        if col not in filtered_df.columns:
            if col == "row_id":
                filtered_df[col] = range(len(filtered_df))
            else:
                filtered_df[col] = None
    
    filtered_df = filtered_df[output_columns]
    
    print(f"Kept {len(filtered_df)} samples ({len(df_pos)} positive, {len(df_neg)} negative)")
    print(f"Discarded {len(df) - len(filtered_df)} samples")
    
    filtered_df.to_csv(args.output, index=False)
    print(f"Saved filtered pseudo-labels to {args.output}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Update silver entry to point to filtered output
            for entry in manifest.get("entries", []):
                if entry.get("split") == "silver_train":
                    # Make path relative to manifest if possible
                    try:
                        rel_path = Path(args.output).relative_to(manifest_path.parent)
                        entry["path"] = str(rel_path)
                    except ValueError:
                        entry["path"] = str(args.output)
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"Updated manifest {args.manifest} to point to {args.output}")

if __name__ == "__main__":
    main()
