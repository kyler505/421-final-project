"""Tune prediction threshold on validation set."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from src.models.baseline import BaselineModel
from src.models.svm import SVMModel
from src.data import load_train_data, get_texts_labels

def parse_args():
    parser = argparse.ArgumentParser(description="Tune threshold on validation data")
    parser.add_argument("--model", required=True, help="Model path (.pkl)")
    parser.add_argument("--val", required=True, help="Validation CSV path")
    parser.add_argument("--mode", choices=["baseline", "svm"], required=True, help="Model mode")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model
    if args.mode == "baseline":
        model = BaselineModel.load(args.model)
    else:
        model = SVMModel.load(args.model)
        
    # Load val data
    val_df = load_train_data(args.val)
    texts, labels = get_texts_labels(val_df)
    
    # Get probabilities
    probs = model.predict_proba(texts)[:, 1]
    
    best_f1 = -1
    best_thresh = 0.5
    
    thresholds = np.linspace(0.01, 0.99, 99)
    
    print(f"Tuning threshold on {len(texts)} samples...")
    
    results = []
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
        
        results.append({
            "threshold": thresh,
            "f1": f1,
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "accuracy": accuracy_score(labels, preds)
        })
        
    print(f"\nBest Threshold: {best_thresh:.3f}")
    print(f"Best F1: {best_f1:.4f}")
    
    # Find metrics at best threshold
    best_idx = np.argmax([r["f1"] for r in results])
    best_metrics = results[best_idx]
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
