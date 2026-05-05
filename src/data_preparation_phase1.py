from __future__ import annotations

import argparse
import csv
import gzip
import pandas as pd
import nltk
from pathlib import Path
import random
import re

# Ensure nltk punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def build_parser():
    p = argparse.ArgumentParser(description='Phase 1: Data Preparation')
    p.add_argument('--mimic-dir', type=Path, default=Path('../mimiciii'), help='Path to MIMIC-III directory')
    p.add_argument('--gold-labels', type=Path, default=Path('train_data-text_and_labels.csv'), help='Path to 20 gold-labeled sentences')
    p.add_argument('--output', type=Path, default=Path('data/phase1_weakly_labeled.csv'), help='Output path for prepared data')
    p.add_argument('--sample-size', type=int, default=1000, help='Number of weakly labeled sentences to sample')
    p.add_argument('--negative-sample-size', type=int, default=1000, help='Number of heuristic negative sentences to sample')
    return p

def is_boilerplate(text: str) -> bool:
    """Heuristic to identify non-useful/boilerplate sentences."""
    text = text.strip().lower()
    if not text:
        return True
    # Boilerplate examples
    boilerplate_patterns = [
        r'^patient was admitted on',
        r'^date of birth:',
        r'^sex:',
        r'^service:',
        r'^dictated by:',
        r'^completed by:',
        r'^attending:',
        r'^disposition:',
        r'^followup:',
        r'^social history:',
        r'^past medical history:',
        r'^allergies:',
        r'^medications on admission:',
        r'^discharge medications:',
        r'^chief complaint:',
        r'^history of present illness:',
        r'^physical examination:',
        r'^brief hospital course:',
        r'^discharge condition:',
        r'^discharge diagnosis:', # Sometimes just the header
        r'^medications:',
        r'^\d+\.', # Just a number
        r'^-+',    # Just dashes
        r'^=+',    # Just equals
        r'^_+$',   # Just underscores
        r'^\[\*\*.*\*\*\]$', # MIMIC de-identification tags
        r'^\d{4}-\d{2}-\d{2}$', # Dates
        r'^page \d+$',
    ]
    for pattern in boilerplate_patterns:
        if re.search(pattern, text):
            return True
    
    # Too short or too long
    words = text.split()
    if len(words) < 3 or len(words) > 128:
        return True
        
    return False

def main(argv=None):
    args = build_parser().parse_args(argv)
    
    # Create output directory if it doesn't exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    mimic_dir = args.mimic_dir
    note_events_file = mimic_dir / 'NOTEEVENTS.csv.gz'
    diagnoses_file = mimic_dir / 'DIAGNOSES_ICD.csv.gz'
    procedures_file = mimic_dir / 'PROCEDURES_ICD.csv.gz'
    
    if not note_events_file.exists():
        print(f"Error: {note_events_file} not found.")
        return 1
    
    print("Loading diagnoses and procedures to identify weakly labeled admissions...")
    # Get HADM_IDs that have diagnoses or procedures
    weakly_positive_hadms = set()
    if diagnoses_file.exists():
        diagnoses_df = pd.read_csv(diagnoses_file, compression='gzip', usecols=['HADM_ID'])
        weakly_positive_hadms.update(diagnoses_df['HADM_ID'].dropna().unique())
        print(f"Added HADM_IDs from diagnoses. Total: {len(weakly_positive_hadms)}")
    
    if procedures_file.exists():
        procedures_df = pd.read_csv(procedures_file, compression='gzip', usecols=['HADM_ID'])
        weakly_positive_hadms.update(procedures_df['HADM_ID'].dropna().unique())
        print(f"Added HADM_IDs from procedures. Total: {len(weakly_positive_hadms)}")

    print(f"Found {len(weakly_positive_hadms)} admissions with ICD codes.")

    print("Processing NOTEEVENTS...")
    useful_categories = {'Discharge summary', 'Radiology'}
    
    positive_candidates = []
    negative_candidates = []
    
    # Process in chunks to save memory
    chunksize = 50000
    try:
        for chunk in pd.read_csv(note_events_file, compression='gzip', 
                               chunksize=chunksize, 
                               usecols=['ROW_ID', 'HADM_ID', 'CATEGORY', 'TEXT']):
            
            # Filter by category
            filtered_chunk = chunk[chunk['CATEGORY'].isin(useful_categories)]
            
            for _, row in filtered_chunk.iterrows():
                text = row['TEXT']
                if not isinstance(text, str) or not text.strip():
                    continue
                
                hadm_id = row['HADM_ID']
                is_weakly_pos_adm = hadm_id in weakly_positive_hadms
                
                sentences = nltk.sent_tokenize(text)
                
                for sent in sentences:
                    sent = sent.strip().replace('\n', ' ')
                    if is_boilerplate(sent):
                        negative_candidates.append(sent)
                    elif is_weakly_pos_adm:
                        positive_candidates.append(sent)
                    
            # Stop early if we have enough candidates to sample from
            if len(positive_candidates) > args.sample_size * 10 and len(negative_candidates) > args.negative_sample_size * 10:
                break
                
    except Exception as e:
        print(f"Warning during chunk processing: {e}")

    print(f"Collected {len(positive_candidates)} positive and {len(negative_candidates)} negative candidates.")

    # Sample
    final_rows = []
    
    # Sample positives
    sampled_positives = random.sample(positive_candidates, min(len(positive_candidates), args.sample_size))
    for i, text in enumerate(sampled_positives):
        final_rows.append({'row_id': f'weak_pos_{i}', 'text': text, 'label': 1})
        
    # Sample negatives (heuristics)
    sampled_negatives = random.sample(negative_candidates, min(len(negative_candidates), args.negative_sample_size))
    for i, text in enumerate(sampled_negatives):
        final_rows.append({'row_id': f'weak_neg_{i}', 'text': text, 'label': 0})

    # Combine with gold labels
    if args.gold_labels.exists():
        print(f"Adding gold labels from {args.gold_labels}...")
        gold_df = pd.read_csv(args.gold_labels)
        for _, row in gold_df.iterrows():
            final_rows.append({
                'row_id': f"gold_{row.get('row_id', 'unknown')}",
                'text': row['text'],
                'label': int(row['label'])
            })
    else:
        print(f"Warning: Gold labels file {args.gold_labels} not found. Skipping gold labels.")

    # Save to CSV
    output_df = pd.DataFrame(final_rows)
    output_df.to_csv(args.output, index=False)
    print(f"Successfully saved {len(output_df)} sentences to {args.output}")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
