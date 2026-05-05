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

# ---------------------------------------------------------------------------
# ICD-codable keyword patterns
# These match the kinds of sentences that contain actual diagnoses, procedures,
# or clinical findings that would be assigned an ICD code.
# ---------------------------------------------------------------------------
_ICD_POSITIVE_PATTERNS = re.compile(
    r'\b(?:'
    # Diagnoses / conditions
    r'pneumonia|sepsis|septic|cellulitis|abscess|embolism|thrombosis|'
    r'hemorrhage|bleeding|anemia|aneurysm|stenosis|obstruction|'
    r'fracture|dislocation|laceration|contusion|'
    r'diabetes|diabetic|hypertension|hypotension|hyponatremia|hyperkalemia|'
    r'hyperlipidemia|hypothyroidism|hyperthyroidism|'
    r'atrial fibrillation|tachycardia|bradycardia|arrhythmia|'
    r'myocardial infarction|heart failure|cardiomyopathy|endocarditis|'
    r'stroke|cerebrovascular|intracranial|subdural|subarachnoid|'
    r'renal failure|kidney disease|cirrhosis|hepatitis|pancreatitis|'
    r'appendicitis|cholecystitis|diverticulitis|colitis|gastritis|'
    r'peritonitis|meningitis|encephalitis|osteomyelitis|'
    r'malignancy|carcinoma|lymphoma|leukemia|tumor|neoplasm|metastasis|'
    r'copd|asthma|respiratory failure|pulmonary edema|pleural effusion|'
    r'pneumothorax|ards|acute respiratory distress|'
    r'urinary tract infection|uti|pyelonephritis|'
    r'deep vein thrombosis|dvt|pulmonary embolism|'
    r'congestive heart failure|chf|cad|coronary artery disease|'
    r'chronic kidney disease|ckd|esrd|end.stage renal|'
    r'cerebral palsy|epilepsy|seizure disorder|'
    r'bipolar|schizophrenia|psychosis|delirium|dementia|alzheimer|'
    r'depression|anxiety disorder|'
    r'obesity|morbid obesity|bmi|'
    r'alcohol abuse|substance abuse|drug abuse|overdose|'
    r'gangrene|necrosis|ischemia|infarct|'
    r'edema|ascites|effusion|'
    # Procedures
    r'intubat|extubat|tracheostomy|ventilat|'
    r'catheter|stent|bypass|graft|'
    r'transfus|dialysis|hemodialysis|'
    r'appendectomy|cholecystectomy|colectomy|gastrectomy|'
    r'arthroplasty|amputation|debridement|'
    r'biopsy|resection|excision|'
    r'angioplasty|angiography|endoscopy|colonoscopy|bronchoscopy|'
    r'laparoscop|thoracotomy|craniotomy|laminectomy|'
    r'pacemaker|defibrillator|implant|'
    # Diagnostic language
    r'diagnosed with|diagnosis of|assessment:|'
    r'impression:|finding[s]? of|consistent with|'
    r'confirmed|revealed|demonstrates|indicating|'
    r'positive for|evidence of|'
    r'started on|treated with|given|administered|prescribed|'
    r'status post|s/p|'
    r'history of|h/o|'
    # Lab / clinical values with diagnostic meaning
    r'elevated|decreased|abnormal|'
    r'mg/dl|meq/l|mmol|troponin|creatinine|bilirubin|lactate|inr|'
    r'white blood cell|wbc|hemoglobin|hematocrit|platelet'
    r')\b',
    re.IGNORECASE,
)

# Patterns that indicate a sentence is NOT ICD-codable
_NEGATIVE_PATTERNS = re.compile(
    r'\b(?:'
    r'the patient was comfortable|patient tolerated|tolerated well|'
    r'will follow up|follow.up appointment|'
    r'please call|return to|if you experience|'
    r'no complaints|doing well|feels well|'
    r'vital signs stable|vitals stable|afebrile|'
    r'diet as tolerated|advance diet|regular diet|'
    r'ambulating|out of bed|physical therapy|'
    r'discharge planning|case manager|social work|'
    r'family meeting|goals of care|code status|'
    r'informed consent|risks and benefits|'
    r'the patient is a \d+.year.old'
    r')\b',
    re.IGNORECASE,
)


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


def is_icd_codable(text: str) -> bool:
    """Check if a sentence likely contains ICD-codable information."""
    return bool(_ICD_POSITIVE_PATTERNS.search(text))


def is_clearly_not_codable(text: str) -> bool:
    """Check if a sentence is clearly NOT ICD-codable clinical text."""
    return bool(_NEGATIVE_PATTERNS.search(text))


def build_parser():
    p = argparse.ArgumentParser(description='Phase 1: Data Preparation')
    p.add_argument('--mimic-dir', type=Path, default=Path('../mimiciii'), help='Path to MIMIC-III directory')
    p.add_argument('--gold-labels', type=Path, default=Path('train_data-text_and_labels.csv'), help='Path to 20 gold-labeled sentences')
    p.add_argument('--output', type=Path, default=Path('data/phase1_weakly_labeled.csv'), help='Output path for prepared data')
    p.add_argument('--sample-size', type=int, default=1000, help='Number of weakly labeled sentences to sample')
    p.add_argument('--negative-sample-size', type=int, default=1000, help='Number of heuristic negative sentences to sample')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    
    # Create output directory if it doesn't exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    mimic_dir = args.mimic_dir
    note_events_file = mimic_dir / 'NOTEEVENTS.csv.gz'
    
    if not note_events_file.exists():
        print(f"Error: {note_events_file} not found.")
        return 1
    
    print("Processing NOTEEVENTS with content-based labeling...")
    useful_categories = {'Discharge summary', 'Radiology'}
    
    positive_candidates = []       # Sentences with ICD-codable content
    negative_clinical_candidates = []  # Clinical sentences WITHOUT ICD content
    boilerplate_candidates = []    # Boilerplate/header sentences
    
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
                
                sentences = nltk.sent_tokenize(text)
                
                for sent in sentences:
                    sent = sent.strip().replace('\n', ' ')
                    
                    if is_boilerplate(sent):
                        boilerplate_candidates.append(sent)
                    elif is_icd_codable(sent):
                        # Sentence mentions a real diagnosis/procedure/finding
                        positive_candidates.append(sent)
                    elif is_clearly_not_codable(sent):
                        # Sentence is clinical but clearly not a diagnosis
                        negative_clinical_candidates.append(sent)
                    else:
                        # Ambiguous clinical sentence -> negative
                        # (conservative: if we can't find ICD keywords, it's not codable)
                        negative_clinical_candidates.append(sent)
                    
            # Stop early if we have enough candidates to sample from
            if len(positive_candidates) > args.sample_size * 5 and \
               len(negative_clinical_candidates) > args.negative_sample_size * 5:
                break
                
    except Exception as e:
        print(f"Warning during chunk processing: {e}")

    print(f"Collected {len(positive_candidates)} positive (ICD-keyword), "
          f"{len(negative_clinical_candidates)} clinical negative, "
          f"and {len(boilerplate_candidates)} boilerplate candidates.")

    # Sample
    final_rows = []
    
    # Sample positives (sentences that contain ICD-related keywords)
    sampled_positives = random.sample(positive_candidates, min(len(positive_candidates), args.sample_size))
    for i, text in enumerate(sampled_positives):
        final_rows.append({'row_id': f'weak_pos_{i}', 'text': text, 'label': 1})
        
    # Sample negatives: 60% clinical negatives, 20% boilerplate, 20% ambiguous
    n_clinical = int(args.negative_sample_size * 0.6)
    n_boilerplate = int(args.negative_sample_size * 0.2)
    n_ambiguous = args.negative_sample_size - n_clinical - n_boilerplate
    
    sampled_clin_neg = random.sample(negative_clinical_candidates, min(len(negative_clinical_candidates), n_clinical + n_ambiguous))
    for i, text in enumerate(sampled_clin_neg):
        final_rows.append({'row_id': f'clin_neg_{i}', 'text': text, 'label': 0})
        
    sampled_boilerplate = random.sample(boilerplate_candidates, min(len(boilerplate_candidates), n_boilerplate))
    for i, text in enumerate(sampled_boilerplate):
        final_rows.append({'row_id': f'boilerplate_{i}', 'text': text, 'label': 0})

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
    
    # Print label distribution
    pos_count = sum(1 for r in final_rows if r['label'] == 1)
    neg_count = sum(1 for r in final_rows if r['label'] == 0)
    print(f"Successfully saved {len(output_df)} sentences to {args.output}")
    print(f"Label distribution: {pos_count} positive ({pos_count/len(output_df)*100:.1f}%), "
          f"{neg_count} negative ({neg_count/len(output_df)*100:.1f}%)")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
