I would build it as a single general semi-supervised text classification pipeline, not a separate per-test rules setup. The project asks for one offline model that accepts any sentence up to 128 words and predicts whether it contains ICD-codable information, and it explicitly allows using additional unlabeled MIMIC-III/public data for training.  ￼

Best project design

1. Frame the problem correctly

This is a binary sentence-level classification task:

Input: one clinical sentence
Output: 1 = ICD-codable medical information, 0 = not ICD-codable

The hard part is that you only get 20 labeled examples, so a fully supervised model trained only on those labels will be weak. The lecture slides directly support this issue: labeled data is expensive, especially medical data, so self-supervised and semi-supervised learning are useful when labels are limited.  ￼

So the reportable idea should be:

“We use a semi-supervised pipeline that learns from large unlabeled MIMIC-III clinical text, then uses the 20 labeled examples only to calibrate the final ICD-codable classifier.”

⸻

2. Build three models, but submit one final ensemble

I would build these models:

Model A: Strong baseline — TF-IDF + Logistic Regression / Linear SVM

Use:

* word n-grams: 1–3
* character n-grams: 3–5
* TF-IDF features
* Linear SVM or Logistic Regression
* class weighting
* probability calibration

Why this matters: with only 20 labels, simple linear models often beat neural networks because they do not overfit as badly.

Features that should help:

* diagnosis words: pneumonia, sepsis, failure, fracture, hypertension
* treatment/procedure words: started, given, intubated, transfused
* lab/vital patterns: numbers + units, mg/dL, WBC, Na, K
* negation/context patterns: no evidence of, denies, history of

This gives you a clean, explainable baseline for the report.

⸻

Model B: Semi-supervised pseudo-labeling model

Use the unlabeled MIMIC-III sentences to generate extra training data.

Pipeline:

1. Extract many sentences from MIMIC-III notes.
2. Clean and segment them into sentences.
3. Train the baseline on the 20 labeled examples.
4. Predict probabilities on unlabeled sentences.
5. Keep only high-confidence pseudo-labels:
    * positive if p >= 0.90
    * negative if p <= 0.10
6. Retrain the classifier on:
    * original 20 gold labels
    * pseudo-labeled examples, but with lower weight
7. Repeat 1–3 rounds.

This matches the lecture idea that semi-supervised learning uses unlabeled data when labels are limited, and the slides specifically mention pseudo-labeling as a semi-supervised method.  ￼

This is probably the most “project-appropriate” approach because it uses the unlabeled MIMIC data allowed by the instructions without manually labeling test data.

⸻

Model C: Offline clinical language model embeddings + classifier

Use an offline pre-trained model only if you already have it downloaded locally and it does not require internet/API access during training or prediction.

Good options:

* ClinicalBERT
* BioBERT
* PubMedBERT
* SapBERT
* MiniLM/Sentence-BERT variant, if already local

I would not fully fine-tune the transformer on 20 labels. Instead:

1. Use the transformer as a frozen feature extractor.
2. Convert each sentence into an embedding.
3. Train Logistic Regression / SVM on embeddings.
4. Optionally combine embeddings with TF-IDF features.

This fits the lecture slide logic that transformers became dominant for NLP because pretraining on large corpora lets them transfer to small labeled tasks.  ￼

⸻

3. Final submitted model

I would submit one final hybrid/ensemble model, not per-test behavior.

Final prediction:

p_final = 0.50 * p_tfidf_logreg
        + 0.30 * p_pseudolabel_model
        + 0.20 * p_embedding_model

Then choose a single global threshold, such as:

label = 1 if p_final >= threshold else 0

Tune the threshold only using:

* leave-one-out validation on the 20 labeled examples
* pseudo-labeled validation sanity checks
* maybe a small validation split from silver/pseudo-labeled data

Do not tune a separate threshold for each test file. That is exactly the part that becomes hard to justify.

⸻

4. Data preparation

I would document the preprocessing carefully because the instructions explicitly require explaining data preparation and segmentation.  ￼

Preprocessing:

lowercase
normalize whitespace
replace numbers with <NUM> sometimes
preserve medical abbreviations
preserve negation words
truncate to 128 words
sentence-segment MIMIC notes
remove obvious headers if they are not sentences
deduplicate near-identical sentences

Do not over-clean clinical text. Abbreviations, numbers, and weird formatting are useful.

⸻

5. Evaluation plan

Because the labeled set is tiny, I would report several validation strategies:

Experiment	Purpose
20-label leave-one-out CV	Estimate performance on scarce labels
Baseline vs pseudo-labeling	Show unlabeled data helped
TF-IDF vs embeddings	Compare classical vs pretrained representations
Threshold sweep	Pick one global decision threshold
Error analysis	Show common false positives/negatives

Main metrics:

* accuracy
* precision
* recall
* F1
* confusion matrix

For this task, I would care most about F1 and recall, because missing ICD-codable sentences is bad.

⸻

6. What I would write as the core method

The clean “Methods” explanation:

We first segment MIMIC-III clinical notes into candidate sentences and normalize them while preserving medically meaningful tokens. We train a TF-IDF linear classifier using the small labeled set, then apply it to a large unlabeled sentence pool to generate high-confidence pseudo-labels. These pseudo-labeled examples are used as weak supervision to retrain the classifier. In parallel, we extract frozen sentence embeddings from an offline pretrained clinical language model and train a calibrated logistic regression classifier. The final prediction is produced by a fixed weighted ensemble with a single global threshold.

That sounds reportable, defensible, and tied to lecture content.

⸻

7. What I would avoid

Avoid:

* per-test thresholds
* manually labeling test examples
* uploading MIMIC/test data to ChatGPT or any generative AI tool
* training a big neural network from scratch on 20 examples
* rules-only model as the final system
* using internet APIs during prediction

The instruction explicitly says the model must work offline and no internet/external APIs are allowed. It also warns not to upload MIMIC data to generative AI resources.  ￼

⸻

Final recommendation

Build this:

Sentence extraction from MIMIC
        ↓
Preprocessing + deduplication
        ↓
TF-IDF baseline classifier
        ↓
High-confidence pseudo-labeling on unlabeled MIMIC sentences
        ↓
Retrained semi-supervised classifier
        ↓
Optional frozen ClinicalBERT/BioBERT embedding classifier
        ↓
Single calibrated ensemble
        ↓
One global threshold
        ↓
test##-pred.csv files

This is the strongest project direction because it is:

* allowed by the instructions
* based on lecture concepts
* explainable in a report
* stronger than using 20 labels alone
* generalizable instead of test-specific
* offline-compatible
* easy to defend in the presentation
