# Biomedical NLP Challenge — medRxiv 2026 Abstracts

This repository contains end-to-end NLP pipelines developed for the **medRxiv 2026 Biomedical Literature Challenge** (M.Sc. in Health Data Science — University of Strasbourg).

For each task, a **classical Machine Learning** baseline (`01_machine_learning/`) is compared against a **Transformer-based Deep Learning** architecture (`02_deep_learning/`).

---

## Repository Structure

### `01_discipline_prediction` — Multi-Class Classification (16 Medical Disciplines)
Predicting the medical category (`category_group`) of raw medRxiv preprints, evaluated with **Macro F1-score** (and **Accuracy** as tie-breaker).
* **`01_machine_learning/Projet1_défi2_ML.ipynb`** *(outputs: `Projet1_Script1_ML.csv`)*: Targeted HTML cleaning, duplicate preprint removal, hybrid word/character `TF-IDF` vectorization, and class-weighted linear classifiers (`LinearSVC` / Logistic Regression).
* **`02_deep_learning/Projet1_défi2_DL.ipynb`** *(outputs: `Projet1_Script2_DL.csv`)*: Fine-tuning of `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` with inverse class-frequency weighted `CrossEntropyLoss`.

### `02_ai_abstract_detection` — AI-Generated Text Detection
Binary classification (`0` = Human-written, `1` = LLM-generated) designed to generalize across unseen generative models, evaluated with **F1-score** (and **Precision** as tie-breaker).
* **`01_machine_learning/Défi2_Projet2_ML.ipynb`** *(outputs: `Projet2_Script1_ML.csv`)*: Custom stylometric features (sentence-length variability/*burstiness*, punctuation patterns, numerical precision, lexical diversity) combined with word/character `TF-IDF` and probability threshold tuning.
* **`02_deep_learning/Défi2_Projet2_DL.ipynb`** *(outputs: `Projet2_Script2_DL.csv`)*: Fine-tuning of `BiomedBERT` with weighted cross-entropy loss and validation-based threshold calibration.

### `03_abstract_recommendation` — Biomedical Information Retrieval
Ranking system returning the **Top-10** most relevant abstracts for clinical and research queries, evaluated using **nDCG@10** and **MRR**.
* **`01_machine_learning/Script1_ML_Projet3.ipynb`** *(outputs: `Défi2_Projet3_ML.csv`)*: Weighted score fusion combining `TF-IDF`, `Okapi BM25`, `Rocchio` pseudo-relevance feedback, `LSA` (`TruncatedSVD`), and character n-grams.
* **`02_deep_learning/Défi2_Script2_DL_Projet3.ipynb`** *(outputs: `Défi2_DL_Projet3.csv`)*: Two-stage *Retrieve & Re-rank* pipeline combining hybrid dense/lexical retrieval (`all-MiniLM-L6-v2` + `BM25`) with neural re-ranking of the top-35 candidates via a `Cross-Encoder` (`ms-marco-MiniLM-L-6-v2`).

---

## Tech Stack
* **Language:** Python
* **Classical ML & IR:** `scikit-learn`, `scipy`, `pandas`, `numpy`
* **Deep Learning & NLP:** `PyTorch`, `Hugging Face Transformers`, `Sentence-Transformers`
