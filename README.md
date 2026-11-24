# Biomedical NLP Challenge — medRxiv 2026 Abstracts

## 📌 Overview

This repository contains end-to-end Natural Language Processing pipelines implemented in **Python** for biomedical literature analysis on **medRxiv 2026** preprints. It demonstrates a complete workflow comparing classical statistical machine learning baselines against modern Transformer-based deep learning architectures across three core NLP tasks.

---

## 📁 Project Structure

### 1. Discipline Prediction (`01_discipline_prediction/`)
* **Objective:** Predict the medical discipline (`category_group`, 16 classes) of raw medRxiv preprints, evaluated using **Macro F1-score** and **Accuracy**.
* **Machine Learning (`01_machine_learning/`):** Targeted HTML cleaning, duplicate preprint removal, hybrid word/character `TF-IDF` vectorization, and class-weighted linear classifiers (`Projet1_défi2_ML.ipynb` $\rightarrow$ `Projet1_Script1_ML.csv`).
* **Deep Learning (`02_deep_learning/`):** Fine-tuning of `BiomedBERT` (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`) with inverse class-frequency weighted `CrossEntropyLoss` (`Projet1_défi2_DL.ipynb` $\rightarrow$ `Projet1_Script2_DL.csv`).
* **Context:** *Academic Data Challenge / Master 2 Health Data Intelligence*

### 2. AI-Written Abstract Detection (`02_ai_abstract_detection/`)
* **Objective:** Classify whether an abstract is human-written (`0`) or LLM-generated (`1`) with out-of-distribution generalization across unseen models, evaluated using **F1-score** and **Precision**.
* **Machine Learning (`01_machine_learning/`):** Custom stylometric features (sentence-length variability/*burstiness*, punctuation patterns, numerical density, lexical diversity) combined with word/character `TF-IDF` and threshold calibration (`Défi2_Projet2_ML.ipynb` $\rightarrow$ `Projet2_Script1_ML.csv`).
* **Deep Learning (`02_deep_learning/`):** Fine-tuning of `BiomedBERT` with class-weighted cross-entropy loss and validation-based probability threshold tuning (`Défi2_Projet2_DL.ipynb` $\rightarrow$ `Projet2_Script2_DL.csv`).
* **Context:** *Academic Data Challenge / Master 2 Health Data Intelligence*

### 3. Abstract Recommendation (`03_abstract_recommendation/`)
* **Objective:** Retrieve and rank the **Top-10** most relevant biomedical abstracts for clinical and research queries, evaluated using **nDCG@10** and **MRR**.
* **Machine Learning (`01_machine_learning/`):** Weighted score fusion combining `TF-IDF`, `Okapi BM25`, `Rocchio` pseudo-relevance feedback, `LSA` (`TruncatedSVD`), and character n-grams (`Script1_ML_Projet3.ipynb` $\rightarrow$ `Défi2_Projet3_ML.csv`).
* **Deep Learning (`02_deep_learning/`):** Two-stage *Retrieve & Re-rank* architecture combining hybrid dense/lexical search (`all-MiniLM-L6-v2` + `BM25`) with neural re-ranking of the top-35 candidates via a `Cross-Encoder` (`Défi2_Script2_DL_Projet3.ipynb` $\rightarrow$ `Défi2_DL_Projet3.csv`).
* **Context:** *Academic Data Challenge / Master 2 Health Data Intelligence*

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`, `scipy`
* **Classical Machine Learning & IR:** `scikit-learn`
* **Deep Learning & Transformers:** `pytorch`, `transformers`, `sentence-transformers`

---

*Author: Redouane Kenniche | Master 2 Health Data Intelligence Student*
