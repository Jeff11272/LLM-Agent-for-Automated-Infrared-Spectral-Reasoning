# LLM Agent for Automated Infrared Spectral Reasoning

> Updated English README with module summaries and notebook entry (2025-09-01).

## Overview

This repository provides an LLM-driven pipeline for automated infrared spectral reasoning—covering preprocessing, feature extraction, retrieval/selection of literature-backed methods, and multi-task inference (classification, regression, anomaly detection).

## How to Run (Notebook Entry)

- **Primary entry:** `main.ipynb`


1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter lab    # or: jupyter notebook
   ```
4. Select the environment kernel, then **Run All** (or follow cell-by-cell instructions in the notebook).

## Command-line (Optional)

If scripts exist under `scripts/` or `src/`, you can also run them directly. Example:

```bash
python scripts/demo.py --config configs/base.yaml
```
_Update the example with the real script and arguments if different._

## Module Overview

Below is an auto-generated summary of Python modules in this repository. The first line of the module docstring or top comments is used where available.

| Module | Purpose |
|---|---|

| `Agent.py` | from  sktime.datasets import load_tecator -------------------- Method selection related (overall voting version) -------------------- Free choice: can pass paper name or index in paper_order (both 0-based and 1-based supported) |
| `Entity_extraction.py` | Defines 1 functions and 0 classes (auto-detected). |
| `Feature_extract.py` | !/usr/bin/env python3 -*- coding: utf-8 -*- ====================== |
| `Generate_single.py` | Unify full-width/strange quotes, remove surrounding quotes and extra punctuation Synonym/alias mapping (keys should be lowercase) Case-insensitive closest match in allowed set (exact match prioritized) |
| `Mul_Generate.py` | -*- coding: utf-8 -*- --------------------------- Helpers: list & split |
| `Preprocess_method.py` | spectral_preprocessing.py Second-order difference matrix D Composite preprocessing functions |
| `Retrieval.py` | If you haven't downloaded the punkt tokenizer yet, uncomment and run the following line once: nltk.download('punkt') Tokenize each paper_name (using nltk.word_tokenize here) |
| `dataset.py` | !/usr/bin/env python3 -*- coding: utf-8 -*- Randomly assign sample count for each class |
| `dataset_config.py` | Defines 1 functions and 0 classes (auto-detected). |
| `other_models.py` | -*- coding: utf-8 -*- ---- Classification metrics / models ---- ---- Regression metrics / models ---- |


## Data

- Provide dataset source, licensing, and preprocessing steps.
- Organize raw/processed data under `data/` with clear train/val/test splits.

## Evaluation & Results

- Report task-specific metrics: Accuracy/F1 (classification), R²/RMSE/MAE (regression), AUC/Precision (anomaly detection).
- Save logs, figures, and tables under `runs/` or `reports/` and reference them here.

## Environment & Dependencies

- Recommended: Python 3.10+
- Install:
```bash
pip install -r requirements.txt
```
- GPU users: ensure the correct CUDA build (e.g., for `torch`) that matches your driver.

## Development Guide

- Code style: `black`/`ruff`; enable `pre-commit`.
- Testing: add `pytest` unit tests for preprocessing and feature extraction functions.
- CI: lint + unit tests on each push.

## License & Citation

- License: MIT / Apache-2.0 (choose and add a LICENSE file).
- Citation: include BibTeX if associated with a paper.
