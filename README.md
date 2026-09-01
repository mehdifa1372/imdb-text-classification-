# Transformer Sentiment Classification on IMDb

A reproducible applied-NLP project for fine-tuning a Hugging Face transformer on the IMDb movie-review dataset. The repository separates reusable training and inference code from the demonstration notebook and exposes a small command-line interface.

## What this project demonstrates

- Transformer fine-tuning for binary text classification.
- Dynamic token padding and configurable sequence length.
- Leakage-aware train/validation/test evaluation.
- Accuracy, precision, recall, and F1 reporting.
- Reproducible seeds and automatic CPU/GPU mixed-precision selection.
- A reusable Python package, CLI, tests, and continuous integration.

## Repository structure

```text
.
├── src/imdb_sentiment/     # Training, evaluation, and inference package
├── tests/                  # Fast unit tests
├── imdb_sentiment_demo.ipynb
├── pyproject.toml
└── .github/workflows/quality.yml
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Train

```bash
imdb-sentiment train \
  --model-name distilbert-base-uncased \
  --output-dir artifacts \
  --epochs 3 \
  --batch-size 8
```

The training command creates a validation split from the original training data. The official IMDb test split remains reserved for the final evaluation.

## Predict

```bash
imdb-sentiment predict \
  --model-path artifacts/final_model \
  --text "The performances were excellent."
```

## Results

This repository does not claim a score that has not been reproduced in a clean run. After training, copy the generated test metrics into the table below and record the commit, model checkpoint, seed, hardware, and runtime.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Reproducible run pending | — | — | — | — |

## Methodology notes

- The original notebook evaluated repeatedly on the test split, which can turn the test set into an implicit tuning set. The package now creates a validation split for model selection and uses the test split once at the end.
- Mixed precision is enabled only when CUDA is available.
- Model checkpoints and downloaded datasets are ignored by Git because they are generated artifacts.
- IMDb reviews may contain offensive language. Treat examples and model output accordingly.

## Limitations

IMDb is a narrow English-language benchmark. Performance does not establish robustness to other domains, languages, sarcasm, distribution shift, or adversarial text. This model should not be used for consequential decisions without domain-specific evaluation and human oversight.

## Author

Mehdi Faraz — computer vision, machine learning, data science, and applied AI.

