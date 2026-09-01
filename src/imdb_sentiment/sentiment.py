"""Training and inference utilities for IMDb transformer classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LOGGER = logging.getLogger(__name__)


def compute_classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return standard binary-classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


class SentimentAnalyzer:
    """Fine-tune and run a transformer sentiment classifier."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        model_path: str | Path | None = None,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.model_path = Path(model_path) if model_path else None
        self.max_length = max_length
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    @staticmethod
    def _trainer_metrics(prediction_output: Any) -> dict[str, float]:
        predictions = np.argmax(prediction_output.predictions, axis=-1)
        return compute_classification_metrics(prediction_output.label_ids, predictions)

    def _tokenize(self, examples: dict[str, list[str]]) -> dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized")
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
        )

    def _prepare_dataset(self, validation_size: float, seed: int) -> DatasetDict:
        raw = load_dataset("imdb")
        development = raw["train"].train_test_split(test_size=validation_size, seed=seed)
        return DatasetDict(
            train=development["train"],
            validation=development["test"],
            test=raw["test"],
        )

    def train(
        self,
        output_dir: str | Path = "artifacts",
        epochs: float = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        validation_size: float = 0.1,
        seed: int = 42,
    ) -> dict[str, float]:
        """Train on IMDb and evaluate once on the held-out test split."""
        if not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")

        set_seed(seed)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Loading IMDb and creating a validation split")
        dataset = self._prepare_dataset(validation_size, seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenized = dataset.map(
            self._tokenize,
            batched=True,
            remove_columns=["text"],
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        arguments = TrainingArguments(
            output_dir=str(output_path / "checkpoints"),
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            fp16=torch.cuda.is_available(),
            seed=seed,
        )
        trainer = Trainer(
            model=self.model,
            args=arguments,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=data_collator,
            compute_metrics=self._trainer_metrics,
        )
        trainer.train()

        LOGGER.info("Running one final evaluation on the held-out IMDb test split")
        test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
        final_path = output_path / "final_model"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        self.model_path = final_path
        return {key: float(value) for key, value in test_metrics.items() if isinstance(value, (int, float))}

    def load_model(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else self.model_path
        if path is None:
            raise ValueError("A model path is required for inference")
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.eval()

    def predict_sentiment(self, text: str) -> str:
        """Return `Positive` or `Negative` for a non-empty review."""
        if not text.strip():
            raise ValueError("text must not be empty")
        if self.model is None or self.tokenizer is None:
            self.load_model()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        model_device = next(self.model.parameters()).device
        inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            prediction = int(self.model(**inputs).logits.argmax(dim=-1).item())
        return "Positive" if prediction == 1 else "Negative"

