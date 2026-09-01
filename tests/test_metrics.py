import numpy as np
import pytest

from imdb_sentiment import SentimentAnalyzer, compute_classification_metrics


def test_metrics_for_perfect_predictions():
    metrics = compute_classification_metrics(np.array([0, 1, 1]), np.array([0, 1, 1]))
    assert metrics == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_empty_prediction_text_is_rejected_before_model_loading():
    analyzer = SentimentAnalyzer(model_path="unused")
    with pytest.raises(ValueError, match="must not be empty"):
        analyzer.predict_sentiment("   ")

