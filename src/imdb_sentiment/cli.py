"""Command-line interface for IMDb sentiment training and inference."""

from __future__ import annotations

import argparse
import json
import logging

from .sentiment import SentimentAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IMDb transformer sentiment classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Fine-tune a model on IMDb")
    train.add_argument("--model-name", default="distilbert-base-uncased")
    train.add_argument("--output-dir", default="artifacts")
    train.add_argument("--epochs", type=float, default=3)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--validation-size", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)

    predict = subparsers.add_parser("predict", help="Classify one review")
    predict.add_argument("--model-path", required=True)
    predict.add_argument("--text", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "train":
        analyzer = SentimentAnalyzer(model_name=args.model_name)
        metrics = analyzer.train(
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_size=args.validation_size,
            seed=args.seed,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    analyzer = SentimentAnalyzer(model_path=args.model_path)
    print(analyzer.predict_sentiment(args.text))


if __name__ == "__main__":
    main()

