# Contributing

Use focused branches and descriptive commits. Do not commit downloaded datasets, model weights, tokens, caches, or full training logs.

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

Changes to model behavior should state the hypothesis, data split, seed, metric, hardware, and comparison baseline. Do not tune against the held-out test split.

