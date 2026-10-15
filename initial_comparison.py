"""Run the initial lexical, semantic, and combined comparison per project.

Rows use sample_id, text, project_id, label_id, and leakage_group_id.
Run separately for each of the four projects.
Example: python initial_comparison.py --train train.jsonl
    --validation validation.jsonl --test test.jsonl
    --embeddings embeddings.npz --output results/initial
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.evaluation import analyze
from common.io import read_rows, write_json
from common.plans import initial_plan
from common.training import train_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    validation_rows = read_rows(args.validation)
    test_rows = read_rows(args.test)
    plan, rows = initial_plan(train_rows, validation_rows, test_rows)
    write_json(args.output / "plan.json", plan)

    settings = ((1.0, 17), (0.25, 17), (0.25, 29), (0.25, 43))
    for fraction, subset_seed in settings:
        result = train_experiment(
            rows,
            plan,
            args.embeddings,
            fraction=fraction,
            subset_seed=subset_seed,
        )
        metrics = analyze(result)
        percentage = round(fraction * 100)
        directory = args.output / f"budget_{percentage:03d}_seed_{subset_seed}"
        write_json(directory / "predictions.json", result)
        write_json(directory / "metrics.json", metrics)


if __name__ == "__main__":
    main()
