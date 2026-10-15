"""Repeat text-group partitions of the development pool per project.
Rows use sample_id, text, project_id, label_id, and leakage_group_id.
Run separately for each of the four projects. Original test groups are excluded.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.evaluation import analyze
from common.io import read_rows, write_json
from common.plans import text_resplit_plan
from common.training import train_experiment

SPLIT_SEEDS = (10517, 10529, 10543)


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
    plan, rows = text_resplit_plan(
        train_rows,
        validation_rows,
        test_rows,
        seeds=SPLIT_SEEDS,
    )
    write_json(args.output / "plan.json", plan)

    for fraction in (1.0, 0.25):
        result = train_experiment(
            rows,
            plan,
            args.embeddings,
            fraction=fraction,
            subset_seed=17,
        )
        metrics = analyze(result)
        percentage = round(fraction * 100)
        directory = args.output / f"budget_{percentage:03d}_seed_17"
        write_json(directory / "predictions.json", result)
        write_json(directory / "metrics.json", metrics)


if __name__ == "__main__":
    main()
