"""MBTA: complete eligible-document coverage with three outer and two inner folds.
Inputs: MBTA development pool, source map, vectors and the recorded group witness.
All 178 eligible groups are tested once; eight pinned groups stay in training.
"""
import argparse
import sys
from pathlib import Path

REPRODUCTION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRODUCTION_ROOT))

from common.io import read_json, read_rows, write_json
from common.plans import document_plan
from common.training import train_experiment
from common.evaluation import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--groups', type=Path, required=True)
    parser.add_argument('--witness', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--historical-plan', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.pool)
    mapping = read_json(args.groups)
    witness = read_json(args.witness)
    plan = document_plan(rows, mapping, mode='nested', witness=witness)
    write_json(args.output / 'plan.json', plan)
    result = train_experiment(rows, plan, args.embeddings)
    write_json(args.output / 'predictions.json', result)
    historical = None
    if args.historical_plan is not None:
        historical = read_json(args.historical_plan)
    metrics = analyze(result, historical)
    write_json(args.output / 'metrics.json', metrics)


if __name__ == '__main__':
    main()
