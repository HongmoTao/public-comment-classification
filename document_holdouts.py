"""MBTA: compare three text-grouped and three document-grouped holdouts.
Inputs: original MBTA train+validation pool and full-project source-group map.
The initial test excerpts must already be excluded from the pool.
Both protocols retain 24 outputs and score the same 18 evaluation categories.
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
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.pool)
    mapping = read_json(args.groups)
    for mode in ('text', 'source'):
        plan = document_plan(rows, mapping, mode=mode)
        destination = args.output / mode
        write_json(destination / 'plan.json', plan)
        result = train_experiment(rows, plan, args.embeddings)
        write_json(destination / 'predictions.json', result)
        write_json(destination / 'metrics.json', analyze(result))


if __name__ == '__main__':
    main()
