"""Run the same frozen-policy experiment as the Colab notebook from a JSON config."""
import argparse
import json
from pathlib import Path

from am_baseline.experiments.value_repair import ExperimentConfig, run_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=['check', 'data', 'train', 'probe', 'search', 'report', 'all'], default='all')
    args = parser.parse_args()
    run_phase(ExperimentConfig(**json.loads(args.config.read_text())), args.phase)


if __name__ == '__main__':
    main()
