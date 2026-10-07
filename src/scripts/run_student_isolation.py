"""Run the Step 2 student-isolation experiment outside Colab (same runner as the notebook).

Examples:
  PYTHONPATH=src python src/scripts/run_student_isolation.py --profile cpu_smoke \
      --checkpoint outputs/tsp_50/stage1_tsp50_with_value_20260424T032357/epoch-99.pt \
      --output_dir /tmp/student_isolation_smoke --phase all
  PYTHONPATH=src python src/scripts/run_student_isolation.py --config my_config.json --phase teacher
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from am_baseline.experiments.student_isolation import (
    PHASES, StudentConfig, profile_overrides, run_phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, help='JSON with StudentConfig fields')
    parser.add_argument('--profile', choices=['main', 'confirm', 'pilot', 'cpu_smoke'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--output_dir')
    parser.add_argument('--targets', nargs='+', help='override the targets to run')
    parser.add_argument('--phase', choices=PHASES, default='all')
    args = parser.parse_args()
    fields = json.loads(args.config.read_text()) if args.config else {}
    if args.profile:
        fields.update(profile_overrides(args.profile))
    for key in ('checkpoint', 'output_dir'):
        if getattr(args, key):
            fields[key] = getattr(args, key)
    if args.targets:
        fields['targets'] = tuple(args.targets)
    for key in ('seeds', 'targets', 'learning_rates'):
        if key in fields:
            fields[key] = tuple(fields[key])
    cfg = StudentConfig(**fields)
    cfg.validate()
    print(json.dumps(asdict(cfg), indent=1))
    run_phase(cfg, args.phase)


if __name__ == '__main__':
    main()
