"""
Training entry point for the Attention Model on TSP.

Usage:
  python src/scripts/train.py --graph_size 20 --baseline rollout
  python src/scripts/train.py --graph_size 20 --batch_size 32 --epoch_size 640 --n_epochs 3  # smoke test
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pprint as pp
import torch
import torch.optim as optim

from am_baseline.config import Config
from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP
from am_baseline.baseline.baselines import (
    NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
)
from am_baseline.training.trainer import train_epoch, validate, rollout
from am_baseline.training.logging import MetricsLogger
from am_baseline.utils.misc import torch_load_cpu


def run(opts):
    pp.pprint(vars(opts))

    torch.manual_seed(opts.seed)

    os.makedirs(opts.save_dir, exist_ok=True)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump({k: v for k, v in vars(opts).items()
                   if not isinstance(v, torch.device)}, f, indent=True, default=str)

    # Logger (CSV always, W&B if --wandb_project is set)
    wandb_config = {k: v for k, v in vars(opts).items()
                    if not isinstance(v, torch.device)} if opts.wandb_project else None
    logger = MetricsLogger(
        opts.save_dir,
        wandb_project=opts.wandb_project,
        wandb_entity=opts.wandb_entity,
        wandb_group=f"tsp_{opts.graph_size}",
        wandb_name=opts.run_name,
        wandb_mode=opts.wandb_mode,
        wandb_config=wandb_config,
    )

    problem = TSP

    # Load checkpoint data if resuming
    load_data = {}
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Model
    model = AttentionModel(opts).to(opts.device)

    # Load model weights if available
    if 'model' in load_data:
        model.load_state_dict({**model.state_dict(), **load_data['model']})

    # Baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts, rollout)
    elif opts.baseline == 'none' or opts.baseline is None:
        baseline = NoBaseline()
    else:
        raise ValueError("Unknown baseline: {}".format(opts.baseline))

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # LR scheduler
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Validation dataset
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset)

    # Resume handling
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda and 'cuda_rng_state' in load_data:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    # Training loop
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_epoch(
            model, optimizer, baseline, lr_scheduler,
            epoch, val_dataset, problem, logger, opts
        )

    logger.close()
    print("Training complete. Results saved to {}".format(opts.save_dir))


if __name__ == "__main__":
    opts = Config.from_args()
    run(opts)
