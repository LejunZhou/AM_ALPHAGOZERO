import os
import time
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    # Problem
    graph_size: int = 20

    # Model
    embedding_dim: int = 128
    hidden_dim: int = 128
    n_encode_layers: int = 3
    n_heads: int = 8
    tanh_clipping: float = 10.0
    normalization: str = 'batch'
    feed_forward_hidden: int = 512

    # Training
    batch_size: int = 512
    epoch_size: int = 1280000
    n_epochs: int = 100
    lr_model: float = 1e-4
    lr_critic: float = 1e-4
    lr_decay: float = 1.0
    max_grad_norm: float = 1.0
    seed: int = 1234

    # Baseline
    baseline: str = 'rollout'  # 'rollout', 'exponential', 'none'
    bl_alpha: float = 0.05
    bl_warmup_epochs: int = 1
    exp_beta: float = 0.8

    # Value head (Stage 1)
    value_enabled: bool = True
    value_hidden_dim: int = 128
    lambda_v: float = 1.0
    value_target_norm: str = 'bl'  # 'bl' (per-instance by greedy rollout cost) or 'sqrt_n'

    # Evaluation
    eval_batch_size: int = 1024
    val_size: int = 10000
    val_dataset: str = None

    # Logging & checkpointing
    log_step: int = 50
    num_workers: int = 4  # DataLoader workers; set 0 on CPU / Windows-dev
    output_dir: str = 'outputs'
    run_name: str = 'run'
    checkpoint_epochs: int = 1
    no_progress_bar: bool = False

    # W&B
    wandb_project: str = None
    wandb_entity: str = None
    wandb_mode: str = 'online'  # 'online', 'offline', 'disabled'

    # Device
    no_cuda: bool = False

    # Loading
    load_path: str = None
    resume: str = None
    epoch_start: int = 0

    # Derived (set in __post_init__)
    use_cuda: bool = field(init=False)
    device: torch.device = field(init=False)
    save_dir: str = field(init=False)

    def __post_init__(self):
        self.use_cuda = torch.cuda.is_available() and not self.no_cuda
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.run_name = "{}_{}".format(self.run_name, time.strftime("%Y%m%dT%H%M%S"))
        self.save_dir = os.path.join(
            self.output_dir,
            "tsp_{}".format(self.graph_size),
            self.run_name
        )
        assert self.epoch_size % self.batch_size == 0, \
            "Epoch size must be integer multiple of batch size!"

    @classmethod
    def from_args(cls, args=None):
        """Create config from CLI arguments."""
        import argparse
        parser = argparse.ArgumentParser(description="AM for TSP")

        # Add all fields as CLI args
        parser.add_argument('--graph_size', type=int, default=20)
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--n_encode_layers', type=int, default=3)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--tanh_clipping', type=float, default=10.)
        parser.add_argument('--normalization', default='batch')
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--epoch_size', type=int, default=1280000)
        parser.add_argument('--n_epochs', type=int, default=100)
        parser.add_argument('--lr_model', type=float, default=1e-4)
        parser.add_argument('--lr_critic', type=float, default=1e-4)
        parser.add_argument('--lr_decay', type=float, default=1.0)
        parser.add_argument('--max_grad_norm', type=float, default=1.0)
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--baseline', default='rollout')
        parser.add_argument('--bl_alpha', type=float, default=0.05)
        parser.add_argument('--bl_warmup_epochs', type=int, default=None)
        parser.add_argument('--exp_beta', type=float, default=0.8)
        parser.add_argument('--no_value', action='store_false', dest='value_enabled',
                            help='Disable the Stage 1 auxiliary value head')
        parser.add_argument('--value_hidden_dim', type=int, default=128)
        parser.add_argument('--lambda_v', type=float, default=1.0,
                            help='Weight on the value MSE loss (0 disables the value head loss)')
        parser.add_argument('--value_target_norm', choices=['bl', 'sqrt_n'], default='bl',
                            help="Normalize cost-to-go targets by per-instance greedy rollout "
                                 "cost ('bl') or by sqrt(graph_size) ('sqrt_n')")
        parser.add_argument('--eval_batch_size', type=int, default=1024)
        parser.add_argument('--val_size', type=int, default=10000)
        parser.add_argument('--val_dataset', type=str, default=None)
        parser.add_argument('--log_step', type=int, default=50)
        parser.add_argument('--num_workers', type=int, default=4,
                            help='DataLoader worker processes (0 = synchronous main-thread loading)')
        parser.add_argument('--output_dir', default='outputs')
        parser.add_argument('--run_name', default='run')
        parser.add_argument('--checkpoint_epochs', type=int, default=1)
        parser.add_argument('--no_progress_bar', action='store_true')
        parser.add_argument('--no_cuda', action='store_true')
        parser.add_argument('--wandb_project', type=str, default=None,
                            help='W&B project name (None = disable W&B)')
        parser.add_argument('--wandb_entity', type=str, default=None,
                            help='W&B team/user name')
        parser.add_argument('--wandb_mode', type=str, default='online',
                            choices=['online', 'offline', 'disabled'])
        parser.add_argument('--load_path', type=str, default=None)
        parser.add_argument('--resume', type=str, default=None)
        parser.add_argument('--epoch_start', type=int, default=0)

        opts = parser.parse_args(args)
        kwargs = vars(opts)

        # Handle bl_warmup_epochs default
        if kwargs['bl_warmup_epochs'] is None:
            kwargs['bl_warmup_epochs'] = 1 if kwargs['baseline'] == 'rollout' else 0

        # Remove None values so dataclass defaults apply
        kwargs = {k: v for k, v in kwargs.items() if v is not None or k in ('val_dataset', 'load_path', 'resume')}

        return cls(**kwargs)
