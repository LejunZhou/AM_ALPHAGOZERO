"""Stage 4 launcher — AlphaGo-Zero-style MCTS self-improvement on TSP.

Mirrors Stage 1's `src/scripts/train.py` plus Stage 4 self-play / MCTS flags.
The script:

  1. Parses CLI options.
  2. Loads a Stage 1 warm-start checkpoint (θ★) via `--load_path` (required).
  3. Constructs a validation dataset of size `--val_size`.
  4. Constructs `MCTSCoach(model, problem, opts, val_dataset, device)` AFTER
     `opts.val_size` is finalized — the init-order trap caught in review:
     `RolloutBaseline.__init__` snapshots `opts.val_size` at construction time,
     so any later override would be a silent no-op.
  5. Optionally calls `coach.load_checkpoint(opts.resume_from)` to resume.
  6. Calls `coach.learn(opts.n_iterations)`.

Usage:
    python src/scripts/train_alphazero.py \
        --load_path outputs/tsp_20/stage1_tsp20_canonical_*/epoch-99.pt \
        --graph_size 20 --n_iterations 100 --M_instances 1000

Output dir: `outputs/tsp_<graph_size>/<run_name>_<timestamp>/`.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import pprint as pp
import time

import numpy as np
import torch

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP
from am_baseline.training.coach import MCTSCoach
from am_baseline.utils.misc import torch_load_cpu


def parse_opts(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 4 — AlphaGo-Zero-style MCTS self-improvement on TSP",
    )

    # ---- Optional: Stage 1 warm-start (omit for from-scratch / Phase F.6) ----
    parser.add_argument(
        '--load_path', type=str, default=None,
        help='Optional Stage 1 checkpoint (e.g. epoch-99.pt) to warm-start θ★. '
             'Omit for from-scratch random-init AGZ training (proposal Phase F.6).',
    )

    # ---- Stage 4 flags -------------------------------------------------------
    parser.add_argument('--n_iterations', type=int, default=100,
                        help='Number of coach iterations to run (default 100).')
    parser.add_argument('--M_instances', type=int, default=1000,
                        help='Self-play instances generated per iteration.')
    parser.add_argument('--n_simulations_train', type=int, default=50,
                        help='K — MCTS simulations per root during self-play. When '
                             '`--n_simulations_schedule step_bracket` is set, this '
                             'value is K_mid (the high-K bucket for tour-steps 1..N/2).')
    parser.add_argument('--n_simulations_schedule', type=str, default='const',
                        choices=['const', 'step_bracket'],
                        help='K-per-step schedule. `const` uses --n_simulations_train '
                             'at every tour-step (existing behavior). `step_bracket` '
                             'uses K(0)=--n_simulations_first, K(1..N/2)=--n_simulations_train, '
                             'K(N/2+1..N-2)=--n_simulations_late, K(N-1)=--n_simulations_last. '
                             'Motivation: TSP first-step is rotation-invariant (all '
                             'starting cities yield the same optimal tour cost), and '
                             'the last step is forced — both deserve fewer sims. See '
                             '_progress/stage4_progress.md F.6.1.5.')
    parser.add_argument('--n_simulations_first', type=int, default=5,
                        help='K at tour-step 0 when schedule=step_bracket.')
    parser.add_argument('--n_simulations_late', type=int, default=10,
                        help='K at tour-steps N/2+1..N-2 when schedule=step_bracket.')
    parser.add_argument('--n_simulations_last', type=int, default=1,
                        help='K at tour-step N-1 (forced) when schedule=step_bracket.')
    parser.add_argument('--buffer_capacity', type=int, default=200000,
                        help='Replay buffer capacity in instances. AGZ-proportional '
                             'pilot uses 50_000 (~50-iter window); main run uses 200_000.')
    parser.add_argument('--train_steps_per_iter', type=int, default=200,
                        help='Minibatch updates per coach iteration.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Train minibatch size (rows drawn from buffer).')
    parser.add_argument('--gate_every', type=int, default=5,
                        help='Gating cadence — paired t-test every k iterations.')
    parser.add_argument('--gate_mode', type=str, default='ttest',
                        choices=['ttest', 'always', 'never'],
                        help='Gating decision rule. ttest = Stage 1 paired-t α=0.05 '
                             '(default); always = accept every gating event (Phase G.5.c, '
                             'AlphaZero-style no-gating); never = reject every event '
                             '(diagnostic: freezes best_model = warm-start).')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze init_embed + embedder (the encoder) and only train '
                             'decoder + value_head. Tests the "shared encoder is the noise '
                             'channel" hypothesis from F.3 v3-v5 plateau diagnosis.')
    parser.add_argument('--temperature_schedule', type=str, default='step30',
                        choices=['const', 'step10', 'step30', 'step50'],
                        help='Per-tour-step temperature schedule for action selection σ_t. '
                             'step10/30/50 keep τ=cfg.temperature for first 10/30/50%% of '
                             'tour-steps (⌈p·N⌉) and switch to τ=0 (argmax) afterwards. '
                             'Training target π_t is always raw τ=1 (decoupled per spec §4.2).')
    parser.add_argument('--dirichlet_epsilon', type=float, default=0.25,
                        help='Dirichlet root-noise mixing weight (AGZ default 0.25).')
    parser.add_argument('--dirichlet_alpha_factor', type=float, default=10.0,
                        help='Dirichlet concentration scale: α = factor / N (AGZ default 10/N).')
    parser.add_argument('--lambda_v', type=float, default=1.0,
                        help='Value-loss weight in the joint MSE+CE objective '
                             '(AGZ default 1.0). Set 0 for policy-only rollout-teacher '
                             'distillation; value_loss is still logged but contributes '
                             'no gradients.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 weight decay applied via the optimizer (AGZ canonical 1e-4).')
    parser.add_argument('--lr_model', type=float, default=1e-4,
                        help='Adam learning rate for the working model.')
    parser.add_argument('--lr_decay', type=float, default=1.0,
                        help='Multiplicative lr decay factor per step-down event. With '
                             '--lr_decay_step_size=1 (default), this is a per-iter exponential '
                             'decay: lr(iter k) = lr_model * lr_decay**k. With '
                             '--lr_decay_step_size=N (N>1), this is a step-wise schedule: '
                             'lr drops by lr_decay every N iters (StepLR-like). '
                             'Default 1.0 = no decay.')
    parser.add_argument('--lr_decay_step_size', type=int, default=1,
                        help='Number of iterations between lr step-down events. 1 (default) gives '
                             'smooth per-iter exponential decay; N>1 gives step-wise decay '
                             '(lr unchanged within a 100-iter window if step_size=100). '
                             'Useful for AGZ-canonical step-anneal schedules.')
    parser.add_argument('--leaf_eval', type=str, default='value_head',
                        choices=['value_head', 'rollout', 'mix'],
                        help='MCTS leaf evaluator: AGZ value head (default), AlphaGo-Lee-style rollout, '
                             "or AlphaGo-Fan/Lee 'mix' (convex blend λ·v_head + (1−λ)·v_rollout).")
    parser.add_argument('--mix_lambda', type=float, default=0.5,
                        help="Mix coefficient λ ∈ [0,1] when --leaf_eval=mix. "
                             "λ=0 collapses to pure rollout; λ=1 to pure value_head. "
                             "Stage 5 §H starts the from-scratch sweep around λ=0.5.")
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Optional path to a Stage 4 iter-{i}.pt checkpoint. Resumes coach state.')

    # ---- Stage-1-shared problem / runtime / logging --------------------------
    parser.add_argument('--graph_size', type=int, default=20,
                        help='TSP graph size N.')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Validation set size (REUSED by RolloutBaseline for gating; '
                             'no separate gate_val_size — see plan F.1 note).')
    parser.add_argument('--val_seed', type=int, default=42,
                        help='Seed for the per-iter val_dataset draw. Pinning makes per-run '
                             'iterations.csv val_avg_cost directly comparable across runs and '
                             'against Stage 1 canonical baseline. Default = 42 matches '
                             'compare_stage1_vs_stage4.py for apples-to-apples eval.')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='Validation/rollout batch size.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient-norm clip (0 disables).')
    parser.add_argument('--mcts_batch_size', type=int, default=1000,
                        help='Instance-parallelism CHUNK SIZE in '
                             'CppBatchMCTSSolver.solve_batch (NOT the per-NN-forward '
                             'batch). Each iter processes ceil(M / mcts_batch_size) '
                             'sequential chunks; cross-instance NN-batching pools '
                             'requests within a chunk. With M=1000 (default), '
                             'mcts_batch_size=1000 means 1 chunk per iter at full GPU '
                             'parallelism — ~5x faster than the prior default of 64. '
                             'See _progress/stage4_progress.md for the sweep result.')
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance level for the gating paired t-test.')
    parser.add_argument('--no_progress_bar', action='store_true')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Force CPU even when CUDA is available.')

    # ---- Output / logging ----------------------------------------------------
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Top-level outputs root.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name (default: stage4_<timestamp>).')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging.')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'])

    # ---- Stage 1 model-arch flags (so loading the checkpoint matches) --------
    # These MUST mirror the architecture of the Stage 1 checkpoint being
    # warm-started; defaults match the canonical Stage 1 run.
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_encode_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--tanh_clipping', type=float, default=10.0)
    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--feed_forward_hidden', type=int, default=512)
    parser.add_argument('--no_value', action='store_false', dest='value_enabled',
                        help='Disable the auxiliary value head (Stage 4 requires it; default ON).')
    parser.add_argument('--value_hidden_dim', type=int, default=128)
    parser.add_argument('--value_target_norm', choices=['bl', 'sqrt_n', 'none'], default='bl',
                        help='Value-head training target normalizer: '
                             '"bl" (default) = z = cost_to_go / bl_val (AGZ-style; '
                             'has calibration drift between training-time and MCTS-time bl_val); '
                             '"sqrt_n" = z = cost_to_go / sqrt(N) (instance-independent, '
                             'time-invariant — fixes drift); '
                             '"none" = z = cost_to_go (raw cost; cleanest semantics, '
                             'MCTS divides by current bl_val at leaf-eval time).')

    return parser.parse_args(argv)


def _finalize_opts(opts):
    """Fill in derived fields that downstream code expects (mirroring Config)."""
    if float(opts.lambda_v) < 0.0:
        raise ValueError(f"--lambda_v must be non-negative, got {opts.lambda_v}")

    # Device.
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device('cuda:0' if opts.use_cuda else 'cpu')

    # Run name + save dir.
    timestamp = time.strftime('%Y%m%dT%H%M%S')
    if opts.run_name is None or opts.run_name == '':
        opts.run_name = f'stage4_{timestamp}'
    else:
        # Append timestamp so re-running with the same name doesn't clobber.
        opts.run_name = f'{opts.run_name}_{timestamp}'
    opts.save_dir = os.path.join(
        opts.output_dir,
        f'tsp_{opts.graph_size}',
        opts.run_name,
    )

    # `no_wandb` cascades to wandb_mode='disabled' so MetricsLogger skips W&B.
    if opts.no_wandb:
        opts.wandb_mode = 'disabled'
        opts.wandb_project = None

    # Stage 4 doesn't use lr_decay, but RolloutBaseline + train infra read some
    # Stage 1 attributes. Provide harmless defaults so the namespace is complete.
    if not hasattr(opts, 'lr_critic'):
        opts.lr_critic = 1e-4
    if not hasattr(opts, 'lr_decay'):
        opts.lr_decay = 1.0

    # Resolve --n_simulations_schedule into a concrete length-N list which
    # MCTSCoach.learn forwards to make_self_play_config (and on into
    # MCTSConfig.n_simulations_per_step). None = scalar n_simulations_train
    # used at every tour-step (existing behavior).
    sched = getattr(opts, 'n_simulations_schedule', 'const')
    if sched == 'const':
        opts.n_simulations_per_step = None
    elif sched == 'step_bracket':
        N = int(opts.graph_size)
        K_first = int(opts.n_simulations_first)
        K_mid = int(opts.n_simulations_train)
        K_late = int(opts.n_simulations_late)
        K_last = int(opts.n_simulations_last)
        if N <= 1:
            opts.n_simulations_per_step = [K_first]
        else:
            mid_cutoff = N // 2  # high bucket includes t=N//2 (Interpretation A)
            schedule = []
            for t in range(N):
                if t == 0:
                    schedule.append(K_first)
                elif t == N - 1:
                    schedule.append(K_last)
                elif t <= mid_cutoff:
                    schedule.append(K_mid)
                else:
                    schedule.append(K_late)
            opts.n_simulations_per_step = schedule
    else:
        raise ValueError(f"Unknown n_simulations_schedule: {sched!r}")
    return opts


def run(opts):
    pp.pprint(vars(opts))

    torch.manual_seed(opts.seed)

    os.makedirs(opts.save_dir, exist_ok=True)
    with open(os.path.join(opts.save_dir, 'args.json'), 'w') as f:
        json.dump(
            {k: v for k, v in vars(opts).items()
             if not isinstance(v, torch.device)},
            f, indent=True, default=str,
        )

    problem = TSP

    # ---- Step 2: Build model (random-init or load Stage 1 warm-start) -------
    model = AttentionModel(opts).to(opts.device)
    if opts.load_path is not None:
        print(f'  [*] Loading Stage 1 warm-start from {opts.load_path}')
        load_data = torch_load_cpu(opts.load_path)
        if 'model' in load_data:
            model.load_state_dict({**model.state_dict(), **load_data['model']})
        else:
            # Some checkpoints save the bare state dict (no nesting); accept that.
            model.load_state_dict({**model.state_dict(), **load_data})
    else:
        print('  [*] No --load_path; starting from random init (proposal Phase F.6).')

    # ---- Step 3: Validation dataset (consumed by RolloutBaseline + validate) -
    # Pin the val draw to opts.val_seed so per-iter val_avg_cost is comparable
    # across runs and against Stage 1 canonical (see methodology fix in F.6).
    _val_torch_state = torch.get_rng_state()
    _val_np_state = np.random.get_state()
    torch.manual_seed(int(opts.val_seed))
    np.random.seed(int(opts.val_seed))
    val_dataset = TSP.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size,
    )
    torch.set_rng_state(_val_torch_state)
    np.random.set_state(_val_np_state)

    # ---- Step 4: Construct the coach AFTER opts.val_size is finalized --------
    # (Init-order trap — see file docstring + plan F.1 note.)
    coach = MCTSCoach(
        model=model,
        problem=problem,
        opts=opts,
        val_dataset=val_dataset,
        device=opts.device,
    )

    # ---- Step 5: Optional resume --------------------------------------------
    if opts.resume_from is not None:
        print(f'  [*] Resuming Stage 4 coach from {opts.resume_from}')
        coach.load_checkpoint(opts.resume_from)

        # `optimizer.load_state_dict` restores the saved lr (which becomes the
        # scheduler's `initial_lr` baseline). Without this override, the
        # `--lr_model` CLI arg has no effect on a resumed run. So if the user
        # passed an `--lr_model` that differs from the saved value, force it
        # into both the optimizer's current `lr`/`initial_lr` AND the scheduler's
        # `base_lrs` so subsequent `scheduler.step()` calls use the new base.
        new_lr = float(opts.lr_model)
        loaded_lr = float(coach.optimizer.param_groups[0]['lr'])
        if abs(new_lr - loaded_lr) > 1e-12:
            print(f'  [*] Overriding loaded lr {loaded_lr:.2e} → {new_lr:.2e} '
                  f'(from --lr_model)')
            for pg in coach.optimizer.param_groups:
                pg['lr'] = new_lr
                pg['initial_lr'] = new_lr
            if hasattr(coach, 'lr_scheduler') and hasattr(coach.lr_scheduler, 'base_lrs'):
                coach.lr_scheduler.base_lrs = (
                    [new_lr] * len(coach.lr_scheduler.base_lrs)
                )

    # ---- Step 6: Run the loop -----------------------------------------------
    try:
        coach.learn(int(opts.n_iterations))
    finally:
        coach.close()

    print(f'Stage 4 training complete. Results saved to {opts.save_dir}')


if __name__ == '__main__':
    opts = parse_opts()
    opts = _finalize_opts(opts)
    run(opts)
