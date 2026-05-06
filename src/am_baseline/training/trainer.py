import os
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from am_baseline.model.attention_model import set_decode_type
from am_baseline.utils.tensor_ops import move_to, value_targets_from_edges


def validate(model, dataset, opts):
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost


def _r2(values, targets):
    """R^2 coefficient of determination. Returns float."""
    ss_res = ((targets - values) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum().clamp(min=1e-12)
    return (1.0 - ss_res / ss_tot).item()


def validate_value(model, dataset, opts):
    """
    Diagnostic pass for the Stage 1 value head.

    Runs greedy decoding on the validation dataset, collects per-step (value, target)
    pairs, and reports R^2 overall and bucketed by tour position
    (early = first quarter, mid = middle half, late = last quarter).

    Uses the current model's greedy cost as per-instance normalizer when
    opts.value_target_norm == 'bl' (same shape as the training-time normalizer).
    Returns None if the value head is disabled.
    """
    if not getattr(model, 'value_enabled', False) or model.value_head is None:
        return None

    set_decode_type(model, "greedy")
    model.eval()

    all_v, all_t = [], []
    with torch.no_grad():
        for bat in DataLoader(dataset, batch_size=opts.eval_batch_size):
            x = move_to(bat, opts.device)
            cost, _ll, pi, values = model(x, return_pi=True, compute_values=True)
            edge_costs = model.problem.get_edge_costs(x, pi)
            rtg = value_targets_from_edges(edge_costs)
            if getattr(opts, 'value_target_norm', 'bl') == 'bl':
                Z = cost.clamp(min=1e-6).unsqueeze(-1)
            else:
                Z = float(opts.graph_size) ** 0.5
            targets = rtg / Z
            all_v.append(values.cpu())
            all_t.append(targets.cpu())

    V = torch.cat(all_v, dim=0)   # (M, N)
    T = torch.cat(all_t, dim=0)

    N = V.size(1)
    q = max(1, N // 4)
    early = slice(0, q)
    mid = slice(q, N - q) if N - q > q else slice(q, N)
    late = slice(N - q, N)

    residual = T - V
    metrics = {
        'r2_overall': _r2(V, T),
        'r2_early': _r2(V[:, early], T[:, early]),
        'r2_mid': _r2(V[:, mid], T[:, mid]),
        'r2_late': _r2(V[:, late], T[:, late]),
        'value_loss': (residual ** 2).mean().item(),   # val-set MSE (matches training loss formula)
        'residual_mean': residual.mean().item(),        # systematic bias (0 = unbiased)
        'value_mean': V.mean().item(),
        'target_mean': T.mean().item(),
    }
    print('Value diagnostic: R2_overall={:.3f}  early={:.3f}  mid={:.3f}  late={:.3f}  '
          'mse={:.4f}  bias={:+.4f}'
          .format(metrics['r2_overall'], metrics['r2_early'],
                  metrics['r2_mid'], metrics['r2_late'],
                  metrics['value_loss'], metrics['residual_mean']))
    return metrics


def rollout(model, dataset, opts):
    """Greedy rollout for evaluation. Used by baselines and validation."""
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size),
                        disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset,
                problem, logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(
        epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=opts.use_cuda,
        persistent_workers=(opts.num_workers > 0),
    )

    # Train mode
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, logger, opts)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # Save checkpoint
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'baseline': baseline.state_dict()
        }
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))

    # Validate
    avg_reward = validate(model, val_dataset, opts)

    # Value head diagnostic (Stage 1)
    value_metrics = validate_value(model, val_dataset, opts)

    # Baseline epoch callback
    baseline_updated = baseline.epoch_callback(model, epoch)

    # Log epoch metrics
    if logger is not None:
        logger.log_epoch(epoch, avg_reward.item(), epoch_duration,
                         optimizer.param_groups[0]['lr'],
                         baseline_updated=baseline_updated,
                         value_metrics=value_metrics)

    # LR scheduler
    lr_scheduler.step()


def train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, logger, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    value_enabled = getattr(model, 'value_enabled', False) and getattr(opts, 'lambda_v', 0.0) > 0

    # Forward pass
    if value_enabled:
        cost, log_likelihood, pi, values = model(x, return_pi=True, compute_values=True)
    else:
        cost, log_likelihood = model(x)
        pi = None
        values = None

    # Baseline
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # REINFORCE loss (unchanged — value head does NOT enter the policy gradient)
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()

    # Value loss (Stage 1: auxiliary MSE on normalized V_CURRENT cost-to-go)
    # target[i] = cost of edges still to traverse from state s_i onwards,
    # INCLUDING the edge selected at step i (V_CURRENT semantics).
    value_loss = None
    if value_enabled:
        edge_costs = model.problem.get_edge_costs(x, pi)                # (B, N)
        rtg = value_targets_from_edges(edge_costs).detach()             # (B, N)
        # Normalizer: per-instance bl_val (preferred) or sqrt(N) fallback
        use_bl_norm = (
            getattr(opts, 'value_target_norm', 'bl') == 'bl'
            and isinstance(bl_val, torch.Tensor) and bl_val.dim() > 0
        )
        if use_bl_norm:
            Z = bl_val.detach().clamp(min=1e-6).unsqueeze(-1)           # (B, 1)
        else:
            Z = float(opts.graph_size) ** 0.5
        targets = rtg / Z                                               # (B, N)
        value_loss = F.mse_loss(values, targets)

    # Combine
    loss = reinforce_loss + bl_loss
    if value_loss is not None:
        loss = loss + opts.lambda_v * value_loss

    # Backward
    optimizer.zero_grad()
    loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0 and logger is not None:
        logger.log_step(step, epoch, batch_id, cost, grad_norms,
                        log_likelihood, reinforce_loss, bl_loss,
                        value_loss=value_loss)


def train_step_alphazero(model, optimizer, batch, opts):
    """Stage 4 distillation training step (AlphaGo-Zero-style).

    Implements the loss in `_plans/stage4_algorithm_spec.md` §3 over a
    stratified-by-step minibatch produced by `MCTSReplayBuffer.sample()` /
    `sample_step()`. All rows in `batch` share the same scalar `state_i`.

    Args:
        model: AttentionModel with encoder, decoder, and value_head heads.
        optimizer: torch.optim.Optimizer; L2 regularization is applied via
            `weight_decay`, NOT in the loss expression.
        batch: dict from `MCTSReplayBuffer.sample(...)`. Required keys:
            state_i (int), coords (B, N, 2), visited (B, N) bool,
            first_a / prev_a (B,), lengths (B,), pi (B, N), z (B,).
        opts: object exposing `lambda_v` (value-loss weight, default 1.0)
            and `device`. `max_grad_norm` is honored if present (>0).

    Returns:
        dict of metrics: policy_loss, value_loss, total_loss,
        mean_entropy_pi, mean_z, gradient_norm.
    """
    from am_baseline.training.coach import reconstruct_state

    device = getattr(opts, "device", torch.device("cpu"))
    if not isinstance(device, torch.device):
        device = torch.device(device)

    model.train()

    # Move minibatch to model device. Buffer is CPU-resident at default
    # capacity (~520 MB); transfer happens here, not in the buffer.
    coords = batch["coords"].to(device=device, dtype=torch.float32)
    pi_target = batch["pi"].to(device=device, dtype=torch.float32)

    # ---- Value-target normalization ---------------------------------------
    # The buffer always stores `z = cost_to_go / bl_val` (frozen at self-play
    # time). At training time we re-derive the actual target based on
    # `opts.value_target_norm` to disentangle the head from bl_val drift:
    #   "bl"     : keep z = cost_to_go / bl_val_training (current default;
    #              suffers from calibration drift between training-time and
    #              MCTS-time bl_val).
    #   "sqrt_n" : z = cost_to_go / sqrt(N) — instance-independent,
    #              time-invariant; no drift.
    #   "none"   : z = cost_to_go (raw cost-to-go; head learns absolute
    #              remaining cost. MCTS leaf-eval converts via division by
    #              current bl_val.).
    z_buf = batch["z"].to(device=device, dtype=torch.float32)         # cost_to_go / bl_val
    bl_val_b = batch["bl_val"].to(device=device, dtype=torch.float32)
    norm_mode = str(getattr(opts, "value_target_norm", "bl"))
    if norm_mode == "bl":
        z_target = z_buf
    elif norm_mode == "none":
        # Recover raw cost-to-go = z_buf * bl_val.
        z_target = z_buf * bl_val_b
    elif norm_mode == "sqrt_n":
        graph_size = float(getattr(opts, "graph_size", coords.size(1)))
        z_target = (z_buf * bl_val_b) / (graph_size ** 0.5)
    else:
        raise ValueError(f"Unknown value_target_norm: {norm_mode!r}")

    # ---- Forward ----------------------------------------------------------
    # Critical (plan B.2): gradient must flow through the same compute graph
    # used during self-play. encode + precompute_decoder are called INSIDE
    # the train step so the encoder's parameters receive gradient.
    encoded = model.encode(coords)                         # (B, N, embed_dim)
    fixed = model.precompute_decoder(encoded)              # AttentionModelFixed

    # Reconstruct state with scalar state.i (stratified batch).
    state = reconstruct_state(
        {
            "state_i": batch["state_i"],
            "coords": coords,
            "visited": batch["visited"].to(device=device),
            "first_a": batch["first_a"].to(device=device),
            "prev_a": batch["prev_a"].to(device=device),
            "lengths": batch["lengths"].to(device=device),
        },
        device=device,
    )

    log_p, mask, glimpse = model.decode_step(fixed, state, return_glimpse=True)
    # decode_step returns (B, 1, N) for log_p / mask. Strip the per-step axis
    # before the cross-entropy to avoid a (B, B, N) broadcast against pi.
    log_p = log_p.squeeze(1)                               # (B, N)
    mask = mask.squeeze(1)                                 # (B, N) bool — True == visited / illegal
    if model.value_head is None:
        raise RuntimeError(
            "train_step_alphazero requires model.value_head to be enabled "
            "(value_enabled=True at construction)."
        )
    v = model.value_head(glimpse)                          # (B,)

    # ---- Policy distillation cross-entropy --------------------------------
    # log_p has -inf at masked positions (the decoder writes -inf into
    # visited cities). pi_target is masked-zero on visited cities by
    # construction (raw normalized visit counts; visited cities had N=0).
    # 0 * -inf = NaN in IEEE float, so we replace -inf with 0 before the
    # multiply. Masked entries then contribute 0 to the sum, which is the
    # correct cross-entropy under the legality mask.
    log_p_safe = torch.where(mask, torch.zeros_like(log_p), log_p)
    policy_loss = -(pi_target * log_p_safe).sum(dim=-1).mean()

    # ---- Value loss -------------------------------------------------------
    # `z_target` was computed above per `opts.value_target_norm` — see comment
    # block. MSE against the value head's per-step output. The head's
    # absolute output range scales with the chosen normalizer:
    #   "bl"     ⇒ v ∈ [0, ~1.5]    (matches Stage 1 shape)
    #   "sqrt_n" ⇒ v ∈ [0, ~1]      (sqrt(N=20) ≈ 4.47 normalizer)
    #   "none"   ⇒ v ∈ [0, ~5]      (raw TSP-20 cost-to-go)
    value_loss = F.mse_loss(v, z_target)

    # ---- Total loss -------------------------------------------------------
    lambda_v = float(getattr(opts, "lambda_v", 1.0))
    total_loss = policy_loss + lambda_v * value_loss

    # ---- Backward + step --------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    # Optional gradient clipping. We compute the unclipped grad-norm for
    # logging regardless of whether clipping is active.
    max_grad_norm = float(getattr(opts, "max_grad_norm", 0.0))
    if max_grad_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=math.inf
        )
    optimizer.step()

    # ---- Metrics ----------------------------------------------------------
    # Entropy of the *target* pi (sanity diagnostic — degenerates as MCTS
    # becomes more peaked). 0 * log(0) := 0; mask-aware via clamp.
    pi_clamped = pi_target.clamp(min=1e-12)
    entropy_pi = -(pi_target * pi_clamped.log()).sum(dim=-1).mean()

    return {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
        "mean_entropy_pi": float(entropy_pi.detach().cpu()),
        "mean_z": float(z_target.mean().detach().cpu()),
        "gradient_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
    }
