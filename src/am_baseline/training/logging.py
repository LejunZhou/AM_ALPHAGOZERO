import os
import csv
import time
import tempfile

import torch


class MetricsLogger:
    """CSV-based metrics logger with optional TensorBoard and W&B support."""

    def __init__(self, log_dir, use_tensorboard=False,
                 wandb_project=None, wandb_entity=None, wandb_group=None,
                 wandb_name=None, wandb_mode='online', wandb_config=None,
                 track_gpu_memory=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Gate GPU-memory sampling: explicit flag wins, else auto-on if CUDA hardware present.
        # Caller (scripts/train.py) should pass opts.use_cuda so CPU runs skip GPU rows.
        self._track_gpu_memory = (
            torch.cuda.is_available() if track_gpu_memory is None
            else bool(track_gpu_memory)
        )

        # CSV logger (always active)
        self.csv_path = os.path.join(log_dir, 'metrics.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'step', 'epoch', 'batch_id', 'avg_cost', 'actor_loss', 'nll',
            'grad_norm', 'grad_norm_clipped', 'value_loss',
            'gpu_mem_peak_mb', 'gpu_mem_alloc_mb', 'time'
        ])

        # Epoch-level CSV
        self.epoch_csv_path = os.path.join(log_dir, 'epochs.csv')
        self.epoch_csv_file = open(self.epoch_csv_path, 'w', newline='')
        self.epoch_csv_writer = csv.writer(self.epoch_csv_file)
        self.epoch_csv_writer.writerow([
            'epoch', 'val_avg_cost', 'epoch_duration', 'lr', 'baseline_updated',
            'val_value_r2_overall', 'val_value_r2_early', 'val_value_r2_mid', 'val_value_r2_late',
            'val_value_loss', 'val_value_residual_mean', 'val_value_mean', 'val_target_mean'
        ])

        # Stage 4: iteration-level CSV (one row per coach iteration). Lazy
        # — only created when the coach calls log_iteration / log_alphazero_step.
        self.iter_csv_path = os.path.join(log_dir, 'iterations.csv')
        self.iter_csv_file = None
        self.iter_csv_writer = None
        # Per-iteration accumulators for log_alphazero_step.
        self._iter_step_accum = None  # dict[str, list[float]] | None
        self._iter_step_count = 0
        self._iter_current = None     # iter_idx currently being accumulated
        # Stage-4 W&B custom-metric registration is gated until first use.
        self._wandb_iter_axis_defined = False
        # Cumulative train-step counter across all iterations (Stage-4). Used
        # to emit a `global_step` series compatible with Stage-1's per-step
        # x-axis so both regimes can be plotted side-by-side.
        self._alphazero_global_step = 0

        # Optional TensorBoard
        self.tb_logger = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_logger = SummaryWriter(log_dir)
            except ImportError:
                print("Warning: tensorboard not available, using CSV only")

        # Optional W&B
        self.wandb_run = None
        if wandb_project is not None:
            try:
                import wandb
                wandb_output_dir = tempfile.mkdtemp()
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    group=wandb_group,
                    name=wandb_name,
                    config=wandb_config,
                    dir=wandb_output_dir,
                    mode=wandb_mode,
                    settings=wandb.Settings(
                        start_method='thread',
                        _disable_stats=False,
                    ),
                    save_code=True,
                )
                # Define x-axes so batch-level and epoch-level metrics don't conflict
                wandb.define_metric("global_step")
                wandb.define_metric("avg_cost", step_metric="global_step")
                wandb.define_metric("actor_loss", step_metric="global_step")
                wandb.define_metric("nll", step_metric="global_step")
                wandb.define_metric("grad_norm", step_metric="global_step")
                wandb.define_metric("grad_norm_clipped", step_metric="global_step")
                wandb.define_metric("value_loss", step_metric="global_step")
                wandb.define_metric("gpu_mem_peak_mb", step_metric="global_step")
                wandb.define_metric("gpu_mem_alloc_mb", step_metric="global_step")
                wandb.define_metric("gpu_mem_util_pct", step_metric="global_step")
                wandb.define_metric("epoch")
                wandb.define_metric("val_avg_cost", step_metric="epoch")
                wandb.define_metric("epoch_duration", step_metric="epoch")
                wandb.define_metric("lr", step_metric="epoch")
                wandb.define_metric("baseline_updated", step_metric="epoch")
                wandb.define_metric("val_value_r2_overall", step_metric="epoch")
                wandb.define_metric("val_value_r2_early", step_metric="epoch")
                wandb.define_metric("val_value_r2_mid", step_metric="epoch")
                wandb.define_metric("val_value_r2_late", step_metric="epoch")
                wandb.define_metric("val_value_loss", step_metric="epoch")
                wandb.define_metric("val_value_residual_mean", step_metric="epoch")
                wandb.define_metric("val_value_mean", step_metric="epoch")
                wandb.define_metric("val_target_mean", step_metric="epoch")
                # Stamp GPU info once (static for the run)
                if self._track_gpu_memory:
                    device_id = torch.cuda.current_device()
                    props = torch.cuda.get_device_properties(device_id)
                    self._gpu_total_mb = props.total_memory / (1024 * 1024)
                    self.wandb_run.summary['gpu_name'] = props.name
                    self.wandb_run.summary['gpu_mem_total_mb'] = self._gpu_total_mb
                    print(f"GPU detected: {props.name} ({self._gpu_total_mb:.0f} MB)")
                else:
                    self._gpu_total_mb = None
                print(f"W&B run initialized: {self.wandb_run.url}")
            except ImportError:
                print("Warning: wandb not installed, using CSV only")
            except Exception as e:
                print(f"Warning: wandb init failed ({e}), using CSV only")

        if not hasattr(self, '_gpu_total_mb'):
            self._gpu_total_mb = (
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                / (1024 * 1024)
            ) if self._track_gpu_memory else None

    def _gpu_mem_sample(self):
        """Sample peak + current allocated memory since the last call, then reset the peak.
        Returns (peak_mb, alloc_mb, util_pct) or (None, None, None) when disabled."""
        if not self._track_gpu_memory:
            return None, None, None
        device = torch.cuda.current_device()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats(device)
        util_pct = 100.0 * peak_mb / self._gpu_total_mb if self._gpu_total_mb else None
        return peak_mb, alloc_mb, util_pct

    def log_step(self, step, epoch, batch_id, cost, grad_norms, log_likelihood,
                 reinforce_loss, bl_loss, value_loss=None):
        avg_cost = cost.mean().item()
        grad_norms_val, grad_norms_clipped = grad_norms
        gn = grad_norms_val[0] if isinstance(grad_norms_val[0], float) else grad_norms_val[0].item()
        gnc = grad_norms_clipped[0] if isinstance(grad_norms_clipped[0], float) else grad_norms_clipped[0].item()
        nll = -log_likelihood.mean().item()
        actor_loss = reinforce_loss.item()
        vloss = value_loss.item() if value_loss is not None else ''

        gpu_peak, gpu_alloc, gpu_util = self._gpu_mem_sample()
        gpu_peak_cell = gpu_peak if gpu_peak is not None else ''
        gpu_alloc_cell = gpu_alloc if gpu_alloc is not None else ''

        # CSV
        self.csv_writer.writerow([
            step, epoch, batch_id, avg_cost, actor_loss, nll, gn, gnc, vloss,
            gpu_peak_cell, gpu_alloc_cell, time.time()
        ])
        self.csv_file.flush()

        # Console
        print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))
        print('grad_norm: {}, clipped: {}'.format(gn, gnc))
        if value_loss is not None:
            print('value_loss: {}'.format(vloss))
        if gpu_peak is not None:
            print('gpu_mem_peak: {:.0f} MB ({:.1f}% of {:.0f} MB)'.format(
                gpu_peak, gpu_util, self._gpu_total_mb))

        # TensorBoard
        if self.tb_logger is not None:
            self.tb_logger.add_scalar('avg_cost', avg_cost, step)
            self.tb_logger.add_scalar('actor_loss', actor_loss, step)
            self.tb_logger.add_scalar('nll', nll, step)
            self.tb_logger.add_scalar('grad_norm', gn, step)
            self.tb_logger.add_scalar('grad_norm_clipped', gnc, step)
            if value_loss is not None:
                self.tb_logger.add_scalar('value_loss', vloss, step)
            if gpu_peak is not None:
                self.tb_logger.add_scalar('gpu_mem_peak_mb', gpu_peak, step)
                self.tb_logger.add_scalar('gpu_mem_alloc_mb', gpu_alloc, step)
                self.tb_logger.add_scalar('gpu_mem_util_pct', gpu_util, step)

        # W&B
        if self.wandb_run is not None:
            import wandb
            payload = {
                'global_step': step,
                'avg_cost': avg_cost,
                'actor_loss': actor_loss,
                'nll': nll,
                'grad_norm': gn,
                'grad_norm_clipped': gnc,
                'epoch': epoch,
            }
            if value_loss is not None:
                payload['value_loss'] = vloss
            if gpu_peak is not None:
                payload['gpu_mem_peak_mb'] = gpu_peak
                payload['gpu_mem_alloc_mb'] = gpu_alloc
                payload['gpu_mem_util_pct'] = gpu_util
            wandb.log(payload)

    def log_epoch(self, epoch, val_avg_cost, epoch_duration, lr, baseline_updated=False,
                  value_metrics=None):
        vm = value_metrics or {}
        r2_overall = vm.get('r2_overall', '')
        r2_early = vm.get('r2_early', '')
        r2_mid = vm.get('r2_mid', '')
        r2_late = vm.get('r2_late', '')
        val_value_loss = vm.get('value_loss', '')
        val_residual_mean = vm.get('residual_mean', '')
        val_value_mean = vm.get('value_mean', '')
        val_target_mean = vm.get('target_mean', '')

        # CSV
        self.epoch_csv_writer.writerow([
            epoch, val_avg_cost, epoch_duration, lr, baseline_updated,
            r2_overall, r2_early, r2_mid, r2_late,
            val_value_loss, val_residual_mean, val_value_mean, val_target_mean
        ])
        self.epoch_csv_file.flush()

        # TensorBoard
        if self.tb_logger is not None:
            self.tb_logger.add_scalar('val_avg_cost', val_avg_cost, epoch)
            self.tb_logger.add_scalar('lr', lr, epoch)
            if value_metrics is not None:
                self.tb_logger.add_scalar('val_value_r2_overall', vm['r2_overall'], epoch)
                self.tb_logger.add_scalar('val_value_r2_early', vm['r2_early'], epoch)
                self.tb_logger.add_scalar('val_value_r2_mid', vm['r2_mid'], epoch)
                self.tb_logger.add_scalar('val_value_r2_late', vm['r2_late'], epoch)
                self.tb_logger.add_scalar('val_value_loss', vm['value_loss'], epoch)
                self.tb_logger.add_scalar('val_value_residual_mean', vm['residual_mean'], epoch)
                self.tb_logger.add_scalar('val_value_mean', vm['value_mean'], epoch)
                self.tb_logger.add_scalar('val_target_mean', vm['target_mean'], epoch)

        # W&B
        if self.wandb_run is not None:
            import wandb
            payload = {
                'epoch': epoch,
                'val_avg_cost': val_avg_cost,
                'epoch_duration': epoch_duration,
                'lr': lr,
                'baseline_updated': int(baseline_updated) if baseline_updated else 0,
            }
            if value_metrics is not None:
                payload['val_value_r2_overall'] = vm['r2_overall']
                payload['val_value_r2_early'] = vm['r2_early']
                payload['val_value_r2_mid'] = vm['r2_mid']
                payload['val_value_r2_late'] = vm['r2_late']
                payload['val_value_loss'] = vm['value_loss']
                payload['val_value_residual_mean'] = vm['residual_mean']
                payload['val_value_mean'] = vm['value_mean']
                payload['val_target_mean'] = vm['target_mean']
            wandb.log(payload)

    # ------------------------------------------------------------------
    # Stage 4 — iteration-level logging
    # ------------------------------------------------------------------

    def _ensure_iter_csv(self):
        """Open `iterations.csv` lazily and write the header row."""
        if self.iter_csv_writer is not None:
            return
        self.iter_csv_file = open(self.iter_csv_path, 'w', newline='')
        self.iter_csv_writer = csv.writer(self.iter_csv_file)
        self.iter_csv_writer.writerow([
            'iter', 'total_instances', 'val_avg_cost',
            'policy_loss_mean', 'value_loss_mean',
            'mean_entropy_pi', 'mean_entropy_policy',
            'policy_grad_norm_mean', 'value_grad_norm_mean', 'grad_norm_mean',
            'value_grad_norm_vh_mean', 'value_grad_norm_shared_mean',
            'mcts_delta_vs_greedy_mean', 'mcts_win_rate_vs_greedy',
            'greedy_cost_mean', 'mcts_cost_mean',
            'gated', 'accepted', 'mcts_wall_s', 'train_wall_s', 'buffer_size',
        ])
        self.iter_csv_file.flush()

    def _ensure_wandb_iter_axis(self):
        """Define the Stage 4 W&B step axis + sample-efficiency custom plot
        once, on first iteration logged. No-op when W&B is disabled."""
        if self._wandb_iter_axis_defined:
            return
        if self.wandb_run is None:
            self._wandb_iter_axis_defined = True
            return
        import wandb
        wandb.define_metric("iteration")
        wandb.define_metric("total_instances")
        wandb.define_metric("val_avg_cost_iter", step_metric="iteration")
        wandb.define_metric("policy_loss_mean", step_metric="iteration")
        wandb.define_metric("value_loss_mean", step_metric="iteration")
        wandb.define_metric("mean_entropy_pi", step_metric="iteration")
        wandb.define_metric("mean_entropy_policy", step_metric="iteration")
        wandb.define_metric("policy_grad_norm_mean", step_metric="iteration")
        wandb.define_metric("value_grad_norm_mean", step_metric="iteration")
        wandb.define_metric("grad_norm_mean", step_metric="iteration")
        wandb.define_metric("value_grad_norm_vh_mean", step_metric="iteration")
        wandb.define_metric("value_grad_norm_shared_mean", step_metric="iteration")
        wandb.define_metric("mcts_delta_vs_greedy_mean", step_metric="iteration")
        wandb.define_metric("mcts_win_rate_vs_greedy", step_metric="iteration")
        wandb.define_metric("greedy_cost_mean", step_metric="iteration")
        wandb.define_metric("mcts_cost_mean", step_metric="iteration")
        wandb.define_metric("policy_grad_norm_step", step_metric="alphazero_train_step")
        wandb.define_metric("value_grad_norm_step", step_metric="alphazero_train_step")
        wandb.define_metric("value_grad_norm_vh_step", step_metric="alphazero_train_step")
        wandb.define_metric("value_grad_norm_shared_step", step_metric="alphazero_train_step")
        wandb.define_metric("gated", step_metric="iteration")
        wandb.define_metric("accepted", step_metric="iteration")
        wandb.define_metric("mcts_wall_s", step_metric="iteration")
        wandb.define_metric("train_wall_s", step_metric="iteration")
        wandb.define_metric("buffer_size", step_metric="iteration")
        # Sample-efficiency custom plot: val cost vs. instances seen.
        wandb.define_metric("val_avg_cost_vs_instances", step_metric="total_instances")
        self._wandb_iter_axis_defined = True

    def log_alphazero_step(self, metrics, iter_idx, step):
        """Stage 4 per-train-step logger. Accumulates into the current
        iteration's running mean buckets so `log_iteration` can flush them.

        `metrics` is the dict returned by `train_step_alphazero`. Required
        keys: policy_loss, value_loss, total_loss, mean_entropy_pi.
        """
        # Reset accumulators if we've crossed an iteration boundary.
        if self._iter_current != iter_idx:
            self._iter_step_accum = {
                'policy_loss': [],
                'value_loss': [],
                'total_loss': [],
                'mean_entropy_pi': [],
                'mean_entropy_policy': [],
                'policy_grad_norm': [],
                'value_grad_norm': [],
                'gradient_norm': [],
                'value_grad_norm_vh': [],
                'value_grad_norm_shared': [],
            }
            self._iter_step_count = 0
            self._iter_current = iter_idx
        for k in self._iter_step_accum:
            if k in metrics:
                self._iter_step_accum[k].append(float(metrics[k]))
        self._iter_step_count += 1

        # W&B per-step. Stage-4-specific names use the `iteration` axis;
        # Stage-1-aligned aliases (`global_step`, `value_loss`) use the
        # `global_step` axis so a Stage-4 per-step run can be plotted in the
        # same W&B panel as a Stage-1 per-step run.
        if self.wandb_run is not None:
            self._ensure_wandb_iter_axis()
            import wandb
            self._alphazero_global_step += 1
            payload = {
                'iteration': iter_idx,
                'alphazero_train_step': step,
                'policy_loss_step': float(metrics.get('policy_loss', 0.0)),
                'value_loss_step': float(metrics.get('value_loss', 0.0)),
                'total_loss_step': float(metrics.get('total_loss', 0.0)),
                'mean_entropy_pi_step': float(metrics.get('mean_entropy_pi', 0.0)),
                'mean_entropy_policy_step': float(metrics.get('mean_entropy_policy', 0.0)),
                'policy_grad_norm_step': float(metrics.get('policy_grad_norm', 0.0)),
                'value_grad_norm_step': float(metrics.get('value_grad_norm', 0.0)),
                'grad_norm_step': float(metrics.get('gradient_norm', 0.0)),
                'value_grad_norm_vh_step': float(metrics.get('value_grad_norm_vh', 0.0)),
                'value_grad_norm_shared_step': float(metrics.get('value_grad_norm_shared', 0.0)),
                # Stage-1-aligned aliases (kept on the `global_step` axis):
                'global_step': int(self._alphazero_global_step),
                'value_loss': float(metrics.get('value_loss', 0.0)),
                'epoch': int(iter_idx),
            }
            wandb.log(payload)

    def log_iteration(self, iter, total_instances, val_avg_cost,
                      gated=False, accepted=None,
                      mcts_wall_s=0.0, train_wall_s=0.0,
                      buffer_size=0, lr=None,
                      mcts_delta_vs_greedy_mean=None,
                      mcts_win_rate_vs_greedy=None,
                      greedy_cost_mean=None,
                      mcts_cost_mean=None):
        """Stage 4 per-iteration logger. Flushes per-step running means from
        `log_alphazero_step` into one CSV row + one W&B point.

        Args mirror plan §D.2 verbatim. `accepted` may be `None` when the
        iteration was not gated (wrote as empty string in CSV).

        Stage-1-aligned W&B aliases: this function ALSO emits the Stage-1
        per-epoch metric names (`epoch`, `val_avg_cost`, `lr`,
        `epoch_duration`, `baseline_updated`) keyed off `epoch=iter` so
        Stage 1 and Stage 4 W&B runs can be compared side-by-side under a
        common `epoch` x-axis. `lr` is optional; if omitted the alias is
        skipped.
        """
        self._ensure_iter_csv()
        self._ensure_wandb_iter_axis()

        # Flush per-step accumulators into running means.
        def _mean(key):
            vals = (self._iter_step_accum or {}).get(key, [])
            return sum(vals) / max(1, len(vals)) if vals else float('nan')

        if self._iter_current == iter and self._iter_step_count > 0:
            policy_loss_mean = _mean('policy_loss')
            value_loss_mean = _mean('value_loss')
            mean_entropy_pi = _mean('mean_entropy_pi')
            mean_entropy_policy = _mean('mean_entropy_policy')
            policy_grad_norm_mean = _mean('policy_grad_norm')
            value_grad_norm_mean = _mean('value_grad_norm')
            grad_norm_mean = _mean('gradient_norm')
            value_grad_norm_vh_mean = _mean('value_grad_norm_vh')
            value_grad_norm_shared_mean = _mean('value_grad_norm_shared')
        else:
            policy_loss_mean = float('nan')
            value_loss_mean = float('nan')
            mean_entropy_pi = float('nan')
            mean_entropy_policy = float('nan')
            policy_grad_norm_mean = float('nan')
            value_grad_norm_mean = float('nan')
            grad_norm_mean = float('nan')
            value_grad_norm_vh_mean = float('nan')
            value_grad_norm_shared_mean = float('nan')

        val_cell = float(val_avg_cost) if val_avg_cost is not None else ''
        gated_cell = int(bool(gated))
        if accepted is None:
            accepted_cell = ''
        else:
            accepted_cell = int(bool(accepted))
        mcts_delta_cell = (
            float(mcts_delta_vs_greedy_mean)
            if mcts_delta_vs_greedy_mean is not None else ''
        )
        mcts_win_cell = (
            float(mcts_win_rate_vs_greedy)
            if mcts_win_rate_vs_greedy is not None else ''
        )
        greedy_cost_cell = (
            float(greedy_cost_mean) if greedy_cost_mean is not None else ''
        )
        mcts_cost_cell = (
            float(mcts_cost_mean) if mcts_cost_mean is not None else ''
        )

        # CSV
        self.iter_csv_writer.writerow([
            iter, int(total_instances), val_cell,
            policy_loss_mean, value_loss_mean,
            mean_entropy_pi, mean_entropy_policy,
            policy_grad_norm_mean, value_grad_norm_mean, grad_norm_mean,
            value_grad_norm_vh_mean, value_grad_norm_shared_mean,
            mcts_delta_cell, mcts_win_cell, greedy_cost_cell, mcts_cost_cell,
            gated_cell, accepted_cell,
            float(mcts_wall_s), float(train_wall_s), int(buffer_size),
        ])
        self.iter_csv_file.flush()

        # Console
        print(
            'iter={} total_instances={} val_avg_cost={} '
            'policy_loss={:.4f} value_loss={:.4f} entropy={:.4f} '
            'pg_norm={:.4f} vg_norm={:.4f} '
            'mcts_delta={} mcts_win={} '
            'gated={} accepted={} mcts_s={:.2f} train_s={:.2f} buf={}'.format(
                iter, int(total_instances),
                f'{val_avg_cost:.6f}' if val_avg_cost is not None else 'NA',
                policy_loss_mean, value_loss_mean, mean_entropy_pi,
                policy_grad_norm_mean, value_grad_norm_mean,
                f'{mcts_delta_vs_greedy_mean:.6f}' if mcts_delta_vs_greedy_mean is not None else 'NA',
                f'{mcts_win_rate_vs_greedy:.3f}' if mcts_win_rate_vs_greedy is not None else 'NA',
                gated_cell, accepted_cell, mcts_wall_s, train_wall_s, buffer_size,
            )
        )

        # TensorBoard
        if self.tb_logger is not None:
            if val_avg_cost is not None:
                self.tb_logger.add_scalar('val_avg_cost_iter', float(val_avg_cost), iter)
            self.tb_logger.add_scalar('policy_loss_mean', policy_loss_mean, iter)
            self.tb_logger.add_scalar('value_loss_mean', value_loss_mean, iter)
            self.tb_logger.add_scalar('mean_entropy_pi', mean_entropy_pi, iter)
            self.tb_logger.add_scalar('mean_entropy_policy', mean_entropy_policy, iter)
            self.tb_logger.add_scalar('policy_grad_norm_mean', policy_grad_norm_mean, iter)
            self.tb_logger.add_scalar('value_grad_norm_mean', value_grad_norm_mean, iter)
            self.tb_logger.add_scalar('grad_norm_mean', grad_norm_mean, iter)
            self.tb_logger.add_scalar('value_grad_norm_vh_mean', value_grad_norm_vh_mean, iter)
            self.tb_logger.add_scalar('value_grad_norm_shared_mean', value_grad_norm_shared_mean, iter)
            self.tb_logger.add_scalar('mcts_wall_s', float(mcts_wall_s), iter)
            self.tb_logger.add_scalar('train_wall_s', float(train_wall_s), iter)
            self.tb_logger.add_scalar('buffer_size', int(buffer_size), iter)
            if mcts_delta_vs_greedy_mean is not None:
                self.tb_logger.add_scalar(
                    'mcts_delta_vs_greedy_mean',
                    float(mcts_delta_vs_greedy_mean),
                    iter,
                )
            if mcts_win_rate_vs_greedy is not None:
                self.tb_logger.add_scalar(
                    'mcts_win_rate_vs_greedy',
                    float(mcts_win_rate_vs_greedy),
                    iter,
                )

        # W&B
        if self.wandb_run is not None:
            import wandb
            payload = {
                'iteration': iter,
                'total_instances': int(total_instances),
                'policy_loss_mean': policy_loss_mean,
                'value_loss_mean': value_loss_mean,
                'mean_entropy_pi': mean_entropy_pi,
                'mean_entropy_policy': mean_entropy_policy,
                'policy_grad_norm_mean': policy_grad_norm_mean,
                'value_grad_norm_mean': value_grad_norm_mean,
                'grad_norm_mean': grad_norm_mean,
                'value_grad_norm_vh_mean': value_grad_norm_vh_mean,
                'value_grad_norm_shared_mean': value_grad_norm_shared_mean,
                'gated': gated_cell,
                'mcts_wall_s': float(mcts_wall_s),
                'train_wall_s': float(train_wall_s),
                'buffer_size': int(buffer_size),
            }
            if mcts_delta_vs_greedy_mean is not None:
                payload['mcts_delta_vs_greedy_mean'] = float(mcts_delta_vs_greedy_mean)
            if mcts_win_rate_vs_greedy is not None:
                payload['mcts_win_rate_vs_greedy'] = float(mcts_win_rate_vs_greedy)
            if greedy_cost_mean is not None:
                payload['greedy_cost_mean'] = float(greedy_cost_mean)
            if mcts_cost_mean is not None:
                payload['mcts_cost_mean'] = float(mcts_cost_mean)
            if val_avg_cost is not None:
                payload['val_avg_cost_iter'] = float(val_avg_cost)
                # Sample-efficiency series: x = total_instances, y = val cost.
                payload['val_avg_cost_vs_instances'] = float(val_avg_cost)
            if accepted is not None:
                payload['accepted'] = int(bool(accepted))

            # Stage-1-aligned aliases: emit the per-epoch Stage-1 metric names
            # keyed off `epoch = iter` so Stage 1 and Stage 4 W&B runs share
            # the `epoch` x-axis. The Stage-4-specific names above are kept
            # so existing Stage-4 dashboards/plots are unaffected.
            payload['epoch'] = int(iter)
            payload['epoch_duration'] = float(mcts_wall_s) + float(train_wall_s)
            payload['baseline_updated'] = int(bool(accepted)) if accepted else 0
            if val_avg_cost is not None:
                payload['val_avg_cost'] = float(val_avg_cost)
            if lr is not None:
                payload['lr'] = float(lr)

            wandb.log(payload)

    def close(self):
        self.csv_file.close()
        self.epoch_csv_file.close()
        if self.iter_csv_file is not None:
            self.iter_csv_file.close()
        if self.tb_logger is not None:
            self.tb_logger.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
