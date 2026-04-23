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

    def close(self):
        self.csv_file.close()
        self.epoch_csv_file.close()
        if self.tb_logger is not None:
            self.tb_logger.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
