import os
import csv
import time
import tempfile


class MetricsLogger:
    """CSV-based metrics logger with optional TensorBoard and W&B support."""

    def __init__(self, log_dir, use_tensorboard=False,
                 wandb_project=None, wandb_entity=None, wandb_group=None,
                 wandb_name=None, wandb_mode='online', wandb_config=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # CSV logger (always active)
        self.csv_path = os.path.join(log_dir, 'metrics.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'step', 'epoch', 'batch_id', 'avg_cost', 'actor_loss', 'nll',
            'grad_norm', 'grad_norm_clipped', 'time'
        ])

        # Epoch-level CSV
        self.epoch_csv_path = os.path.join(log_dir, 'epochs.csv')
        self.epoch_csv_file = open(self.epoch_csv_path, 'w', newline='')
        self.epoch_csv_writer = csv.writer(self.epoch_csv_file)
        self.epoch_csv_writer.writerow([
            'epoch', 'val_avg_cost', 'epoch_duration', 'lr', 'baseline_updated'
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
                wandb.define_metric("epoch")
                wandb.define_metric("val_avg_cost", step_metric="epoch")
                wandb.define_metric("epoch_duration", step_metric="epoch")
                wandb.define_metric("lr", step_metric="epoch")
                wandb.define_metric("baseline_updated", step_metric="epoch")
                print(f"W&B run initialized: {self.wandb_run.url}")
            except ImportError:
                print("Warning: wandb not installed, using CSV only")
            except Exception as e:
                print(f"Warning: wandb init failed ({e}), using CSV only")

    def log_step(self, step, epoch, batch_id, cost, grad_norms, log_likelihood,
                 reinforce_loss, bl_loss):
        avg_cost = cost.mean().item()
        grad_norms_val, grad_norms_clipped = grad_norms
        gn = grad_norms_val[0] if isinstance(grad_norms_val[0], float) else grad_norms_val[0].item()
        gnc = grad_norms_clipped[0] if isinstance(grad_norms_clipped[0], float) else grad_norms_clipped[0].item()
        nll = -log_likelihood.mean().item()
        actor_loss = reinforce_loss.item()

        # CSV
        self.csv_writer.writerow([
            step, epoch, batch_id, avg_cost, actor_loss, nll, gn, gnc, time.time()
        ])
        self.csv_file.flush()

        # Console
        print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))
        print('grad_norm: {}, clipped: {}'.format(gn, gnc))

        # TensorBoard
        if self.tb_logger is not None:
            self.tb_logger.add_scalar('avg_cost', avg_cost, step)
            self.tb_logger.add_scalar('actor_loss', actor_loss, step)
            self.tb_logger.add_scalar('nll', nll, step)
            self.tb_logger.add_scalar('grad_norm', gn, step)
            self.tb_logger.add_scalar('grad_norm_clipped', gnc, step)

        # W&B
        if self.wandb_run is not None:
            import wandb
            wandb.log({
                'global_step': step,
                'avg_cost': avg_cost,
                'actor_loss': actor_loss,
                'nll': nll,
                'grad_norm': gn,
                'grad_norm_clipped': gnc,
                'epoch': epoch,
            })

    def log_epoch(self, epoch, val_avg_cost, epoch_duration, lr, baseline_updated=False):
        # CSV
        self.epoch_csv_writer.writerow([
            epoch, val_avg_cost, epoch_duration, lr, baseline_updated
        ])
        self.epoch_csv_file.flush()

        # TensorBoard
        if self.tb_logger is not None:
            self.tb_logger.add_scalar('val_avg_cost', val_avg_cost, epoch)
            self.tb_logger.add_scalar('lr', lr, epoch)

        # W&B
        if self.wandb_run is not None:
            import wandb
            wandb.log({
                'epoch': epoch,
                'val_avg_cost': val_avg_cost,
                'epoch_duration': epoch_duration,
                'lr': lr,
                'baseline_updated': int(baseline_updated) if baseline_updated else 0,
            })

    def close(self):
        self.csv_file.close()
        self.epoch_csv_file.close()
        if self.tb_logger is not None:
            self.tb_logger.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
