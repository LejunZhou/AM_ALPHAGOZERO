import os
import time
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from am_baseline.model.attention_model import set_decode_type
from am_baseline.utils.tensor_ops import move_to


def validate(model, dataset, opts):
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost


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
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

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

    # Baseline epoch callback
    baseline_updated = baseline.epoch_callback(model, epoch)

    # Log epoch metrics
    if logger is not None:
        logger.log_epoch(epoch, avg_reward.item(), epoch_duration,
                         optimizer.param_groups[0]['lr'],
                         baseline_updated=baseline_updated)

    # LR scheduler
    lr_scheduler.step()


def train_batch(model, optimizer, baseline, epoch, batch_id, step, batch, logger, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Forward pass
    cost, log_likelihood = model(x)

    # Baseline
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # REINFORCE loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Backward
    optimizer.zero_grad()
    loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0 and logger is not None:
        logger.log_step(step, epoch, batch_id, cost, grad_norms,
                        log_likelihood, reinforce_loss, bl_loss)
