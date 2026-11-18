"""The retained two-stage training protocol, without filesystem side effects.

Call ``seed_everything`` BEFORE constructing the data bundle and model, then call
``train``. The trainer deliberately does not reseed an already initialized model.
``training.head_lr`` is the reference PRETRAINING learning rate; without an
explicit ``head_lr_mult``, joint training uses the backbone's learning rate.
"""

from copy import deepcopy
import random

import numpy as np
import torch
from torch.nn import functional as F

from .backbones.factory import _freeze_running_stats


PRETRAIN_EPOCHS = 10
REFERENCE_RESTART_EPOCH = 5
JOINT_START_EPOCH = 2


def seed_everything(seed):
    """Match the legacy RNG setup, without changing determinism/backend flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state(device):
    return (
        torch.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
        torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    )


def _restore_rng(state):
    cpu, python, numpy, cuda = state
    torch.set_rng_state(cpu)
    random.setstate(python)
    np.random.set_state(numpy)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _snapshot(module):
    state = module.state_dict()
    for name, value in state.items():
        state[name] = value.detach().cpu().clone()
    return state


def _batch(batch, device):
    if len(batch) == 5:
        x, y, _, x_mark, y_mark = batch
    elif len(batch) == 4:
        x, y, x_mark, y_mark = batch
    else:
        raise ValueError("Expected (x, y, [original_y,] x_mark, y_mark)")
    return tuple(value.to(device).float() for value in (x, y, x_mark, y_mark))


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _stop(settings, stale, epoch):
    return (
        settings["early_stop"]
        and stale >= settings["patience"]
        and epoch + 1 >= settings["min_epochs"]
    )


@torch.no_grad()
def evaluate(model, loader, device=None):
    """Element-weighted float32 MSE/MAE, but batch-mean auxiliary losses.

    Retain the incomplete test batch and the loader's iteration RNG draw. Do not
    replace this reduction with a mean of batch MSEs or CPU/float64 accumulation.
    The normalizer must return unregularized reference MSE in eval/no-grad mode.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    squared_error = torch.zeros(1, device=device)
    absolute_error = torch.zeros(1, device=device)
    count = torch.tensor(0, device=device)
    tasks, references = [], []
    for batch in loader:
        x, y, x_mark, y_mark = _batch(batch, device)
        prediction = model(x, x_mark, y_mark)
        tasks.append(float(F.mse_loss(prediction, y)))
        references.append(float(model.normalizer.reference_loss(y)))
        prediction, y = prediction.contiguous(), y.contiguous()
        error = prediction.view(-1) - y.view(-1)
        squared_error += torch.sum(error * error, dim=0)
        absolute_error += torch.sum(torch.abs(error), dim=0)
        count += y.numel()
    if not tasks:
        raise ValueError("Cannot evaluate an empty loader")
    return {
        "mse": float(squared_error / count),
        "mae": float(absolute_error / count),
        "task_loss": _mean(tasks),
        "reference_loss": _mean(references),
    }


def _pretrain_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        x, y, x_mark, y_mark = _batch(batch, device)
        optimizer.zero_grad()
        # The full frozen backbone forward is required for dropout RNG parity.
        model(x, x_mark, y_mark)
        loss = model.normalizer.reference_loss(y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"reference_loss": _mean(losses)}


def _stage2_epoch(model, loader, optimizer, settings, device):
    model.train()
    normalizer = model.normalizer
    head = [parameter for parameter in normalizer.parameters() if parameter.requires_grad]
    if head and "head_lr_mult" in settings:
        optimizer.param_groups[1]["lr"] = (
            settings["head_lr_mult"] * settings["head_lr"]
            * optimizer.param_groups[0]["lr"] / settings["lr"]
        )
    rates = [group["lr"] for group in optimizer.param_groups]
    mean_weight, reference_weight = (0.0, 1.0) if head else (10.0, 0.0)
    tasks, references, means, diagnostic_references = [], [], [], []
    for batch in loader:
        x, y, x_mark, y_mark = _batch(batch, device)
        optimizer.zero_grad()
        prediction = model(x, x_mark, y_mark)
        task = F.mse_loss(prediction, y)
        # Use the live BCT cache: never resample the stochastic reference head.
        anchor = normalizer.reference_prediction.transpose(1, 2)
        remainder = prediction - anchor
        mean_energy = normalizer.project(remainder.transpose(1, 2)).square().mean()
        reference = normalizer.reference_loss(y) if reference_weight else task.new_zeros(())
        (task + mean_weight * mean_energy + reference_weight * reference).backward()
        # Legacy diagnostics call the loss again before the optimizer update.
        with torch.no_grad():
            diagnostic_references.append(float(normalizer.reference_loss(y)))
        optimizer.step()
        tasks.append(float(task.detach()))
        references.append(float(reference.detach()))
        means.append(float(mean_energy.detach()))
    return {
        "task_loss": _mean(tasks),
        "reference_loss": _mean(references),
        "diagnostic_reference_loss": _mean(diagnostic_references),
        "mean_energy": _mean(means),
        "mean_weight": mean_weight,
        "reference_weight": reference_weight,
        "lr": rates[0],
        "head_lr": rates[1] if head else None,
    }


def log_epoch(record):
    """An optional compact, RNG-neutral ``on_epoch`` callback."""
    if record["stage"] == "pretrain":
        text = (
            f"pretrain {record['epoch']:3d} lr={record['lr']:.6g} "
            f"train_ref={record['train']['reference_loss']:.8g} "
            f"val_ref={record['val']['reference_loss']:.8g}"
        )
    else:
        text = (
            f"stage2 {record['epoch']:3d} lr={record['train']['lr']:.6g} "
            f"train_mse={record['train']['task_loss']:.8g} "
            f"val_mse={record['val']['mse']:.8g} val_mae={record['val']['mae']:.8g} "
            f"test_mse={record['test']['mse']:.8g} test_mae={record['test']['mae']:.8g}"
        )
    print(text + (" *" if record["improved"] else ""), flush=True)


def train(model, data, config, *, device=None, on_epoch=None):
    """Fit an initialized model; return selected CPU weights, metrics and history.

    No files are written. The caller owns model/data construction, seeding before
    construction, and artifact export. Data loaders must already implement the
    historical shuffling, drop-last, worker, split and scaling conventions. Pass
    ``on_epoch=log_epoch`` for compact progress. Custom callbacks must not consume
    RNG, iterate loaders or change the model if exact replay is required.
    """
    settings = config["training"]
    if settings["epochs"] < 1:
        raise ValueError("At least one stage-two epoch is required")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(device)
    # Factory-built backbones already have this hook. Custom test models may not.
    hook = None
    if _freeze_running_stats not in model.backbone._forward_pre_hooks.values():
        hook = model.backbone.register_forward_pre_hook(_freeze_running_stats)
    try:
        return _train(model, data, settings, device, on_epoch)
    finally:
        if hook is not None:
            hook.remove()


def _train(model, data, settings, device, on_epoch=None):
    normalizer = model.normalizer
    model.backbone.requires_grad_(False)
    normalizer.requires_grad_(True)
    optimizer = torch.optim.Adam(normalizer.parameters(), lr=settings["head_lr"])
    initial_reference = _snapshot(normalizer)
    initial_rng = _rng_state(device)
    best_reference, best_pretrain_epoch, stale = float("inf"), 0, 0
    pretrain_state = None
    pretrain_history = []

    for epoch in range(PRETRAIN_EPOCHS):
        if epoch == REFERENCE_RESTART_EPOCH:
            normalizer.load_state_dict(initial_reference)
            optimizer.state.clear()
            _restore_rng(initial_rng)
        released = epoch >= REFERENCE_RESTART_EPOCH
        normalizer.feature_gain_raw.requires_grad_(released)
        if not released:
            normalizer.feature_gain_raw.grad = None
        if released:
            for group in optimizer.param_groups:
                group["lr"] = settings["head_lr"] * settings["pretrain_lr_decay"] ** max(
                    epoch - REFERENCE_RESTART_EPOCH - 1, 0
                )
        rate = optimizer.param_groups[0]["lr"]
        training = _pretrain_epoch(model, data.train_loader, optimizer, device)
        validation = evaluate(model, data.val_loader, device)
        improved = validation["reference_loss"] < best_reference - settings["min_delta"]
        if improved:
            best_reference = validation["reference_loss"]
            best_pretrain_epoch = epoch + 1
            stale = 0
            pretrain_state = _snapshot(model)
        else:
            stale += 1
        pretrain_history.append({
            "stage": "pretrain",
            "epoch": epoch + 1, "lr": rate, "gain_trainable": released,
            "train": training, "val": validation, "improved": improved,
        })
        for group in optimizer.param_groups:
            group["lr"] = settings["head_lr"] * settings["pretrain_lr_decay"] ** epoch
        if on_epoch is not None:
            on_epoch(deepcopy(pretrain_history[-1]))
        if _stop(settings, stale, epoch):
            break

    if pretrain_state is not None:
        model.load_state_dict(pretrain_state)
    normalizer.requires_grad_(False)
    model.backbone.requires_grad_(True)
    optimizer = torch.optim.Adam([{
        "name": "backbone", "params": list(model.backbone.parameters()),
        "lr": settings["lr"], "weight_decay": 0.0,
    }])
    best_val, best_epoch, stale = float("inf"), 0, 0
    selected_state = None
    history = []

    for epoch in range(settings["epochs"]):
        if epoch == JOINT_START_EPOCH:
            normalizer.requires_grad_(True)
            optimizer.add_param_group({
                "params": list(normalizer.parameters()),
                "lr": optimizer.param_groups[0]["lr"], "weight_decay": 0.0,
            })
        training = _stage2_epoch(model, data.train_loader, optimizer, settings, device)
        validation = evaluate(model, data.val_loader, device)
        # Always evaluate test here: even an ordered loader advances the RNG.
        test = evaluate(model, data.test_loader, device)
        improved = validation["mse"] < best_val - settings["min_delta"]
        if improved:
            best_val, best_epoch, stale = validation["mse"], epoch + 1, 0
            selected_state = _snapshot(model)
        else:
            stale += 1
        rate = settings["lr"] * settings["lr_decay"] ** epoch
        for group in optimizer.param_groups:
            group["lr"] = rate
        history.append({
            "stage": "stage2",
            "epoch": epoch + 1, "train": training, "val": validation,
            "test": test, "improved": improved, "next_lr": rate,
        })
        if on_epoch is not None:
            on_epoch(deepcopy(history[-1]))
        if _stop(settings, stale, epoch):
            break

    if selected_state is None:
        raise RuntimeError("No finite validation winner was available for export")
    model.load_state_dict(selected_state)
    metrics = {
        "val": evaluate(model, data.val_loader, device),
        "test": evaluate(model, data.test_loader, device),
    }
    return {
        "state_dict": selected_state,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "best_pretrain_epoch": best_pretrain_epoch,
        "best_reference_mse": best_reference,
        "metrics": metrics,
        "history": history,
        "pretrain_history": pretrain_history,
    }
