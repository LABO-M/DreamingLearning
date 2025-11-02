
import os
import math
import random
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local modules
from utils.models import LSTMWindowModel
from utils.trainer_datawindow import DreamingTrainer

# Optional: Optuna
try:
    import optuna
except Exception as e:
    optuna = None
    print("Warning: Optuna is not installed. Install with `pip install optuna`.")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_trainer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    dream_weight: float,
    dream_noise_std: float,
    use_huber: bool,
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
) -> DreamingTrainer:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = DreamingTrainer(
        model=model,
        optimizer=opt,
        task_type="regression",
        temperature=temperature,
        dream_weight=dream_weight,
        dream_noise_std=dream_noise_std,
        grad_clip=1.0,
        mixed_precision=False,
    )
    if use_huber:
        try:
            trainer.criterion = nn.SmoothL1Loss(beta=1.0)  # Huber-like
        except TypeError:
            trainer.criterion = nn.HuberLoss(delta=1.0)
    return trainer

def objective_factory(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_size: int,
    num_labels: int,
    label_width: int,
    device: Optional[torch.device],
    use_huber: bool,
    max_epochs: int = 20,
    patience: int = 5,
    seed: int = 42,
):
    def objective(trial) -> float:
        set_seed(seed + trial.number)

        hidden_size  = trial.suggest_categorical("hidden_size", [128, 256, 384, 512])
        num_layers   = trial.suggest_int("num_layers", 1, 5, step=1)
        dropout      = trial.suggest_float("dropout", 0.0, 0.5)
        bidirectional= trial.suggest_categorical("bidirectional", [False, True])
        mlp_hidden   = trial.suggest_categorical("mlp_hidden", [None, 256, 512])

        lr           = trial.suggest_float("lr", 1e-3, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
        temperature  = trial.suggest_float("temperature", 0.5, 2.0)

        dream_weight_late = trial.suggest_float("dream_weight_late", 0.0, 0.4)
        dream_noise_std   = trial.suggest_float("dream_noise_std", 0.0, 0.05)

        model = LSTMWindowModel(
            input_size=input_size,
            label_width=label_width,
            num_labels=num_labels,
            task_type="regression",
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            mlp_hidden=mlp_hidden,
        )
        device_ = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device_)

        trainer = make_trainer(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            temperature=temperature,
            dream_weight=0.0,
            dream_noise_std=0.0,
            use_huber=use_huber,
            device=device_,
        )

        best_val = math.inf
        wait = 0

        for epoch in range(1, max_epochs + 1):
            if epoch > max_epochs // 2:
                trainer.dream_weight = dream_weight_late
                trainer.dream_noise_std = dream_noise_std

            trainer.train_epoch(train_loader, val_loader, epoch=epoch, log_interval=200)
            val_loss = trainer.evaluate(val_loader, split_name="val")["val_loss"]

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                wait = 0
            else:
                wait += 1
            if optuna is not None and hasattr(trial, "should_prune") and trial.should_prune():
                raise optuna.TrialPruned()
            if wait >= patience:
                break

        return float(best_val)

    return objective

def run_tuning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_size: int,
    num_labels: int,
    label_width: int = 1,
    n_trials: int = 20,
    device: Optional[torch.device] = None,
    use_huber: bool = False,
    seed: int = 42,
    max_epochs: int = 20,
    patience: int = 5,
):
    if optuna is None:
        raise ImportError("Optuna is not installed. Please `pip install optuna`.")

    study = optuna.create_study(direction="minimize")
    objective = objective_factory(
        train_loader, val_loader, input_size, num_labels, label_width, device, use_huber,
        max_epochs=max_epochs, patience=patience, seed=seed
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params

    # Retrain best model
    hidden_size   = best_params.get("hidden_size", 256)
    num_layers    = best_params.get("num_layers", 2)
    dropout       = best_params.get("dropout", 0.1)
    bidirectional = best_params.get("bidirectional", False)
    mlp_hidden    = best_params.get("mlp_hidden", 512)
    lr            = best_params.get("lr", 2e-3)
    weight_decay  = best_params.get("weight_decay", 1e-5)
    temperature   = best_params.get("temperature", 1.0)
    dream_weight_late = best_params.get("dream_weight_late", 0.2)
    dream_noise_std   = best_params.get("dream_noise_std", 0.01)

    model = LSTMWindowModel(
        input_size=input_size,
        label_width=label_width,
        num_labels=num_labels,
        task_type="regression",
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        mlp_hidden=mlp_hidden,
    )
    device_ = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device_)

    trainer = make_trainer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        temperature=temperature,
        dream_weight=0.0,
        dream_noise_std=0.0,
        use_huber=use_huber,
        device=device_
    )

    best_val = math.inf
    wait = 0
    best_state = None
    for epoch in range(1, max_epochs + 1):
        if epoch > max_epochs // 2:
            trainer.dream_weight = dream_weight_late
            trainer.dream_noise_std = dream_noise_std

        trainer.train_epoch(train_loader, val_loader, epoch=epoch, log_interval=200)
        val_loss = trainer.evaluate(val_loader, split_name="val")["val_loss"]

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device_)
    return model, best_params
