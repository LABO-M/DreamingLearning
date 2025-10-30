
from typing import Optional, Dict, Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

Device = torch.device

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.cnt += n
    @property
    def avg(self) -> float:
        return self.sum / max(1, self.cnt)

def _maybe_quantize_to_bins(y: torch.Tensor, bins: Optional[torch.Tensor]) -> torch.Tensor:
    if bins is None:
        raise ValueError("Classification task requires integer labels or `bins` for quantization.")
    return torch.bucketize(y, bins)

def _sample_categorical_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    B, L, D, C = probs.shape
    samples = torch.multinomial(probs.view(-1, C), num_samples=1).view(B, L, D)
    return samples

def _ensure_label_dtype(labels: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "classification":
        return labels.long()
    else:
        return labels.float()

class DreamingTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        task_type: Literal["classification", "regression"],
        device: Optional[Device] = None,
        num_classes: Optional[int] = None,
        bins: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        dream_weight: float = 1.0,
        dream_noise_std: float = 0.0,
        grad_clip: Optional[float] = 1.0,
        mixed_precision: bool = False,
        label_smoothing: float = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.task_type = task_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.num_classes = num_classes
        self.bins = bins.to(self.device) if bins is not None else None

        self.temperature = float(temperature)
        self.dream_weight = float(dream_weight)
        self.dream_noise_std = float(dream_noise_std)

        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        try:
            self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        except Exception:
            # Fallback for PyTorch versions that deprecate cuda.amp.GradScaler init
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda', enabled=mixed_precision)

        if task_type == "classification":
            if self.num_classes is None:
                raise ValueError("num_classes is required for classification.")
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.MSELoss()

    def _compute_supervised_loss(self, logits_or_preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            B, L, D, C = logits_or_preds.shape
            loss = self.criterion(
                logits_or_preds.view(B * L * D, C),
                labels.view(B * L * D),
            )
            return loss
        else:
            return self.criterion(logits_or_preds, labels)

    @torch.no_grad()
    def _make_dream_labels(self, logits_or_preds: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return _sample_categorical_from_logits(logits_or_preds, self.temperature)
        else:
            noise = 0.0
            if self.dream_noise_std > 0:
                noise = torch.randn_like(logits_or_preds) * self.dream_noise_std
            return (logits_or_preds + noise).detach()

    def _forward_model(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        if self.task_type == "classification":
            if outputs.dim() != 4:
                raise RuntimeError(f"Model should return (B,L,D,C) for classification, got {outputs.shape}")
        else:
            if outputs.dim() != 3:
                raise RuntimeError(f"Model should return (B,L,D) for regression, got {outputs.shape}")
        return outputs

    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            if labels.dtype not in (torch.long, torch.int64):
                if self.bins is None:
                    raise ValueError("Float labels given but `bins` not provided for classification.")
                labels = _maybe_quantize_to_bins(labels, self.bins)
        return _ensure_label_dtype(labels, self.task_type)

    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epoch: int = 1,
        log_interval: int = 50,
    ) -> Dict[str, Any]:
        self.model.train()
        loss_meter = AverageMeter()

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(self.device)
            y = y.to(self.device)
            y_sup = self._prepare_labels(y)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                logits_or_preds = self._forward_model(x)
                sup_loss = self._compute_supervised_loss(logits_or_preds, y_sup)

                y_dream = self._make_dream_labels(logits_or_preds.detach())
                logits_or_preds_2 = self._forward_model(x)
                dream_loss = self._compute_supervised_loss(logits_or_preds_2, _ensure_label_dtype(y_dream, self.task_type))

                total_loss = sup_loss + self.dream_weight * dream_loss

            self.scaler.scale(total_loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = x.size(0)
            loss_meter.update(total_loss.item(), batch_size)

            if step % log_interval == 0:
                print(f"[Epoch {epoch}] Step {step}/{len(train_loader)}  loss={loss_meter.avg:.4f}")

        metrics = {"train_loss": loss_meter.avg}
        if val_loader is not None:
            metrics.update(self.evaluate(val_loader, split_name="val"))
        return metrics

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, split_name: str = "test") -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()

        for x, y in data_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_sup = self._prepare_labels(y)

            outputs = self._forward_model(x)
            loss = self._compute_supervised_loss(outputs, y_sup)
            loss_meter.update(loss.item(), x.size(0))

        return {f"{split_name}_loss": loss_meter.avg}

    @torch.no_grad()
    def predict(
        self,
        data_loader: DataLoader,
        mode: Literal["full_horizon", "single_step"] = "full_horizon",
        temperature: Optional[float] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        self.model.eval()
        outs = []
        use_T = self.temperature if temperature is None else float(temperature)

        for x, _ in data_loader:
            x = x.to(self.device)
            outputs = self._forward_model(x)

            if mode == "single_step":
                if self.task_type == "classification":
                    if return_logits:
                        outs.append(outputs[:, 0])
                    else:
                        samples = _sample_categorical_from_logits(outputs[:, 0:1], use_T)
                        outs.append(samples[:, 0])
                else:
                    outs.append(outputs[:, 0])
            else:
                if self.task_type == "classification":
                    if return_logits:
                        outs.append(outputs)
                    else:
                        samples = _sample_categorical_from_logits(outputs, use_T)
                        outs.append(samples)
                else:
                    outs.append(outputs)

        return torch.cat(outs, dim=0)
