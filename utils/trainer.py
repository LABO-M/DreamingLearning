# utils/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import random
from torch.distributions import Normal
from torch.optim import Adam
import random
import numpy as np
import os
import math
from scipy.special import betainc

# ===================== ユーティリティ =====================

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def gaussian_nll(mu, logvar, target):
    """
    連続値のガウスNLL（定数項は省略可）
    mu/logvar/target: [B, L, 1]
    L = 0.5*((y - μ)^2 / σ^2 + log σ^2)
    """
    var = logvar.exp().clamp_min(1e-12)
    nll = 0.5 * ((target - mu) ** 2 / var + logvar)
    return nll.mean()

def student_t_nll(mu, log_s, log_nu, y):
    """
    mu, log_s, log_nu, y: shape [B, L, 1]
    返り値: バッチ平均の NLL（定数項込み）
    """
    s  = log_s.clamp(-8.0, 4.0).exp()             # σ in [~0.0003, ~54]
    nu = log_nu.clamp(math.log(2.05), math.log(60.0)).exp()  # ν>2
    z  = (y - mu) / (s + 1e-12)                   # 標準化
    z2 = z*z

    # NLL = -log f_tν(y|μ,σ)
    # 参考式: 0.5*log(pi*nu) + log(s) + lgamma((nu+1)/2) - lgamma(nu/2) + 0.5*(nu+1)*log(1+z^2/nu)
    const = 0.5*(math.log(math.pi)) + torch.log(nu)/2 + torch.log(s)
    nll = const + torch.lgamma((nu+1)/2) - torch.lgamma(nu/2) + 0.5*(nu+1)*torch.log1p(z2/nu)
    return nll.mean()

@torch.no_grad()
def student_t_cdf_standard(x: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """
    標準 t(v) の CDF。x, nu はブロードキャスト可能であること。
    公式:
      t >= 0:  F(t) = 1 - 0.5 * I_{ v/(v + t^2) }(v/2, 1/2)
      t <  0:  F(t) = 0.5 * I_{ v/(v + t^2) }(v/2, 1/2)
    ※ torch.special.betainc は正則化不完全ベータ I_x(a,b)
    """
    x_np = x.detach().cpu().numpy()
    nu_np = nu.detach().cpu().numpy()

    # t >= 0 と t < 0 で場合分け
    t = nu_np / (nu_np + x_np**2)
    a = 0.5 * nu_np
    b = 0.5
    inc = betainc(a, b, t).clip(0.0, 1.0)

    cdf_np = np.where(x_np >= 0, 1.0 - 0.5 * inc, 0.5 * inc)
    return torch.from_numpy(cdf_np).to(x.device, dtype=x.dtype)

@torch.no_grad()
def student_t_ppf_standard(p: torch.Tensor, nu: torch.Tensor, iters: int = 50) -> torch.Tensor:
    """
    標準 t(v) の PPF。cdf(x; v)=p を二分法で解く。
    p, nu はブロードキャスト可能な同shape想定。
    """
    device = p.device
    p = p.to(device).clamp(1e-6, 1 - 1e-6)
    nu = nu.to(device).clamp_min(2.001)

    # 初期区間（正規近似で中心を置きつつ十分広くとる）
    normal = torch.distributions.Normal(torch.tensor(0.0, device=device),
                                        torch.tensor(1.0, device=device))
    z0 = normal.icdf(p)
    lo = (z0 - 50.0).clone()
    hi = (z0 + 50.0).clone()

    for _ in range(iters):
        mid = (lo + hi) * 0.5
        c = student_t_cdf_standard(mid, nu)
        go_left = (c >= p)
        hi = torch.where(go_left, mid, hi)
        lo = torch.where(go_left, lo, mid)

    return (lo + hi) * 0.5

class TemperatureScheduler:
    def __init__(self, kind="constant", base=1.5, **kw):
        self.kind, self.base, self.kw = kind, float(base), kw
        self.best = float("inf"); self.bad_epochs = 0

    def step(self, epoch=None, val_metric=None):
        k = self.kind
        if k == "constant":
            return self.base
        if k == "cosine":
            t_min = float(self.kw.get("t_min", 1.2))
            t_max = float(self.kw.get("t_max", 1.8))
            period = max(1, int(self.kw.get("period", 10)))
            phase = (epoch % period) / period
            return t_min + 0.5*(t_max - t_min)*(1 - math.cos(2*math.pi*phase))
        if k == "step":
            milestones = self.kw.get("milestones", [10, 20])
            gamma = float(self.kw.get("gamma", 1.1))
            T = self.base
            for m in milestones:
                if epoch is not None and epoch >= m:
                    T *= gamma
            return T
        if k == "plateau":
            patience = int(self.kw.get("patience", 3))
            factor   = float(self.kw.get("factor", 1.1))
            if val_metric is None: return self.base
            if val_metric < self.best - 1e-6:
                self.best = val_metric; self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            return self.base * (factor if self.bad_epochs >= patience else 1.0)
        return self.base

# ===================== メイン：DreamingTrainer =====================

class DreamingTrainer:
    """
    連続値用 Dreaming 学習（LSTM＋ガウスヘッド想定）
    - model(x, hidden)-> (output, hidden_next)
      output[...,0]=μ, output[...,1]=logσ² というあなたの現行仕様に合わせています
    - data: list of sequences（各要素 shape=[L] か [L,D], D=1+n_exo）
    """

    def __init__(self, model, optimizer, device='cpu', loss_mode: str ="gaussian", # "gaussian"|"studentT"
                 # 粒度
                 interleave_mode="epoch",  # "epoch"|"batch"|"mixed"
                 epochs=5, warm_vanilla_steps=0,
                 max_vanilla_steps_per_epoch=200,
                 dreaming_steps_per_epoch=50,
                 dreaming_seq_len=50,
                 K_interleave=6, lambda_d=0.2,
                 # 自由度
                 lambda_nu_l2: float = 0.0,
                 lambda_nu_inv: float = 0.0,
                 lambda_scale_l2: float = 0.0,
                 nu_target: float = 10.0,
                 nu_floor: float = 3.0,
                 nu_clamp: tuple = (2.05, 60.0),
                 # 温度
                 temperature=1.5,
                 temperature_schedule="constant",
                 temperature_params=None,
                 # 外生
                 exo_mode="hold",  # "hold"|"resample"|"roll"
                 # 数値安定・再現
                 grad_clip=1.0, logvar_clamp=(-20, 10), seed=42):

        self.model = model
        self.optim = optimizer
        self.device = device
        self.loss_mode = loss_mode

        self.interleave_mode = interleave_mode
        self.epochs = epochs
        self.warm_vanilla_steps = warm_vanilla_steps
        self.max_vanilla_steps_per_epoch = max_vanilla_steps_per_epoch
        self.dreaming_steps_per_epoch = dreaming_steps_per_epoch
        self.dreaming_seq_len = dreaming_seq_len
        self.K_interleave = K_interleave
        self.lambda_d = lambda_d

        self.lambda_nu_l2 = lambda_nu_l2
        self.lambda_nu_inv = lambda_nu_inv
        self.lambda_scale_l2 = lambda_scale_l2
        self.nu_target = float(nu_target)
        self.nu_floor = float(nu_floor)
        self.nu_clamp = (float(nu_clamp[0]), float(nu_clamp[1]))

        self.temperature = temperature
        self.sched = TemperatureScheduler(temperature_schedule, base=temperature,
                                          **(temperature_params or {}))
        self.exo_mode = exo_mode

        self.grad_clip = grad_clip
        self.logvar_clamp = logvar_clamp
        self.seed = seed

    # ---------- 公開API ----------
    def train(self, data, val_data=None):
        """
        data: list[ np.ndarray | list | torch.Tensor ]
              各要素は shape=[L] か [L,D], D=1+n_exo
              先頭列（index=0）をターゲットとみなします。
        """
        seed_all(self.seed)
        self.model.to(self.device)

        # Warm-up (Vanilla only)
        if self.warm_vanilla_steps > 0:
            print(f"[WARMUP] vanilla_steps={self.warm_vanilla_steps}")
            self.model.train()
            steps = 0
            for seq in self._epoch_iter(data):
                loss_v,mse_v, mae_v = self._train_step_vanilla(seq)
                steps += 1
                if steps % 50 == 0:
                    print(f"[WARMUP] step {steps}, loss={loss_v:.6f}, mse={mse_v:.6f}, mae={mae_v:.6f}")
                if steps >= self.warm_vanilla_steps:
                    break

        best_val = float('inf')

        # Epoch loop
        for e in range(1, self.epochs + 1):
            # 温度更新（plateauはVal後にも再評価）
            T = self.sched.step(epoch=e)

            if self.interleave_mode == "epoch":
                v_avg, d_avg = self._run_epoch_mode(data, T)
            elif self.interleave_mode == "batch":
                v_avg, d_avg = self._run_batch_mode(data, T)
            else:  # "mixed"
                v_avg, d_avg = self._run_mixed_mode(data, T)

            # Validation（任意）
            val_nll = None
            if val_data is not None:
                val_nll = self.evaluate_nll(val_data)
                best_val = min(best_val, val_nll)
                print(f"[E{e}] VAL nll={val_nll:.6f} (best={best_val:.6f})")

            # plateau の場合はValで再調整
            if isinstance(self.sched, TemperatureScheduler) and self.sched.kind == "plateau":
                T = self.sched.step(epoch=e, val_metric=val_nll)

            print(f"[E{e}] SUMMARY: vanilla_loss={v_avg:.6f}, dreaming_loss={d_avg:.6f}, T={T:.3f}")

    def evaluate_nll(self, data, max_batches=256):
        """簡易バリデーション（ガウスNLLの平均）"""
        self.model.eval()
        losses, seen = [], 0
        with torch.no_grad():
            for seq in self._epoch_iter(data):
                x = self._to_tensor(seq)                               # [1,L,D]
                inputs, targets = x[:, :-1, :], x[:, 1:, [0]]          # [1,L-1,D], [1,L-1,1]
                output, _ = self.model(inputs)
                loss, _, _ = self._compute_loss_and_metrics(output, targets).item()
                losses.append(loss)
                seen += 1
                if seen >= max_batches: break
        self.model.train()
        return float(np.mean(losses)) if losses else float('inf')

    # ---------- 内部：1epochの回し方 ----------
    def _run_epoch_mode(self, data, T):
        # VANILLA block
        print(f"[VANILLA] (epoch mode) start")
        v_sum, v_steps = 0.0, 0
        v_mse_sum, v_mae_sum = 0.0, 0.0
        for seq in self._epoch_iter(data):
            loss_v, mse_v, mae_v = self._train_step_vanilla(seq)
            v_sum += loss_v; v_steps += 1
            v_mse_sum += mse_v
            v_mae_sum += mae_v
            if v_steps % 50 == 0:
                print(f"[VANILLA] step {v_steps}/{self.max_vanilla_steps_per_epoch}, loss={loss_v:.6f}, mse={mse_v:.6f}, mae={mae_v:.6f}")
            if v_steps >= self.max_vanilla_steps_per_epoch:
                break

        # DREAMING block
        print(f"[DREAMING] (epoch mode) start, T={T:.3f}")
        d_sum, d_steps = 0.0, 0
        d_mse_sum, d_mae_sum = 0.0, 0.0
        for _ in range(self.dreaming_steps_per_epoch):
            start_window = self._make_start_window_from(data)
            dream_seq = self._sample_sequence_continuous_exo(start_window, self.dreaming_seq_len, T)
            loss_d, mse_d, mae_d = self._train_step_dreaming(dream_seq)
            d_sum += loss_d; d_steps += 1
            d_mse_sum += mse_d
            d_mae_sum += mae_d
            if d_steps % 20 == 0:
                print(f"[DREAMING] step {d_steps}/{self.dreaming_steps_per_epoch}, loss={loss_d:.6f}, mse={mse_d:.6f}, mae={mae_d:.6f}")

        v_avg = v_sum / max(1, v_steps)
        v_mse = v_mse_sum / max(1, v_steps)
        v_mae = v_mae_sum / max(1, v_steps)
        d_avg = d_sum / max(1, d_steps)
        d_mse = d_mse_sum / max(1, d_steps)
        d_mae = d_mae_sum / max(1, d_steps)

        print(f"[EPOCH MODE] VANILLA avg_loss={v_avg:.6f}, DREAMING avg_loss={d_avg:.6f}, mse={v_mse:.6f}, mae={v_mae:.6f}")
        print(f"[EPOCH MODE] DREAMING avg_loss={d_avg:.6f}, mse={d_mse:.6f}, mae={d_mae:.6f}")

        return v_avg, d_avg

    def _run_batch_mode(self, data, T):
        print(f"[INTERLEAVE] (batch mode) K={self.K_interleave}, T={T:.3f}")
        v_sum, d_sum, v_steps, d_steps = 0.0, 0.0, 0, 0
        v_mse_sum, v_mae_sum = 0.0, 0.0
        d_mse_sum, d_mae_sum = 0.0, 0.0
        for i, seq in enumerate(self._epoch_iter(data), 1):
            loss_v, mse_v, mae_v = self._train_step_vanilla(seq)
            v_sum += loss_v; v_steps += 1
            v_mse_sum += mse_v; v_mae_sum += mae_v
            if i % self.K_interleave == 0:
                start_window = self._make_start_window_from_seq(seq)
                dreamed = self._sample_sequence_continuous_exo(start_window, max(1, self.dreaming_seq_len//2), T)
                loss_d, mse_d, mae_d = self._train_step_dreaming(dreamed)
                d_sum += loss_d; d_steps += 1
                d_mse_sum += mse_d; d_mae_sum += mae_d
            if v_steps >= self.max_vanilla_steps_per_epoch: break

        v_avg = v_sum / max(1, v_steps)
        d_avg = d_sum / max(1, d_steps)
        v_mse = v_mse_sum / max(1, v_steps)
        v_mae = v_mae_sum / max(1, v_steps)
        d_mse = d_mse_sum / max(1, d_steps)
        d_mae = d_mae_sum / max(1, d_steps)

        print(f"[BATCH MODE] VANILLA avg_loss={v_avg:.6f}, mse={v_mse:.6f}, mae={v_mae:.6f}")
        print(f"[BATCH MODE] DREAMING avg_loss={d_avg:.6f}, mse={d_mse:.6f}, mae={d_mae:.6f}")

        return v_avg, d_avg

    def _run_mixed_mode(self, data, T):
        print(f"[MIXED] lambda_d={self.lambda_d}, T={T:.3f}")
        v_sum, d_sum, steps = 0.0, 0.0, 0
        for seq in self._epoch_iter(data):
            # Vanilla（forwardだけ、後で合成）
            x = self._to_tensor(seq)
            inputs, targets = x[:, :-1, :], x[:, 1:, [0]]
            output, _ = self.model(inputs)
            mu, logvar = self._parse_mu_logvar(output)
            loss_v = gaussian_nll(mu, logvar, targets)

            # 短いDreamを同一イテレーションで合成
            start_window = self._make_start_window_from_seq(seq)
            dreamed = self._sample_sequence_continuous_exo(start_window, max(1, self.dreaming_seq_len//2), T)
            inp, tgt = self._dream_to_batch(dreamed)
            out_d, _ = self.model(inp)
            loss_d, _, _, _ = self._compute_loss_and_metrics(out_d, tgt)

            loss = loss_v + self.lambda_d * loss_d
            self.optim.zero_grad(); loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optim.step()

            v_sum += float(loss_v.item()); d_sum += float(loss_d.item()); steps += 1
            if steps >= self.max_vanilla_steps_per_epoch: break
        return v_sum / max(1, steps), d_sum / max(1, steps)

    # ---------- 内部：学習1ステップ ----------
    def _train_step_vanilla(self, seq):
        self.model.train()
        x = self._to_tensor(seq)                               # [1,L,D]
        if x.size(1) < 2:
            return 0.0, 0.0, 0.0
        inputs, targets = x[:, :-1, :], x[:, 1:, [0]]          # [1,L-1,D], [1,L-1,1]
        output, _ = self.model(inputs)                         # output[...,0]=μ,1=logσ²
        loss, mse, mae, _ = self._compute_loss_and_metrics(output, targets)
        self.optim.zero_grad(); loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optim.step()
        return float(loss.item()), mse, mae

    def _train_step_dreaming(self, dream_seq):
        """
        dream_seq: [1, K, D]  （_sample_sequence_continuous_exo の出力をそのまま入れる）
        """
        self.model.train()
        assert isinstance(dream_seq, torch.Tensor) and dream_seq.dim() == 3
        inp = dream_seq[:, :-1, :]         # [1,K-1,D]
        tgt = dream_seq[:, 1:, [0]]        # [1,K-1,1]  ← 教師はターゲット列のみ

        out, _ = self.model(inp)
        loss, mse, mae, _ = self._compute_loss_and_metrics(out, tgt)

        self.optim.zero_grad(); loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optim.step()
        return float(loss.item()), mse, mae


    # ---------- 内部：Dreaming 生成 ----------
    @torch.no_grad()
    def _sample_sequence_continuous_exo(self, start_window, seq_len, T):
        """
        start_window: [1, W, D]（D=1+n_exo）
        生成：σ_T^2 = T·σ^2（＝あなたの現行実装の「variance * temperature」と等価）
        exo_mode:
          - "hold": 外生は直近値を固定
          - "resample": dataから外生をランダム抽出（簡易）
          - "roll": 簡易AR(1)で少しだけ動かす
        """
        was_training = self.model.training
        self.model.eval()

        x = start_window.clone().to(self.device)  # [1,W,D]
        dreamed_target = []
        hidden = None

        for _ in range(seq_len):
            output, hidden = self.model(x, hidden)     # output: [1,W,2]
            mu_t  = output[:, -1:, [0]]
            lv_t  = output[:, -1:, [1]]
            lv_t  = lv_t.clamp(self.logvar_clamp[0], self.logvar_clamp[1])

            var   = lv_t.exp().clamp_min(1e-12)
            var_T = var * max(1e-8, float(T))          # σ_T^2 = T·σ^2
            sigma_T = var_T.sqrt()

            r_next = mu_t + sigma_T * torch.randn_like(mu_t)  # [1,1,1]

            # 外生を次時点に整える
            D = x.size(-1)
            if D > 1:
                exo_prev = x[:, -1:, 1:]  # [1,1,n_exo]
                if self.exo_mode == "hold":
                    exo_next = exo_prev
                elif self.exo_mode == "resample":
                    # 簡易：同バッチ開始窓の外生をランダムにノイズ付加（本格実装はデータから厳密抽出）
                    exo_next = exo_prev + 0.01 * torch.randn_like(exo_prev)
                else:  # "roll"
                    exo_next = 0.9 * exo_prev + 0.1 * torch.randn_like(exo_prev)
                x_next = torch.cat([r_next, exo_next], dim=-1)   # [1,1,D]
            else:
                x_next = r_next

            dreamed_target.append(x_next)
            x = torch.cat([x, x_next], dim=1)

        dream_seq = torch.cat(dreamed_target, dim=1)  # [1,seq_len,D]
        if was_training: self.model.train()
        return dream_seq

    # ---------- 予測 ----------
    @torch.no_grad()
    def predict_in_sample(self, seq):
        """
        seq: [L, D]（全期間でも、train部分だけでもOK）
        戻り値: dict { "y": [L-1], "y_hat": [L-1], "err": [L-1] }  ※numpy
        （時刻 t+1 の真値 y と 1ステップ予測 μ_t を整列）
        """
        self.model.eval()
        x = self._to_tensor(seq)                  # [1,L,D]
        if x.size(1) < 2:
            return {"y": np.array([]), "y_hat": np.array([]), "err": np.array([])}
        inputs, targets = x[:, :-1, :], x[:, 1:, [0]]   # [1,L-1,D], [1,L-1,1]
        out, _ = self.model(inputs)
        mu, _ = self._parse_mu_logvar(out)        # [1,L-1,1]
        y     = targets.squeeze(0).squeeze(-1).cpu().numpy()
        y_hat = mu.squeeze(0).squeeze(-1).cpu().numpy()
        return {"y": y, "y_hat": y_hat, "err": (y_hat - y)}

    @torch.no_grad()
    def predict_out_of_sample(self, seq, test_len, use_mean=True, T=1.0):
        """
        seq: [L, D], test_len: 後ろ test_len ステップを予測
        use_mean=True: μ をそのまま予測値に（推奨）
                False: N(μ, T·σ^2) から1サンプル（シミュレーション用途）
        戻り値: dict { "y": [test_len], "y_hat": [test_len] }  ※numpy
        """
        self.model.eval()
        train_arr, test_arr = self.split_train_test(seq, test_len)  # [Ltr,D], [Lte,D]
        x_tr = self._to_tensor(train_arr)                           # [1,Ltr,D]

        # 1) train 部分で隠れ状態を温める
        out_tr, hidden = self.model(x_tr)

        # 2) 直近時点のベクトル（target は真値、exo は真値）から開始
        last_vec = x_tr[:, -1:, :].clone()   # [1,1,D]
        preds = []
        for i in range(test_len):
            # 真の外生（列1..）を test から取得
            exo_true_next = torch.tensor(test_arr[i:i+1, 1:], dtype=torch.float32, device=self.device).view(1,1,-1)

            # 直前の target（最初は train の真値、その後は前回予測）をベクトル化
            target_prev = last_vec[:, -1:, [0]]  # [1,1,1]

            # モデルで1歩先の分布を出す（コンテキストは last_vec）
            out, hidden = self.model(last_vec, hidden)   # out: [1,1,2]
            if self.loss_mode == "gaussian":
                mu, logvar = self._parse_mu_logvar(out)
                if use_mean:
                    target_next = mu
                else:
                    var_T = logvar.exp().clamp_min(1e-12) * max(1e-8, float(T))
                    target_next = mu + var_T.sqrt() * torch.randn_like(mu)
            elif self.loss_mode == "studentT":
                mu = out[..., [0]]
                log_s = out[..., [1]]
                log_nu = out[..., [2]]
                s = log_s.exp().clamp_min(1e-12)
                nu = log_nu.exp().clamp_min(2.01)
                if use_mean:
                    target_next = mu
                else:
                    dist = torch.distributions.StudentT(
                        df = nu.squeeze(-1),loc = mu.squeeze(-1), scale = s.squeeze(-1)
                    )
                    sample = dist.sample()
                    target_next = sample.view_as(mu)
            else:
                raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

            # 次の入力ベクトルを作る（target_next と “真の” exo を結合）
            if last_vec.size(-1) > 1:
                x_next = torch.cat([target_next, exo_true_next], dim=-1)  # [1,1,D]
            else:
                x_next = target_next

            preds.append(target_next.item())
            last_vec = x_next  # ロール

        y_true = test_arr[:, 0]        # 真のターゲット
        y_hat  = np.array(preds, dtype=np.float32)
        return {"y": y_true, "y_hat": y_hat}

    # ---------- 内部：前処理/補助 ----------
    def _to_tensor(self, seq):
        """
        入力seq（list/ndarray/tensor） -> torch.Tensor [1,L,D]
        D=1（外生なし） も D>1（外生あり）もOK
        """
        if isinstance(seq, torch.Tensor):
            arr = seq
        else:
            arr = torch.tensor(seq, dtype=torch.float32)
        if arr.dim() == 1:
            arr = arr.view(1, -1, 1)
        elif arr.dim() == 2:
            arr = arr.unsqueeze(0)
        else:
            raise ValueError("seq must be shape [L] or [L,D]")
        return arr.to(self.device)

    def split_train_test(self, seq, test_len):
        """
        seq: np.ndarray or torch.Tensor [L, D]（列0=ターゲット）
        test_len: 後ろから test_len ステップをテストに
        戻り値: train_seq [L_tr, D], test_seq [L_te, D]
        """
        arr = seq.detach().cpu().numpy() if isinstance(seq, torch.Tensor) else np.asarray(seq)
        assert arr.ndim == 2, "seq must be [L, D]"
        L = arr.shape[0]
        assert 1 <= test_len < L, "test_len を見直してください"
        return arr[:L - test_len], arr[L - test_len:]

    def _parse_mu_logvar(self, output):
        """
        あなたの現行モデル仕様に合わせて：
        output[...,0]=μ, output[...,1]=logσ²
        """
        mu  = output[..., [0]]
        lv  = output[..., [1]].clamp(self.logvar_clamp[0], self.logvar_clamp[1])
        return mu, lv

    def _make_start_window_from(self, data):
        """実データからランダムな開始窓を作る（外生があってもOK）"""
        seq = random.choice(data)
        arr = self._to_tensor(seq)                # [1,L,D]
        L = arr.size(1)
        W = max(5, min(self.dreaming_seq_len, L-1))
        s = random.randint(0, max(0, L-1-W))
        return arr[:, s:s+W, :]                   # [1,W,D]

    def _make_start_window_from_seq(self, seq):
        arr = self._to_tensor(seq)
        L = arr.size(1)
        W = max(5, min(self.dreaming_seq_len, L-1))
        s = random.randint(0, max(0, L-1-W))
        return arr[:, s:s+W, :]

    def _dream_to_batch(self, dreamed_list):
        """
        Dreamingで得たターゲット列 -> 学習用の (inputs, targets)
        inputs: [1,K-1,1], targets: [1,K-1,1]
        """
        y = torch.tensor(dreamed_list, dtype=torch.float32, device=self.device).view(1, -1, 1)
        return y[:, :-1, :], y[:, 1:, :]

    def _epoch_iter(self, data):
        """シーケンス単位を1バッチとして回す簡易イテレータ"""
        idx = list(range(len(data)))
        random.shuffle(idx)
        for i in idx:
            yield data[i]

    def _mse_mae(self, mu: torch.Tensor, target: torch.Tensor):
        """
        mu, target: [B, L, 1]
        戻り: mse(float), mae(float)
        """
        with torch.no_grad():
            diff = (mu - target)
            mse = float((diff ** 2).mean().item())
            mae = float(diff.abs().mean().item())
        return mse, mae
    @torch.no_grad()
    def evaluate_metrics(self, data, max_batches=512):
        """
        data: list[[L,D]] の系列群（train用にもtest用にもOK）
        戻り: {"nll":..., "mse":..., "mae":...} の平均

        NLL: ガウスNLL平均
        MSE: 点誤差平均
        MAE: 点誤差平均
        Coverage: 90%信頼区間カバレッジ平均
        """
        self.model.eval()
        nlls, mses, maes = [], [], []
        covs_90 = []
        seen = 0

            # 分位点用の常数（正規）
        standard_normal = torch.distributions.Normal(0.0, 1.0)
        q = 0.90
        alpha = 1.0 - q

        for seq in self._epoch_iter(data):
            x = self._to_tensor(seq)
            if x.size(1) < 2: continue
            inputs, targets = x[:, :-1, :], x[:, 1:, [0]]
            out, _ = self.model(inputs)
            nll, mse, mae, _ = self._compute_loss_and_metrics(out, targets)
            nlls.append(nll.item())
            mses.append(mse); maes.append(mae)

            if self.loss_mode == "gaussian":
                mu  = out[..., 0]                                 # [B,L]
                var = out[..., 1].exp().clamp_min(1e-12)          # logvar -> var
                sigma = var.sqrt()
                z_lo = standard_normal.icdf(torch.tensor(alpha/2, device=out.device))
                z_hi = standard_normal.icdf(torch.tensor(1.0 - alpha/2, device=out.device))
                L = mu + sigma * z_lo
                U = mu + sigma * z_hi

            elif self.loss_mode == "studentT":
                mu     = out[..., 0]
                s      = out[..., 1].exp().clamp_min(1e-12)
                nu     = out[..., 2].exp().clamp_min(2.01)
                # 形状を mu と合わせたテンソルで icdf を評価
                p_lo = torch.full_like(mu, alpha/2)
                p_hi = torch.full_like(mu, 1.0 - alpha/2)
                t_lo = student_t_ppf_standard(p_lo, nu)
                t_hi = student_t_ppf_standard(p_hi, nu)
                L = mu + s * t_lo
                U = mu + s * t_hi

            else:
                raise ValueError(f"unknown loss_mode: {self.loss_mode}")
            tgt = targets.squeeze(-1)  # [B,L]
            in_interval = ((tgt >= L) & (tgt <= U)).float().mean().item()
            covs_90.append(in_interval)

            seen += 1
            if seen >= max_batches: break
        return {
            "nll": float(np.mean(nlls)) if nlls else float("nan"),
            "mse": float(np.mean(mses)) if mses else float("nan"),
            "mae": float(np.mean(maes)) if maes else float("nan"),
            "coverage_90": float(np.mean(covs_90)) if covs_90 else float("nan"),
            "cov_loss_90": (float(np.mean(covs_90)) - 0.90) ** 2 if covs_90 else float("nan"),
        }
    def _compute_loss_and_metrics(self, out, targets):
        """
        out: [B,L,2 or 3], targets: [B,L,1]
        戻り: (loss, mse, mae, extra_dict)
        """
        if self.loss_mode == "gaussian":
            mu = out[..., [0]]
            lv = out[..., [1]]
            loss = gaussian_nll(mu, lv, targets)
        elif self.loss_mode == "studentT":
            mu     = out[..., [0]]
            log_s  = out[..., [1]]
            log_nu = out[..., [2]]
            base_loss = student_t_nll(mu, log_s, log_nu, targets)

            # 正則化項
            reg = 0.0
            if self.lambda_nu_l2 > 0.0:
                reg = reg + self.lambda_nu_l2 * torch.mean((log_nu - math.log(self.nu_target))**2)
            if self.lambda_nu_inv > 0.0:
                nu = log_nu.exp()
                reg = reg + self.lambda_nu_inv * torch.mean(
                    (self.nu_floor / nu)**2
                )
            if self.lambda_scale_l2 > 0.0:
                reg = reg + self.lambda_scale_l2 * torch.mean(log_s**2)

            loss = base_loss + reg
        else:
            raise ValueError(f"unknown loss_mode: {self.loss_mode}")

        # 点誤差（参考用）
        mse = torch.mean((out[..., [0]] - targets)**2).item()
        mae = torch.mean(torch.abs(out[..., [0]] - targets)).item()
        return loss, mse, mae, {}



def train_portfolio(model, train_loader, val_loader, n_assets, optimizer, epochs=10, temperature=1.5):
    vol_criterion = nn.MSELoss()
    corr_criterion = nn.MSELoss()
    device = next(model.parameters()).device

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        dreaming_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # ボラティリティと相関の正解データを計算
            true_vols = torch.var(targets, dim=1)
            true_corrs = torch.stack([
                torch.corrcoef(targets[i].T)[torch.triu(torch.ones(n_assets, n_assets), diagonal=1).bool()]
                for i in range(targets.size(0))
            ]).to(device)

            optimizer.zero_grad()
            pred_vols, pred_corrs = model(inputs)
            loss_vol = vol_criterion(pred_vols, true_vols)
            loss_corr = corr_criterion(pred_corrs, true_corrs)
            loss = loss_vol + loss_corr
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Dreaming Phase
        model.eval()
        dreaming_inputs = next(iter(train_loader))[0].to(device)
        generated_sequences = generate_dreaming_data(model, dreaming_inputs, temperature)

        model.train()
        d_loss = 0
        for seq in generated_sequences:
            true_vols = torch.var(seq, dim=1)
            true_corrs = torch.stack([
                torch.corrcoef(seq[i].T)[torch.triu(torch.ones(n_assets, n_assets), diagonal=1).bool()]
                for i in range(seq.size(0))
            ]).to(device)

            optimizer.zero_grad()
            pred_vols, pred_corrs = model(seq)
            d_loss_vol = vol_criterion(pred_vols, true_vols)
            d_loss_corr = corr_criterion(pred_corrs, true_corrs)
            loss = d_loss_vol + d_loss_corr
            loss.backward()
            optimizer.step()
            d_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_dreaming_loss = d_loss / len(generated_sequences)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Dreaming Loss: {avg_dreaming_loss:.4f}")

def generate_dreaming_data(model, inputs, temperature, seq_len=50):
    model.eval()
    B, L, D = inputs.shape
    generated = []
    with torch.no_grad():
        for i in range(B):
            current_input = inputs[i, -1, :].unsqueeze(0).unsqueeze(0)
            generated_seq = current_input.clone()

            for _ in range(seq_len):
                lstm_out, _ = model.lstm(current_input)
                mean_pred = model.vol_head(lstm_out[:, -1, :])

                # 相関ヘッドの出力を利用して分散を予測（モデル定義に合わせる）
                log_variance_pred = model.corr_head(lstm_out[:, -1, :])[:, :D] # ボラティリティ分だけ使用

                variance = torch.exp(log_variance_pred) * temperature

                dist = Normal(mean_pred, torch.sqrt(variance))
                next_val = dist.sample()

                # ★ここが修正点★
                # next_valの形状を [B, L, D] に合わせる
                next_val_reshaped = next_val.unsqueeze(0).unsqueeze(0) # [1, 1, D]

                generated_seq = torch.cat((generated_seq, next_val_reshaped), dim=1)
                current_input = next_val_reshaped
            generated.append(generated_seq[:, 1:, :])
    return generated
