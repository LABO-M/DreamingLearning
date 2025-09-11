# utils/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import random
from torch.distributions import Normal
from torch.optim import Adam
import random

def sample_sequence(model, start_token, seq_len, temperature=1.0, device='cpu'):
    model.eval()
    generated = [start_token]
    input_token = torch.tensor([[start_token]], device=device)
    hidden = None

    with torch.no_grad():  # 勾配計算不要
        for _ in range(seq_len - 1):
            logits, hidden = model(input_token, hidden)  # logits: [1, 1, vocab_size]
            logits = logits[:, -1, :] / temperature       # 温度でスケーリング
            probs = F.softmax(logits, dim=-1)             # ソフトマックスで確率分布化
            next_token = torch.multinomial(probs, num_samples=1)  # サンプリング
            generated.append(next_token.item())
            input_token = next_token.unsqueeze(0)         # 次ステップの入力に整形

    return generated
def train(model, data, vocab_size, optimizer, device='cpu',
          temperature=1.5, dreaming_ratio=0.2, dreaming_seq_len=20, epochs=5):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        total_dreaming_loss = 0

        # --- 通常の学習（Vanilla phase） ---
        for x in data:
            x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]
            if x.size(1) < 2:
                continue  # 入力長不足のデータをスキップ

            inputs = x[:, :-1]
            targets = x[:, 1:]

            optimizer.zero_grad()
            output, _ = model(inputs)  # 出力: [1, seq_len-1, vocab_size]
            loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- ドリーミング学習（Dreaming phase） ---
        dreaming_steps = int(len(data) * dreaming_ratio)
        for _ in range(dreaming_steps):
            start_token = random.randint(0, vocab_size - 1)
            generated = sample_sequence(model, start_token, dreaming_seq_len, temperature, device)

            input_seq = torch.tensor(generated[:-1], dtype=torch.long, device=device).unsqueeze(0)
            target_seq = torch.tensor(generated[1:], dtype=torch.long, device=device).unsqueeze(0)

            optimizer.zero_grad()
            output, _ = model(input_seq)
            dreaming_loss = criterion(output.reshape(-1, vocab_size), target_seq.reshape(-1))
            dreaming_loss.backward()
            optimizer.step()
            total_dreaming_loss += dreaming_loss.item()

        avg_loss = total_loss / max(1, len(data))
        avg_dreaming_loss = total_dreaming_loss / max(1, dreaming_steps)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}, dreaming_loss = {avg_dreaming_loss:.4f}")


def train_price(model, data, optimizer, device='cpu',
                temperature=1.5, dreaming_ratio=0.2, dreaming_seq_len=50, epochs=5):
    # MSELossは平均と分散のパラメータ予測には適さないため、
    # 損失関数を修正する必要がある。ここでは、簡単のため、
    # 平均の予測に対してのみMSEを計算する。
    criterion = MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_dreaming_loss = 0

        # --- 通常の学習 ---
        for seq in data:
            arr = torch.tensor(seq, dtype=torch.float32, device=device).view(1, -1, 1)
            input_seq = arr[:, :-1]
            target_seq = arr[:, 1:]

            optimizer.zero_grad()
            # モデルから平均と対数分散を出力
            output, _ = model(input_seq)
            mean_pred = output[:, :, 0:1] # 平均の予測
            loss = criterion(mean_pred, target_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- Dreaming学習（ギブスサンプリングを実装） ---
        dreaming_steps = int(len(data) * dreaming_ratio)
        for _ in range(dreaming_steps):
            idx = torch.randint(0, len(data), (1,)).item()
            start = torch.tensor([data[idx][0]], dtype=torch.float32).to(device)
            generated = [start]

            model.eval()
            with torch.no_grad():
                input_val = start.view(1, 1, 1)
                hidden = None
                for _ in range(dreaming_seq_len - 1):
                    # モデルから平均と対数分散を取得
                    output, hidden = model(input_val, hidden)
                    mean = output[:, -1, 0]
                    log_variance = output[:, -1, 1]

                    # サンプリング温度を分散に乗じて探索を制御
                    variance = torch.exp(log_variance) * temperature

                    # 予測されたガウス分布からサンプリング
                    dist = torch.distributions.Normal(mean, torch.sqrt(variance))
                    next_val = dist.sample()
                    generated.append(next_val)
                    input_val = next_val.view(1, 1, 1)

            model.train()
            gen_seq = torch.stack(generated).view(1, -1, 1)
            inp = gen_seq[:, :-1]
            tgt = gen_seq[:, 1:]

            optimizer.zero_grad()
            # 人工データの平均の予測に対して損失を計算
            d_output, _ = model(inp)
            d_mean_pred = d_output[:, :, 0:1]
            d_loss = criterion(d_mean_pred, tgt)
            d_loss.backward()
            optimizer.step()
            total_dreaming_loss += d_loss.item()

        print(f"[Epoch {epoch+1}] loss={total_loss/len(data):.6f}, dreaming_loss={total_dreaming_loss/max(1, dreaming_steps):.6f}")


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
