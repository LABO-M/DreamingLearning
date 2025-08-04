# utils/trainer.py

import torch
import torch.nn.functional as F
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
