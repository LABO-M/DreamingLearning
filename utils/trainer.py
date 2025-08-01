# utils/trainer.py

import torch
import torch.nn.functional as F
import random

def sample_sequence(model, start_token, seq_len, temperature=1.0, device='cpu'):
    model.eval()
    generated = [start_token]
    input_token = torch.tensor([[start_token]], device=device)
    hidden = None

    for _ in range(seq_len - 1):
        logits, hidden = model(input_token, hidden)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
        input_token = next_token

    return generated

def train(model, data, vocab_size, optimizer, device='cpu',
          temperature=1.5, dreaming_ratio=0.2, dreaming_seq_len=20, epochs=5):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        random.shuffle(data)

        for x in data:
            x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
            targets = x[:, 1:]
            inputs = x[:, :-1]

            optimizer.zero_grad()
            output, _ = model(inputs)
            loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

        for _ in range(int(len(data) * dreaming_ratio)):
            start_token = random.randint(0, vocab_size - 1)
            generated = sample_sequence(model, start_token, dreaming_seq_len, temperature, device)

            input_seq = torch.tensor(generated[:-1], dtype=torch.long, device=device).unsqueeze(0)
            target_seq = torch.tensor(generated[1:], dtype=torch.long, device=device).unsqueeze(0)

            optimizer.zero_grad()
            output, _ = model(input_seq)
            dreaming_loss = criterion(output.reshape(-1, vocab_size), target_seq.reshape(-1))
            dreaming_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss = {loss.item():.4f}, dreaming_loss = {dreaming_loss.item():.4f}")
