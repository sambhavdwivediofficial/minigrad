from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import (
    Tensor, Embedding, LSTM, Linear,
    Adam, clip_grad_norm_, softmax_cross_entropy,
    save_checkpoint, load_checkpoint, Module, Sequential,
    TensorDataset, DataLoader,
)

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a small autodiff engine can still learn real patterns. "
    "gradients flow backward through every tensor in the graph. "
    "the quick brown fox jumps over the lazy dog again and again. "
)


class CharLSTM(Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, seed=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, seed=seed)
        self.lstm = LSTM(embed_dim, hidden_size, seed=seed)
        self.head = Linear(hidden_size, vocab_size, seed=seed)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        hidden, state = self.lstm(x)
        batch, seq_len, hidden_size = hidden.shape
        logits = self.head(hidden.reshape(batch * seq_len, hidden_size))
        return logits.reshape(batch, seq_len, -1), state


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def make_training_pairs(text, stoi, seq_len):
    data = np.array([stoi[c] for c in text])
    x = np.array([data[i : i + seq_len] for i in range(len(data) - seq_len)])
    y = np.array([data[i + 1 : i + seq_len + 1] for i in range(len(data) - seq_len)])
    return x, y


def generate(model, stoi, itos, seed_text, length, temperature=0.8, seed=0):
    rng = np.random.default_rng(seed)
    tokens = [stoi[c] for c in seed_text]
    for _ in range(length):
        context = np.array([tokens[-32:]])
        logits, _ = model(context)
        last_logits = logits.data[0, -1] / temperature
        probabilities = np.exp(last_logits - last_logits.max())
        probabilities /= probabilities.sum()
        next_token = rng.choice(len(probabilities), p=probabilities)
        tokens.append(int(next_token))
    return "".join(itos[t] for t in tokens)


def parse_args():
    parser = ArgumentParser(prog="char_lstm_demo")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generate-length", type=int, default=120)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/char_lstm_model.npz"))
    return parser.parse_args()


def main():
    args = parse_args()
    stoi, itos = build_vocab(CORPUS)
    x, y = make_training_pairs(CORPUS, stoi, args.seq_len)

    model = CharLSTM(len(stoi), args.embed_dim, args.hidden_size, seed=args.seed)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True, seed=args.seed)

    for epoch in range(1, args.epochs + 1):
        total_loss, total_correct, total_count = 0.0, 0, 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits, _ = model(x_batch)
            targets = y_batch.reshape(-1).astype(np.int64)
            loss = softmax_cross_entropy(logits.reshape(-1, len(stoi)), targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(targets)
            total_correct += int((logits.data.argmax(axis=-1).reshape(-1) == targets).sum())
            total_count += len(targets)

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:4d} loss={total_loss / total_count:.5f} next_char_accuracy={total_correct / total_count:.3f}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, args.checkpoint)
    print(f"Saved model parameters to {args.checkpoint}")

    sample = generate(model, stoi, itos, seed_text="the quick", length=args.generate_length, seed=args.seed)
    print("Generated text:")
    print(sample)


if __name__ == "__main__":
    main()
