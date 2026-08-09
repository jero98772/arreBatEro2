"""
Character-level Bigram Language Model, matching Andrej Karpathy's
"Let's build GPT" tutorial (the very first, simplest baseline model).

Kept intentionally simple so the whole training loop is easy to follow
and fast to run on CPU. Swap `BigramLanguageModel` out for something
bigger later if you want fancier metrics.
"""

import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "input.txt"
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def ensure_dataset() -> str:
    """Load the tiny shakespeare dataset, downloading it on first run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                "Could not download the tinyshakespeare dataset automatically "
                f"({exc}). Please download it manually from {DATA_URL} and "
                f"save it to {DATA_PATH}"
            ) from exc
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


class Tokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # each token directly reads off the logits for the next token
        # from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    @torch.no_grad()
    def generate_with_probs(self, idx, max_new_tokens: int, temperature: float = 1.0):
        """Generate tokens, also returning the probability the model
        assigned to each chosen token (used to color tokens in the UI)."""
        chosen_probs = []
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            chosen_probs.append(probs.gather(1, idx_next).item())
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, chosen_probs
