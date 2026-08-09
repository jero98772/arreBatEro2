"""
Background training loop for the bigram LM, instrumented with the
metrics you'd want to babysit during training:

  train loss / val loss / train ppl / val ppl / learning rate /
  grad norm / param norm / tokens processed

Events are appended to an in-memory, append-only list (`self.history`)
that the SSE endpoint replays/polls from. Training itself runs in a
background thread so it doesn't block the FastAPI event loop.
"""

import math
import threading
from dataclasses import dataclass, asdict
from typing import Optional

import torch

from .model import BigramLanguageModel, Tokenizer, ensure_dataset


@dataclass
class TrainConfig:
    max_iters: int = 3000
    batch_size: int = 32
    block_size: int = 8
    learning_rate: float = 1e-2
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 10
    warmup_iters: int = 100
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    max_new_tokens_sample: int = 160
    sample_prompt: str = "\n"


class TrainingManager:
    def __init__(self):
        text = ensure_dataset()
        self.tokenizer = Tokenizer(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = TrainConfig()
        self.model: Optional[BigramLanguageModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.history: list[dict] = []
        self.is_training = False
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.current_step = 0
        self.tokens_processed = 0

        self._init_model()

    # ---------------------------------------------------------- lifecycle
    def _init_model(self):
        torch.manual_seed(1337)
        self.model = BigramLanguageModel(self.tokenizer.vocab_size).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.current_step = 0
        self.tokens_processed = 0
        self.history = []

    def reset(self):
        with self._lock:
            if self.is_training:
                raise RuntimeError("Cannot reset while training is in progress")
            self._init_model()

    # -------------------------------------------------------------- data
    def get_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data
        bs, bl = self.config.batch_size, self.config.block_size
        ix = torch.randint(len(data) - bl, (bs,))
        x = torch.stack([data[i : i + bl] for i in ix])
        y = torch.stack([data[i + 1 : i + bl + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    # --------------------------------------------------------- scheduler
    def _get_lr(self, it: int) -> float:
        cfg = self.config
        base_lr = cfg.learning_rate
        min_lr = base_lr * cfg.min_lr_ratio
        if it < cfg.warmup_iters:
            return base_lr * (it + 1) / max(1, cfg.warmup_iters)
        if it >= cfg.max_iters:
            return min_lr
        decay_ratio = (it - cfg.warmup_iters) / max(1, (cfg.max_iters - cfg.warmup_iters))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (base_lr - min_lr)

    # -------------------------------------------------------------- eval
    @torch.no_grad()
    def _estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.get_batch(split)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    @torch.no_grad()
    def _param_norm(self) -> float:
        total = torch.zeros(1, device=self.device)
        for p in self.model.parameters():
            total += p.data.float().norm(2) ** 2
        return total.sqrt().item()

    @torch.no_grad()
    def _sample(self):
        cfg = self.config
        prompt_ids = self.tokenizer.encode(cfg.sample_prompt) or [0]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        self.model.eval()
        out_idx, probs = self.model.generate_with_probs(idx, cfg.max_new_tokens_sample)
        self.model.train()
        generated_ids = out_idx[0].tolist()
        new_ids = generated_ids[len(prompt_ids):]
        tokens = [
            {"char": self.tokenizer.itos[i], "prob": p}
            for i, p in zip(new_ids, probs)
        ]
        return {"prompt": cfg.sample_prompt, "tokens": tokens}

    # ----------------------------------------------------------- control
    def start(self, config: Optional[dict] = None):
        with self._lock:
            if self.is_training:
                raise RuntimeError("Training already in progress")
            if config:
                for k, v in config.items():
                    if hasattr(self.config, k) and v is not None:
                        setattr(self.config, k, v)
            for g in self.optimizer.param_groups:
                g["lr"] = self.config.learning_rate
            self.is_training = True
            self._stop_flag.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_flag.set()

    def _emit(self, event: dict):
        self.history.append(event)

    # -------------------------------------------------------- train loop
    def _run(self):
        cfg = self.config
        try:
            start_step = self.current_step
            for it in range(start_step, cfg.max_iters):
                if self._stop_flag.is_set():
                    break

                lr = self._get_lr(it)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

                xb, yb = self.get_batch("train")
                _, loss = self.model(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                ).item()
                self.optimizer.step()

                self.current_step = it + 1
                self.tokens_processed += cfg.batch_size * cfg.block_size
                loss_val = loss.item()
                is_diverged = math.isnan(loss_val) or math.isinf(loss_val) or loss_val > 50

                if it % cfg.log_interval == 0 or it == cfg.max_iters - 1 or is_diverged:
                    self._emit({
                        "type": "step",
                        "step": self.current_step,
                        "train_loss": loss_val,
                        "train_ppl": math.exp(min(loss_val, 20)),
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "param_norm": self._param_norm(),
                        "tokens_processed": self.tokens_processed,
                    })

                if is_diverged:
                    self._emit({
                        "type": "error",
                        "message": (
                            f"Training loss diverged (train_loss={loss_val}) "
                            f"at step {self.current_step}. Stopping."
                        ),
                    })
                    break

                if (it % cfg.eval_interval == 0) or (it == cfg.max_iters - 1):
                    losses = self._estimate_loss()
                    sample = self._sample()
                    self._emit({
                        "type": "eval",
                        "step": self.current_step,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "train_ppl": math.exp(min(losses["train"], 20)),
                        "val_ppl": math.exp(min(losses["val"], 20)),
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "param_norm": self._param_norm(),
                        "tokens_processed": self.tokens_processed,
                        "sample": sample,
                    })
        except Exception as exc:  # pragma: no cover - defensive
            self._emit({"type": "error", "message": str(exc)})
        finally:
            self.is_training = False
            self._emit({"type": "done", "step": self.current_step})

    # ------------------------------------------------------------- talk
    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 1.0):
        prompt_ids = self.tokenizer.encode(prompt) or [0]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        self.model.eval()
        out_idx, probs = self.model.generate_with_probs(idx, max_new_tokens, temperature)
        self.model.train()
        generated_ids = out_idx[0].tolist()
        prompt_len = len(prompt_ids)
        prompt_tokens = [{"char": self.tokenizer.itos[i], "prob": None} for i in prompt_ids]
        generated_tokens = [
            {"char": self.tokenizer.itos[i], "prob": p}
            for i, p in zip(generated_ids[prompt_len:], probs)
        ]
        return {
            "text": self.tokenizer.decode(generated_ids),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
        }

    # ----------------------------------------------------------- status
    def status(self):
        return {
            "is_training": self.is_training,
            "current_step": self.current_step,
            "max_iters": self.config.max_iters,
            "vocab_size": self.tokenizer.vocab_size,
            "device": self.device,
            "dataset_chars": len(self.train_data) + len(self.val_data),
            "tokens_processed": self.tokens_processed,
            "config": asdict(self.config),
        }
