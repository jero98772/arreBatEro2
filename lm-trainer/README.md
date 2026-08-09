# Tiny Shakespeare — Char LM Trainer

A small FastAPI app around the bigram character-level language model from
Andrej Karpathy's *"Let's build GPT"* tutorial, with two interfaces:

- **Talk** — prompt the current model and watch it generate text one
  character at a time, each character tinted by how confident the model
  was in that choice (brick = unsure → gold → sage = confident).
- **Train** — kick off a training run and watch it live: loss, validation
  loss, train/val perplexity, learning rate, gradient norm, parameter norm,
  and tokens processed, all streamed in real time over SSE, plus a live
  sample generation and automatic warnings for loss divergence, gradient
  spikes, and validation-loss-rising-while-train-loss-falls (overfitting).

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Internet access on first run only, to auto-download the Tiny Shakespeare
  dataset (~1.1MB) from GitHub. If you're offline, download it yourself from
  `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
  and save it to `app/data/input.txt`.

## Run it

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000**.

## How it's organized

```
app/
  main.py         FastAPI routes (pages, /api/generate, /api/train/*)
  model.py        Tokenizer + BigramLanguageModel + dataset download
  training.py     TrainingManager: background training loop + metrics
  templates/      index.html (Talk + Train tabs)
  static/         style.css, app.js (charts, SSE handling, token coloring)
  data/           input.txt (downloaded automatically)
```

The model trains in a background thread so the server stays responsive.
Metrics are appended to an in-memory list and streamed to the browser over
Server-Sent Events (`/api/train/stream`), which replays the full run on
connect so a page refresh mid-training doesn't lose your charts.

## Notes

- The bigram model is intentionally tiny (it predicts the next character
  from only the previous one), so it trains fast but its ceiling is low —
  expect recognizably-Shakespeare-ish *shapes* (capitalized names, line
  breaks, some real short words) rather than coherent sentences. The whole
  training/metrics/UI scaffolding here is written to be model-agnostic, so
  you can swap `BigramLanguageModel` in `app/model.py` for something
  bigger (e.g. a small self-attention Transformer) without touching the
  rest of the app.
- "Stop" pauses the run; "Start" again resumes from the current step using
  the same model weights. "Reset model" reinitializes the model and clears
  all metrics history.
