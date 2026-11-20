"""Load the bundled model and evaluation data, and measure perplexity."""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "distilgpt2"
DATA_PATH = ROOT / "data" / "eval_texts.jsonl"
SEQ_LEN = 128


def load_model(model_dir: Path = MODEL_DIR) -> GPT2LMHeadModel:
    """Build distilgpt2 from its config and load the bundled weights.

    The weights are stored as fp16 shards (to fit in git) and loaded as fp32.
    """
    model_dir = Path(model_dir)
    model = GPT2LMHeadModel(GPT2Config.from_pretrained(model_dir))
    with open(model_dir / "model.safetensors.index.json", encoding="utf-8") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    state = {}
    for shard in shards:
        state.update({k: v.float() for k, v in load_file(str(model_dir / shard)).items()})
    missing, _unexpected = model.load_state_dict(state, strict=False)
    # lm_head is tied to the token embeddings, so it is never in the file.
    if set(missing) - {"lm_head.weight"}:
        raise RuntimeError(f"weights missing from {model_dir}: {sorted(missing)}")
    model.tie_weights()
    return model


def load_tokenizer(model_dir: Path = MODEL_DIR) -> GPT2TokenizerFast:
    return GPT2TokenizerFast.from_pretrained(Path(model_dir))


def load_eval_ids(tokenizer: GPT2TokenizerFast, path: Path = DATA_PATH) -> torch.Tensor:
    """Every sample is exactly SEQ_LEN tokens, so no padding is needed."""
    with open(path, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
    ids = [tokenizer(t)["input_ids"] for t in texts]
    bad = [i for i, x in enumerate(ids) if len(x) != SEQ_LEN]
    if bad:
        raise ValueError(f"samples {bad} are not {SEQ_LEN} tokens long")
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def perplexity(model: GPT2LMHeadModel, input_ids: torch.Tensor, batch_size: int = 10) -> float:
    """Token-level perplexity over all samples (next-token prediction)."""
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(input_ids), batch_size):
        batch = input_ids[start : start + batch_size]
        loss = model(batch, labels=batch).loss
        # The loss averages over the shifted targets: SEQ_LEN - 1 per sample.
        n = batch.shape[0] * (batch.shape[1] - 1)
        total_nll += loss.item() * n
        total_tokens += n
    return math.exp(total_nll / total_tokens)
