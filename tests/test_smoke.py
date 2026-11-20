"""Smoke tests: the pipeline loads and runs end to end."""

import math

import torch

from src.compress import LowRankLinear, block_params, compress_model
from src.eval import load_eval_ids, load_model, load_tokenizer, perplexity


def test_data_loads():
    ids = load_eval_ids(load_tokenizer())
    assert ids.shape == (50, 128)


def test_perplexity_is_finite():
    model = load_model()
    ids = load_eval_ids(load_tokenizer())[:5]
    assert math.isfinite(perplexity(model, ids))


def test_compress_runs():
    model = compress_model(load_model(), 0.5)
    assert isinstance(model.transformer.h[0].mlp.c_fc, LowRankLinear)
    out = model(torch.zeros(1, 8, dtype=torch.long)).logits
    assert out.shape == (1, 8, model.config.vocab_size)
    assert block_params(model) > 0
