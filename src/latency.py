"""CPU latency measurement. Pre-validated: do not modify.

Mirrors the production setup: 4 CPU threads, no GPU, batch of 1, 128 tokens.
"""

import statistics
import time

import torch


@torch.no_grad()
def median_latency_ms(model, input_ids: torch.Tensor, warmup: int = 3, runs: int = 15, threads: int = 4) -> float:
    """Median wall-clock milliseconds for one forward pass over a single sample."""
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    was_training = model.training
    model.eval()
    try:
        sample = input_ids[:1]
        for _ in range(warmup):
            model(sample)
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000)
        return statistics.median(timings)
    finally:
        model.train(was_training)
        torch.set_num_threads(previous)
