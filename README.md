# compress-eval

Experiments for shrinking NordLogix's GPT-2-class model (distilgpt2) so it is
cheaper to serve on 4-core CPU servers.

Dev's open PR (`pr/PR_DESCRIPTION.md`) replaces the MLP layers of each
transformer block with a truncated-SVD low-rank factorization.

## Layout

```
data/eval_texts.jsonl   50 evaluation samples, exactly 128 tokens each (no padding)
models/distilgpt2/      bundled weights (no internet needed)
src/eval.py             model + data loading, perplexity
src/compress.py         SVD factorization of the MLP layers, parameter counting
src/latency.py          CPU latency helper (pre-validated, do not modify)
src/run_experiment.py   sweep CLI
tests/                  pytest
pr/PR_DESCRIPTION.md    Dev's PR
```

## Run

```bash
# baseline (rank ratio 1.0 = the model as shipped) plus two ratios
python -m src.run_experiment --rank-ratio 1.0 0.7 0.5 --out results/sweep.csv

# add CPU latency (median of 15 runs, 4 threads)
python -m src.run_experiment --rank-ratio 1.0 0.7 --latency

python -m pytest -q
```

Each experiment runs in under 15 seconds on CPU.

The rank ratio is the kept rank as a fraction of the largest possible rank,
`min(in_features, out_features)`, for each factorized layer. It is not the
fraction of weights kept.
