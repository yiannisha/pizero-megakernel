# B200 BF16 baseline run

Measured on 2026-10-15 with NVIDIA B200, PyTorch 2.8.0 + CUDA 12.8,
Triton 3.4.0, and the upstream revisions recorded in [b200.json](b200.json).

Batch 1, three 224×224 views, zero language tokens, chunk 50, action dimension
32, ten flow steps. Five configured warmups and 100 timed iterations per backend.
Each backend ran in a fresh process. Inputs and weights are synthetic; OpenPI
and Triton have independent initialized weights. These results establish latency
baselines, not pretrained policy accuracy or cross-backend numerical parity.

| Backend | Median wall ms | Median CUDA-event ms |
| --- | ---: | ---: |
| openpi-eager | 154.482 | 154.458 |
| openpi-compiled | 57.573 | 57.548 |
| openpi-cudagraph | 39.021 | 39.003 |
| realtime-triton | 21.233 | 21.214 |

Eager and CUDA graph outputs matched the original OpenPI sampling path exactly.
Inductor captured one graph with no graph breaks. Compiled output had 0.845%
normalized RMSE and cosine 0.999965 relative to eager, passing the documented
BF16 smoke-test limits. It failed the stricter pointwise `rtol=0.02, atol=0.02`
comparison, which is retained in the JSON. Default Inductor fusion/precision
settings remain enabled. B200 used realtime-vla's general Triton path; its SM120
specialization was not selected.

Reproduce from the repository root after setup:

```bash
.venv/bin/python benchmarks/run.py --backend all --warmup 5 --iterations 100 --output results/b200.json
```

See the root [README](../README.md) for timing scope, precision policies,
checkpoint support, numerical gates, and shape controls. The JSON includes
individual samples, environment metadata, validation metrics, and source hashes.
