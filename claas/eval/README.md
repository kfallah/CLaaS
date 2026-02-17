# CLaaS Eval Harness

Automated evaluation for SDPO continual learning. Runs feedback loops against a live CLaaS stack and measures whether training shifts the model toward preferred behaviours without collapsing.

## Running

Via the CLI:

```bash
claas eval \
    --openclaw-url http://localhost:18789 \
    --proxy-url http://localhost:8000 \
    --preferences no_emoji \
    --metrics logprob \
    --num-steps 10
```

Or as a module:

```bash
python -m claas.eval --preferences no_emoji --metrics all --num-steps 20
```

Results are written to `--output-dir` (default `./eval_results`). View them in the browser:

```http
GET /v1/eval?results_dir=./eval_results
```

## Metrics

Select metrics with `--metrics` (comma-separated or a preset).

| Metric | Preset | Needs generation | What it measures |
|---|---|---|---|
| `logprob` | `quick`, `all` | No | Logprob margin between preferred/dispreferred response pairs. Positive margin = model favours the preferred response. Delta from baseline tracks training progress. |
| `compliance` | `all` | Yes | Generates responses to probe prompts, runs a programmatic verifier (e.g. emoji count, sentence count, keyword presence), and averages the pass rate. |
| `general` | `all` | Yes | Coding task (fibonacci, exec + verify) + 3 IFEval-style instruction-following probes. Measures capability retention during training. |
| `collapse` | `all` | Yes | Three collapse detectors: **token entropy** (distribution confidence), **self-ROUGE-L** (output diversity across stochastic samples), and **logprob drift** (mean logprob shift from baseline). |

Presets: `all` = logprob,compliance,general,collapse. `quick` = logprob.

### Collapse thresholds

| Signal | Alert threshold | What it means |
|---|---|---|
| Entropy ratio | < 0.6 | Token distribution > 40% narrower than baseline |
| Self-ROUGE-L | > 0.85 | Stochastic samples are nearly identical |
| Logprob drift | > 2.0 nats | ~7x change in mean token probability from baseline |

### Verifiers (used by `compliance`)

| Verifier | Preference | Pass condition |
|---|---|---|
| `no_emoji` | no_emoji | Zero emoji characters in response |
| `concise` | concise | <= 3 sentences (linear decay to 0.0 at 9+) |
| `identity` | identity | "kuro" appears in response (case-insensitive) |

## Output format

```text
eval_results/
├── summary.json              # Per-preference pass/fail verdicts
└── <preference>/
    ├── metadata.json          # Run config + LoRA ID
    ├── baseline.json          # Pre-training metric snapshot
    └── steps.jsonl            # One JSON object per feedback step
```

Each line in `steps.jsonl` contains: step number, timestamp, feedback given, SDPO training metrics, eval metrics (logprob margin, compliance, general capability, collapse), and rollout transcripts.
