# CLaaS Eval Harness

Automated evaluation for SDPO continual learning. Runs feedback loops against a live CLaaS stack and measures whether training shifts the model toward preferred behaviours without collapsing.

## Configuration (Hydra)

The eval harness uses [Hydra](https://hydra.cc/) for configuration. The default config lives in `configs/base.yaml` and can be overridden via CLI arguments using Hydra's `key=value` syntax.

### Config file: `configs/base.yaml`

```yaml
mode: tinker                         # execution backend: local | tinker | modal
claas_url: http://localhost:8080     # CLaaS API endpoint
vllm_url: http://localhost:8000      # vLLM endpoint (auto-set to claas_url in tinker mode)
vllm_model_name: qwen3-8b           # model identifier for vLLM API calls
base_model: Qwen/Qwen3-30B-A3B      # base model for LoRA init (Tinker name)

preferences:                         # preferences to train
  - no_emoji
  - concise
  - identity

metrics:                             # metrics to evaluate per step
  - logprob
  - compliance
  - general
  - collapse

num_steps: 20
batch_size: 4
steps_per_batch: 1                   # gradient updates per batch
feedback_repetitions: 1              # times to repeat feedback string
collapse_steps: [0, 5, 10, 15, 19]  # steps where collapse metric runs
plots: true                          # generate matplotlib plots
seed: 42
lora_id_prefix: eval
output_dir: ./data/evals

openclaw_url: null                   # OpenClaw gateway (null = use vllm_url directly)
```

### Overriding config via CLI

Hydra overrides are positional arguments after `uv run python -m claas.eval`:

```bash
# Run only conciseness for 10 steps
uv run python -m claas.eval 'preferences=[concise]' num_steps=10

# Override base model and mode
uv run python -m claas.eval base_model=Qwen/Qwen3-30B-A3B mode=tinker

# Use a custom config directory
uv run python -m claas.eval --config-dir ./my_configs --config-name my_config
```

### Programmatic usage

```python
from claas.eval.config import build_harness_config
from claas.eval.runner import run_harness
from claas.eval.types import EvalConfig
import asyncio

config = build_harness_config(
    EvalConfig(
        preferences=["concise"],
        num_steps=5,
    )
)
asyncio.run(run_harness(config))
```

### Environment variables (secrets)

Secrets are resolved from env vars at runtime, NOT stored in config:

| Variable | Required for | Purpose |
|---|---|---|
| `CLAAS_TINKER_API_KEY` | Tinker mode | Tinker SDK authentication |
| `VLLM_API_KEY` | Local mode | vLLM server auth token |
| `GEMINI_API_KEY` | `general` metric | Gemini-based capability evaluation |

## Running (Tinker mode, no GPU)

### 1. Install dependencies

```bash
uv sync --extra tinker --extra dev
```

### 2. Start the CLaaS API

```bash
CLAAS_TINKER_API_KEY="tml-..." \
  uv run python -m claas.api --config-name tinker
```

### 3. Run the eval

```bash
CLAAS_TINKER_API_KEY="tml-..." \
  uv run python -m claas.eval 'preferences=[concise]' num_steps=20
```

## Known gotchas

**Tinker model naming**: Tinker uses its own model identifiers that differ from HuggingFace names. For example, the HuggingFace model `Qwen/Qwen3-Coder-30B-A3B-Instruct` is `Qwen/Qwen3-30B-A3B` in Tinker. Sampling will work with either name, but LoRA training init will reject the HuggingFace name with a 400 error. Always use the Tinker name in `base_model`.

**API entry point**: Run the API via Hydra (`python -m claas.api --config-name ...`) instead of loading `claas.api:web_app` directly.

**Collapse metric is slow**: The `collapse` metric generates multiple stochastic samples per step. It only runs at steps listed in `collapse_steps` (default `[0, 5, 10, 15, 19]`) to limit overhead.

## Metrics

Select metrics with the `metrics` list in config or via override.

| Metric | What it measures |
|---|---|
| `logprob` | Logprob margin between preferred/dispreferred response pairs. Positive margin = model favours the preferred response. Delta from baseline tracks training progress. |
| `compliance` | Generates responses to probe prompts, runs a programmatic verifier (e.g. emoji count, sentence count, keyword presence), and averages the pass rate. |
| `general` | Coding task (fibonacci, exec + verify) + 3 IFEval-style instruction-following probes. Measures capability retention during training. |
| `collapse` | Three collapse detectors: **token entropy** (distribution confidence), **self-ROUGE-L** (output diversity across stochastic samples), and **logprob drift** (mean logprob shift from baseline). |

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
data/evals/<run-id>/
├── summary.json              # Per-preference pass/fail verdicts
└── <preference>/
    ├── metadata.json          # Run config + LoRA ID
    ├── baseline.json          # Pre-training metric snapshot
    └── steps.jsonl            # One JSON object per feedback step
```

Each line in `steps.jsonl` contains: step number, timestamp, feedback given, SDPO training metrics, eval metrics (logprob margin, compliance, general capability, collapse), and rollout transcripts.

Results can be viewed in the browser at `GET /v1/eval?results_dir=./data/evals`.
