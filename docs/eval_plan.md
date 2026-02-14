# SDPO Continual Learning Benchmark Plan

## 1. Experiment Design

**Goal:** Demonstrate that SDPO single-step feedback produces measurable preference adaptation without catastrophic forgetting, across 20 sequential feedback rounds per preference type.

**Key insight from the SDPO/SDFT papers:** On-policy self-distillation preserves prior capabilities because the teacher signal is generated *by the current model conditioned on feedback*, so the gradient update stays near the current policy manifold. Our benchmark must verify this property holds in the single-step LoRA regime on real chat preferences.

### Independent Variables

Each preference type is tested in **isolation** (fresh LoRA per type). This gives us clean ablation — one feedback axis per experiment run.

| Preference ID | Feedback String | Positive Probe | Negative Probe |
|---|---|---|---|
| `no_emoji` | "Don't use any emojis in your responses" | Response with 0 emojis | Response with 3+ emojis |
| `concise` | "Be more concise, keep responses under 3 sentences" | Response ≤3 sentences | Response >6 sentences |
| `identity` | "Your name is Kuro, always introduce yourself as Kuro" | Response containing "Kuro" | Response with generic "I'm an AI assistant" |

### Dependent Variables (measured after each of 20 feedback steps)

1. **Preference Compliance Score** (0–1) — does the model follow the feedback?
2. **Collapse Detection** — logprob-based entropy + coding performance
3. **General Capability Score** — simple coding task + instruction following

---

## 2. Metrics: Detailed Specification

### 2.1 Preference Compliance (Primary Signal)

For each preference, we define a **programmatic verifier** (no LLM judge needed — deterministic, fast):

```text
no_emoji:     score = 1.0 if count(emoji_chars, response) == 0 else 0.0
concise:      score = 1.0 if count_sentences(response) <= 3 else max(0, 1 - (n_sentences - 3) / 6)
identity:     score = 1.0 if "kuro" in response.lower() else 0.0
```

**Probe prompts** (3 per preference, rotated each step to avoid memorization):
- Generic chat: "Hey, what's up?" / "Tell me something interesting" / "How are you today?"
- Task-specific: Prompts that naturally tempt the model toward the *wrong* behavior (e.g., for `no_emoji`, "Write an enthusiastic greeting!")

After each feedback step, sample 3 responses from the probe set and average compliance.

### 2.2 Logprob-Based Preference Shift (Dense Signal)

This is the **core quantitative metric** that doesn't require generation — just a forward pass:

For each preference type, define a **(positive_example, negative_example)** pair:

```python
# Example for no_emoji preference:
positive = "Hello! How can I help you today? I'm happy to assist with anything."
negative = "Hello! 😊 How can I help you today? 🎉 I'm happy to assist! ✨"

# Compute:
logprob_positive = sum(token_logprobs(model, prompt + positive))  # via vLLM /v1/completions
logprob_negative = sum(token_logprobs(model, prompt + negative))

preference_margin = logprob_positive - logprob_negative
```

**What to track per step:**
- `preference_margin` (should increase with training — model prefers the positive example more)
- `logprob_positive` (should increase or stay stable)
- `logprob_negative` (should decrease)

This uses your existing `_fetch_rollout_logprobs()` infrastructure — no extra tooling needed.

### 2.3 Collapse Detection

Three fast signals that catch mode collapse or entropy death:

**a) Token-level entropy on a neutral prompt:**
```python
# Use vLLM logprobs endpoint with top_logprobs=20
# Compute: H = -sum(p * log(p)) for top-20 tokens at each position
# Average over response positions
# ALERT if entropy drops >40% from step 0 baseline
```

**b) Response diversity (n=3 samples, temperature=0.7):**
```python
# Generate 3 responses to same prompt
# Compute pairwise ROUGE-L between them
# ALERT if mean self-ROUGE-L > 0.85 (near-identical outputs)
```

**c) Mean logprob magnitude:**
```python
# Track mean logprob across response tokens
# Collapse → logprobs become very negative (model confused) or very peaked (0.0)
# ALERT if |mean_logprob - baseline_mean_logprob| > 2.0
```

### 2.4 General Capability Preservation

One deterministic coding task, evaluated programmatically:

```python
CODING_PROMPT = """Write a Python function called `fibonacci` that takes an integer n 
and returns the nth Fibonacci number. Use iterative approach, not recursive. 
Include a docstring."""

# Verification:
def verify_coding(response: str) -> dict:
    """Extract code, exec it, test it."""
    code = extract_code_block(response)
    namespace = {}
    try:
        exec(code, namespace)
        fn = namespace.get("fibonacci")
        assert fn(0) == 0
        assert fn(1) == 1
        assert fn(10) == 55
        return {"correct": True, "has_docstring": '"""' in code or "'''" in code}
    except:
        return {"correct": False, "has_docstring": False}
```

Plus 3 quick IFEval-style verifiable instructions:
```text
1. "Write exactly 3 sentences about Python." → verify sentence count == 3
2. "List 5 benefits of exercise. Use numbered list." → verify 5 numbered items
3. "Explain recursion without using the word 'function'." → verify 'function' not in response
```

`Score = (coding_correct * 0.5) + (ifeval_pass_rate * 0.5)`

---

## 3. Test Harness Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                    Benchmark Runner                         │
│  (Python script, runs on same L40S)                        │
│                                                            │
│  for pref_type in [no_emoji, concise, identity]:           │
│    POST /v1/lora/init  (fresh LoRA)                        │
│    baseline = run_eval_suite()                             │
│                                                            │
│    for step in range(20):                                  │
│      1. Generate chat via OpenClaw API                     │
│         (Gemini acts as user with hidden preference)       │
│      2. Gemini provides feedback string                    │
│      3. POST /v1/feedback  (SDPO gradient step)            │
│      4. run_eval_suite()   (~60s per suite)                │
│         ├─ preference_compliance (3 probes, temp=0)        │
│         ├─ logprob_margins (2 fwd passes via vLLM)         │
│         ├─ collapse_checks (entropy + diversity)           │
│         └─ general_capability (coding + IFEval)            │
│      5. Log all metrics to JSON                            │
│                                                            │
│  Total: 3 prefs × 21 evals × ~60s ≈ 63 min (unbatched)   │
│  With batching: ~24 min (see timing budget below)          │
└────────────────────────────────────────────────────────────┘
```

### Gemini Judge/User Configuration

Gemini plays two roles:

**Role 1 — Simulated User** (generates the chat that triggers feedback):
```text
System: You are testing an AI chatbot. Have a natural conversation about coding, 
daily life, or general knowledge. Keep messages short (1-2 sentences). 
Your hidden preference: {PREFERENCE_DESCRIPTION}. 
After the chatbot responds, evaluate whether it followed your preference.
Reply with ONLY a JSON: {"satisfied": bool, "feedback": "string or null"}
If not satisfied, feedback should be a natural rephrasing of your preference.
```

**Role 2 — is not needed.** The compliance verifier is deterministic (§2.1), and the Gemini user naturally varies its feedback phrasing across steps, which tests robustness to paraphrase.

### Timing Budget (30-minute target)

The bottleneck is the vLLM sleep/wake cycle per feedback step:
- sleep + verify GPU: ~5s
- distill (1 SDPO step): ~3s  
- wake + reload LoRA: ~8s
- Total per feedback: ~16s

Eval suite per step:
- 3 compliance probes (generation): ~6s
- 2 logprob forward passes: ~2s
- 3 diversity samples: ~6s
- coding + IFEval (4 generations): ~8s
- Gemini API call: ~2s
- Total per eval: ~24s

Per preference: 20 steps × (16s feedback + 24s eval) = **~13 min**

**3 preferences × 13 min = ~39 min total.** To hit 30 min:

- Batch all probe prompts into a single vLLM request per step
- Skip diversity check every other step (steps 0, 5, 10, 15, 20 only)
- Pre-tokenize positive/negative examples once

With batching: ~8s per eval → 20 × (16 + 8) = **~8 min per preference, ~24 min for all 3**.

---

## 4. Output Format & Visualization

### Per-Step JSON Log

```json
{
  "preference": "no_emoji",
  "step": 5,
  "timestamp": "2026-02-13T10:30:00Z",
  "feedback_given": "Please stop using emojis in your replies",
  "sdpo_metrics": {
    "distill_loss": 0.023,
    "kl_reg": 0.0015,
    "mean_is_ratio": 1.02,
    "clip_fraction": 0.0
  },
  "eval": {
    "preference_compliance": 0.67,
    "logprob_margin": {
      "positive_logprob": -45.2,
      "negative_logprob": -52.1,
      "margin": 6.9,
      "margin_delta_from_baseline": 3.2
    },
    "collapse": {
      "mean_entropy": 4.1,
      "entropy_ratio_to_baseline": 0.95,
      "self_rouge_l": 0.42,
      "mean_logprob_drift": 0.3,
      "alert": false
    },
    "general": {
      "coding_correct": true,
      "coding_has_docstring": true,
      "ifeval_pass_rate": 1.0,
      "general_score": 1.0
    }
  }
}
```

### Summary Plots (generated at end of run)

1. **Learning curve:** `preference_compliance` (y) vs `step` (x) — should show monotonic increase, ideally reaching >0.8 by step 10–15.

2. **Logprob margin curve:** `margin` (y) vs `step` (x) — dense signal, should correlate with compliance but be smoother.

3. **Collapse dashboard:** 3 subplots showing entropy ratio, self-ROUGE-L, and logprob drift. All should stay within alert thresholds (shaded green zone).

4. **Forgetting plot:** `general_score` (y) vs `step` (x) — should remain flat. This is the key SDPO selling point.

5. **SDPO diagnostics:** `kl_reg` and `distill_loss` over steps — confirm training is working (loss should decrease, KL should stay small).

---

## 5. Success Criteria

| Metric | Pass | Marginal | Fail |
|--------|------|----------|------|
| Preference compliance at step 20 | ≥0.8 | 0.5–0.8 | <0.5 |
| Logprob margin increase over baseline | >2.0 nats | 0.5–2.0 | <0.5 |
| General capability retention | >0.9× baseline | 0.7–0.9× | <0.7× |
| Entropy ratio to baseline | >0.6 | 0.4–0.6 | <0.4 |
| Self-ROUGE-L | <0.85 | 0.85–0.95 | >0.95 |

**An experiment "passes" if:** preference compliance ≥0.8 AND general capability >0.9× baseline AND no collapse alerts.

---

## 6. Implementation Priority

1. **Phase 1 (MVP, ~1 day):** Logprob margin tracking only. No generation needed — just forward passes via vLLM `/v1/completions` with `max_tokens=1, prompt_logprobs=1`. This reuses `_fetch_rollout_logprobs()`. Wire it into a loop that does: init LoRA → 20× (POST /v1/feedback with hardcoded feedback string + measure margin). Skip Gemini entirely. Run all 3 preferences.

2. **Phase 2 (~1 day):** Add generative eval (compliance probes + coding task). Add the Gemini simulated user for natural feedback variation.

3. **Phase 3 (~0.5 day):** Collapse detection, IFEval checks, plotting, JSON logging.

4. **Phase 4 (optional):** Ablation on SDPO hyperparameters (alpha, kl_reg_weight, learning rate) to find optimal single-step config. Compare against naive SFT baseline (replace SDPO loss with cross-entropy on feedback-conditioned teacher output).
