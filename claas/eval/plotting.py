"""Summary plots for evaluation results.

Generates 5 plots saved to {output_dir}/plots/.
Gracefully skips plots for metrics not collected in the current run.
"""

from __future__ import annotations

import json
import logging
import os

from .types import StepResult, step_result_from_dict

logger = logging.getLogger(__name__)


def _load_steps(output_dir: str, preference: str) -> list[StepResult]:
    """Load step results from JSONL file."""
    path = os.path.join(output_dir, preference, "steps.jsonl")
    if not os.path.exists(path):
        return []
    steps: list[StepResult] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(step_result_from_dict(json.loads(line)))
    return steps


def generate_plots(output_dir: str, preferences: list[str]) -> None:
    """Generate all summary plots. Requires matplotlib."""
    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    all_steps: dict[str, list[StepResult]] = {}
    for pref in preferences:
        steps = _load_steps(output_dir, pref)
        if steps:
            all_steps[pref] = steps

    if not all_steps:
        logger.warning("No step data found — skipping plots")
        return

    _plot_logprob_margins(all_steps, plots_dir, plt)
    _plot_learning_curves(all_steps, plots_dir, plt)
    _plot_collapse_dashboard(all_steps, plots_dir, plt)
    _plot_forgetting(all_steps, plots_dir, plt)
    _plot_sdpo_diagnostics(all_steps, plots_dir, plt)

    logger.info("Plots saved to %s", plots_dir)


def _plot_logprob_margins(
    all_steps: dict[str, list[StepResult]], plots_dir: str, plt: object,
) -> None:
    """Plot logprob margin vs step."""
    fig, ax = plt.subplots(figsize=(10, 6))  # type: ignore[union-attr]
    has_data = False

    for pref, steps in all_steps.items():
        x, y = [], []
        for s in steps:
            if s.eval.logprob_margin:
                x.append(s.step)
                y.append(s.eval.logprob_margin.margin)
        if x:
            ax.plot(x, y, marker="o", label=pref)
            has_data = True

    if not has_data:
        plt.close(fig)  # type: ignore[union-attr]
        return

    ax.set_xlabel("Step")
    ax.set_ylabel("Logprob Margin (positive - negative)")
    ax.set_title("Logprob Margin Over Training Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "logprob_margins.png"), dpi=150)
    plt.close(fig)  # type: ignore[union-attr]


def _plot_learning_curves(
    all_steps: dict[str, list[StepResult]], plots_dir: str, plt: object,
) -> None:
    """Plot preference compliance vs step."""
    fig, ax = plt.subplots(figsize=(10, 6))  # type: ignore[union-attr]
    has_data = False

    for pref, steps in all_steps.items():
        x, y = [], []
        for s in steps:
            if s.eval.preference_compliance is not None:
                x.append(s.step)
                y.append(s.eval.preference_compliance)
        if x:
            ax.plot(x, y, marker="o", label=pref)
            has_data = True

    if not has_data:
        plt.close(fig)  # type: ignore[union-attr]
        return

    ax.set_xlabel("Step")
    ax.set_ylabel("Preference Compliance")
    ax.set_title("Learning Curves: Preference Compliance Over Steps")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "learning_curves.png"), dpi=150)
    plt.close(fig)  # type: ignore[union-attr]


def _plot_collapse_dashboard(
    all_steps: dict[str, list[StepResult]], plots_dir: str, plt: object,
) -> None:
    """Plot collapse metrics: entropy ratio, self-ROUGE-L, logprob drift."""
    has_data = False
    for steps in all_steps.values():
        for s in steps:
            if s.eval.collapse:
                has_data = True
                break
        if has_data:
            break

    if not has_data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # type: ignore[union-attr]

    for pref, steps in all_steps.items():
        x_ent, y_ent = [], []
        x_rouge, y_rouge = [], []
        x_drift, y_drift = [], []

        for s in steps:
            collapse = s.eval.collapse
            if not collapse:
                continue
            x_ent.append(s.step)
            y_ent.append(collapse.entropy_ratio_to_baseline)
            x_rouge.append(s.step)
            y_rouge.append(collapse.self_rouge_l)
            x_drift.append(s.step)
            y_drift.append(collapse.mean_logprob_drift)

        if x_ent:
            axes[0].plot(x_ent, y_ent, marker="o", label=pref)
        if x_rouge:
            axes[1].plot(x_rouge, y_rouge, marker="o", label=pref)
        if x_drift:
            axes[2].plot(x_drift, y_drift, marker="o", label=pref)

    # Entropy ratio
    axes[0].set_title("Entropy Ratio to Baseline")
    axes[0].set_xlabel("Step")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Self-ROUGE-L
    axes[1].set_title("Self-ROUGE-L")
    axes[1].set_xlabel("Step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Logprob drift
    axes[2].set_title("Logprob Drift from Baseline")
    axes[2].set_xlabel("Step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Collapse Detection Dashboard")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "collapse_dashboard.png"), dpi=150)
    plt.close(fig)  # type: ignore[union-attr]


def _plot_forgetting(
    all_steps: dict[str, list[StepResult]], plots_dir: str, plt: object,
) -> None:
    """Plot general capability score vs step."""
    fig, ax = plt.subplots(figsize=(10, 6))  # type: ignore[union-attr]
    has_data = False

    for pref, steps in all_steps.items():
        x, y = [], []
        for s in steps:
            if s.eval.general:
                x.append(s.step)
                y.append(s.eval.general.general_score)
        if x:
            ax.plot(x, y, marker="o", label=pref)
            has_data = True

    if not has_data:
        plt.close(fig)  # type: ignore[union-attr]
        return

    ax.set_xlabel("Step")
    ax.set_ylabel("General Capability Score")
    ax.set_title("Forgetting: General Capability Over Training Steps")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "forgetting.png"), dpi=150)
    plt.close(fig)  # type: ignore[union-attr]


def _plot_sdpo_diagnostics(
    all_steps: dict[str, list[StepResult]], plots_dir: str, plt: object,
) -> None:
    """Plot SDPO training metrics (distill_loss, kl_reg) over steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # type: ignore[union-attr]
    has_data = False

    for pref, steps in all_steps.items():
        x_loss, y_loss = [], []
        x_kl, y_kl = [], []

        for s in steps:
            sdpo = s.sdpo_metrics
            if not sdpo:
                continue
            if sdpo.distill_loss is not None:
                x_loss.append(s.step)
                y_loss.append(sdpo.distill_loss)
                has_data = True
            if sdpo.kl_reg is not None:
                x_kl.append(s.step)
                y_kl.append(sdpo.kl_reg)
                has_data = True

        if x_loss:
            ax1.plot(x_loss, y_loss, marker="o", label=pref)
        if x_kl:
            ax2.plot(x_kl, y_kl, marker="o", label=pref)

    if not has_data:
        plt.close(fig)  # type: ignore[union-attr]
        return

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Distillation Loss")
    ax1.set_title("SDPO Distillation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("KL Regularization")
    ax2.set_title("KL Regularization Term")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("SDPO Training Diagnostics")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "sdpo_diagnostics.png"), dpi=150)
    plt.close(fig)  # type: ignore[union-attr]
