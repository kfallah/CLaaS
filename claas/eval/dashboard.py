"""Eval dashboard: read eval results from disk and render HTML.

Discovers all runs under a parent results directory (each subdirectory
containing a ``summary.json``), renders newest-first as collapsible
sections with embedded plot images.
"""

from __future__ import annotations

import base64
import dataclasses
import html
import json
import logging
import os
from pathlib import Path
from typing import Any

from .types import (
    StepResult,
    TinkerDistillMetrics,
    step_result_from_dict,
)

logger = logging.getLogger(__name__)

EVAL_DASHBOARD_TEMPLATE = Path(__file__).resolve().parent.parent / "dashboard" / "eval_dashboard.html"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_json(directory: str, filename: str) -> dict[str, Any]:
    """Load a JSON file from a directory, returning {} on failure."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}


def _load_steps(run_dir: str, preference: str) -> list[StepResult]:
    """Load step results from JSONL file."""
    path = os.path.join(run_dir, preference, "steps.jsonl")
    if not os.path.exists(path):
        return []
    steps: list[StepResult] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    steps.append(step_result_from_dict(json.loads(line)))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSONL line in %s", path)
    return steps


def _load_baseline(run_dir: str, preference: str) -> dict[str, Any]:
    """Load baseline.json for a preference."""
    return _load_json(os.path.join(run_dir, preference), "baseline.json")


def _discover_runs(results_dir: str) -> list[str]:
    """Find run directories that contain a summary.json.

    Scans child directories of *results_dir* for ``summary.json`` files.
    If *results_dir* itself also contains a ``summary.json`` it is
    included as a run (shown as the directory name).  All runs are
    returned newest-first by summary.json mtime.
    """
    if not os.path.isdir(results_dir):
        return []

    runs: list[tuple[float, str]] = []

    # Check child directories
    for entry in os.listdir(results_dir):
        child = os.path.join(results_dir, entry)
        if os.path.isdir(child) and os.path.isfile(os.path.join(child, "summary.json")):
            mtime = os.path.getmtime(os.path.join(child, "summary.json"))
            runs.append((mtime, child))

    # Include results_dir itself if it has a summary.json (legacy single-run)
    root_summary = os.path.join(results_dir, "summary.json")
    if os.path.isfile(root_summary):
        runs.append((os.path.getmtime(root_summary), results_dir))

    # Newest first
    runs.sort(reverse=True)
    return [path for _, path in runs]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: float | None, precision: int = 3) -> str:
    """Format a float for display, or return '-' for None."""
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


# ---------------------------------------------------------------------------
# Plot embedding
# ---------------------------------------------------------------------------

def _embed_plots(run_dir: str) -> str:
    """Read ``{run_dir}/plots/*.png``, base64-encode, return ``<img>`` grid."""
    plots_dir = os.path.join(run_dir, "plots")
    if not os.path.isdir(plots_dir):
        return ""

    images: list[tuple[str, str]] = []  # (filename, base64 data)
    for fname in sorted(os.listdir(plots_dir)):
        if not fname.lower().endswith(".png"):
            continue
        fpath = os.path.join(plots_dir, fname)
        try:
            with open(fpath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            images.append((fname, b64))
        except OSError as exc:
            logger.warning("Failed to read plot %s: %s", fpath, exc)

    if not images:
        return ""

    img_tags = "\n".join(
        '<div class="plot-item">'
        '<img src="data:image/png;base64,{b64}" alt="{alt}" />'
        "<p>{alt}</p>"
        "</div>".format(b64=b64, alt=html.escape(fname))
        for fname, b64 in images
    )
    return '<div class="plot-grid">{}</div>'.format(img_tags)


# ---------------------------------------------------------------------------
# Summary table (no verdicts)
# ---------------------------------------------------------------------------

def _summary_table_rows(summary: dict[str, Any], run_id: str) -> str:
    """Render summary table rows — raw metrics only, no verdicts."""
    if not summary:
        return '<tr><td colspan="6">No summary data found.</td></tr>'

    rows: list[str] = []
    for pref_name, data in summary.items():
        rows.append(
            """
            <tr>
              <td>{pref}</td>
              <td>{lora_id}</td>
              <td>{margin_delta}</td>
              <td>{compliance}</td>
              <td>{cap_ratio}</td>
              <td><a href="#pref-{run_id}-{pref}">Details</a></td>
            </tr>
            """.format(
                pref=html.escape(pref_name),
                lora_id=html.escape(str(data.get("lora_id", ""))),
                margin_delta=_fmt(data.get("logprob_margin_delta")),
                compliance=_fmt(data.get("final_compliance")),
                cap_ratio=_fmt(data.get("capability_ratio")),
                run_id=html.escape(run_id),
            )
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Step detail rows (unchanged logic, only IDs namespaced by run)
# ---------------------------------------------------------------------------

def _step_detail_rows(steps: list[StepResult], preference: str, run_id: str) -> str:
    """Render step-by-step detail rows for a preference experiment."""
    if not steps:
        return '<tr><td colspan="8">No step data found.</td></tr>'

    rows: list[str] = []
    for step in steps:
        detail_id = f"step-{run_id}-{preference}-{step.step}"
        margin_str = _fmt(step.eval.logprob_margin.margin) if step.eval.logprob_margin else "-"
        delta_str = (
            _fmt(step.eval.logprob_margin.margin_delta_from_baseline)
            if step.eval.logprob_margin
            else "-"
        )
        compliance_str = _fmt(step.eval.preference_compliance)
        if isinstance(step.sdpo_metrics, TinkerDistillMetrics):
            loss_str = _fmt(step.sdpo_metrics.adv_mean)
        elif step.sdpo_metrics:
            loss_str = _fmt(step.sdpo_metrics.distill_loss)
        else:
            loss_str = "-"

        # Summary row
        rows.append(
            """
            <tr>
              <td>{step}</td>
              <td>{timestamp}</td>
              <td>{margin}</td>
              <td>{delta}</td>
              <td>{compliance}</td>
              <td>{loss}</td>
              <td>{timing}</td>
              <td><button type="button" onclick="toggleDetails('{detail_id}', this)">Expand</button></td>
            </tr>
            """.format(
                step=step.step,
                timestamp=html.escape(step.timestamp),
                margin=margin_str,
                delta=delta_str,
                compliance=compliance_str,
                loss=loss_str,
                timing=_fmt(step.timing_s, 1),
                detail_id=detail_id,
            )
        )

        # Expandable detail row
        sections: list[str] = []

        # Feedback
        sections.append(
            "<section><h3>Feedback</h3><pre>{}</pre></section>".format(
                html.escape(step.feedback_given)
            )
        )

        # Prompt
        sections.append(
            "<section><h3>Prompt</h3><pre>{}</pre></section>".format(
                html.escape(step.prompt_used)
            )
        )

        # Response
        if step.response_text:
            sections.append(
                "<section><h3>Response</h3><pre>{}</pre></section>".format(
                    html.escape(step.response_text)
                )
            )

        # Training metrics
        if step.sdpo_metrics:
            metrics_label = (
                "Tinker Distill Metrics"
                if isinstance(step.sdpo_metrics, TinkerDistillMetrics)
                else "Local Distill Metrics"
            )
            metrics_json = json.dumps(
                dataclasses.asdict(step.sdpo_metrics), indent=2, sort_keys=True,
            )
            sections.append(
                "<section><h3>{}</h3><pre>{}</pre></section>".format(
                    metrics_label, html.escape(metrics_json),
                )
            )

        # Eval metrics (excluding rollouts to keep it compact)
        eval_data = dataclasses.asdict(step.eval)
        eval_data.pop("rollouts", None)
        eval_json = json.dumps(eval_data, indent=2, sort_keys=True)
        sections.append(
            "<section><h3>Eval Metrics</h3><pre>{}</pre></section>".format(
                html.escape(eval_json)
            )
        )

        # Rollouts (collapsible)
        if step.eval.rollouts:
            rollout_parts: list[str] = []
            for i, rollout in enumerate(step.eval.rollouts):
                msgs_text = "\n".join(
                    f"[{m.get('role', '?')}] {m.get('content', '')[:200]}"
                    for m in rollout.messages
                )
                meta_json = json.dumps(rollout.metadata, indent=2, sort_keys=True, default=str)
                rollout_parts.append(
                    """
                    <details>
                      <summary>Rollout {idx}: {metric} &mdash; {task}</summary>
                      <pre>{msgs}</pre>
                      <pre>{meta}</pre>
                    </details>
                    """.format(
                        idx=i,
                        metric=html.escape(rollout.metric),
                        task=html.escape(str(rollout.metadata.get("task", ""))),
                        msgs=html.escape(msgs_text),
                        meta=html.escape(meta_json),
                    )
                )
            sections.append(
                "<section><h3>Rollouts ({count})</h3>{rollouts}</section>".format(
                    count=len(step.eval.rollouts),
                    rollouts="\n".join(rollout_parts),
                )
            )

        rows.append(
            """
            <tr id="{detail_id}" class="detail-row">
              <td colspan="8">
                <div class="detail-panel">
                  {sections}
                </div>
              </td>
            </tr>
            """.format(
                detail_id=detail_id,
                sections="\n".join(sections),
            )
        )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Per-preference sections within a run
# ---------------------------------------------------------------------------

def _preference_sections(summary: dict[str, Any], run_dir: str, run_id: str) -> str:
    """Render per-preference detail sections with step tables."""
    if not summary:
        return "<p>No eval results found.</p>"

    sections: list[str] = []
    for pref_name in summary:
        baseline = _load_baseline(run_dir, pref_name)
        steps = _load_steps(run_dir, pref_name)

        baseline_json = json.dumps(baseline, indent=2, sort_keys=True) if baseline else "N/A"

        step_rows = _step_detail_rows(steps, pref_name, run_id)

        sections.append(
            """
            <div id="pref-{run_id}-{pref}" class="pref-section">
              <h3>{pref}</h3>
              <details>
                <summary>Baseline Metrics</summary>
                <pre>{baseline}</pre>
              </details>
              <table>
                <thead>
                  <tr>
                    <th>Step</th>
                    <th>Timestamp</th>
                    <th>Margin</th>
                    <th>Delta</th>
                    <th>Compliance</th>
                    <th>Loss</th>
                    <th>Time (s)</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {step_rows}
                </tbody>
              </table>
            </div>
            """.format(
                run_id=html.escape(run_id),
                pref=html.escape(pref_name),
                baseline=html.escape(baseline_json),
                step_rows=step_rows,
            )
        )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Render a single run
# ---------------------------------------------------------------------------

def _render_run(run_dir: str, run_id: str) -> str:
    """Render one run as a collapsible ``<details>`` block."""
    summary = _load_json(run_dir, "summary.json")
    summary_rows = _summary_table_rows(summary, run_id)
    pref_sections = _preference_sections(summary, run_dir, run_id)
    plots_html = _embed_plots(run_dir)

    return """
    <details class="run-section">
      <summary><strong>{run_id}</strong></summary>
      <table>
        <thead>
          <tr>
            <th>Preference</th>
            <th>LoRA ID</th>
            <th>Margin Delta</th>
            <th>Compliance</th>
            <th>Capability</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {summary_rows}
        </tbody>
      </table>
      {plots_html}
      {pref_sections}
    </details>
    """.format(
        run_id=html.escape(run_id),
        summary_rows=summary_rows,
        plots_html=plots_html,
        pref_sections=pref_sections,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def eval_dashboard_html(
    results_dir: str, *, page: int = 1, per_page: int = 20
) -> str:
    """Discover all runs under *results_dir* and render the dashboard HTML."""
    from ..dashboard.pagination import paginate, render_pagination_nav

    runs = _discover_runs(results_dir)
    total = len(runs)
    info = paginate(total, page, per_page)
    page_runs = runs[info.offset : info.offset + info.per_page]

    if not page_runs:
        content = "<p>No eval runs found in <code>{}</code>.</p>".format(
            html.escape(results_dir),
        )
    else:
        parts: list[str] = []
        for run_dir in page_runs:
            run_id = os.path.basename(run_dir) or results_dir
            parts.append(_render_run(run_dir, run_id))
        content = "\n".join(parts)

    extra_params: dict[str, str] = {}
    if results_dir != "./data/evals":
        extra_params["results_dir"] = results_dir
    nav = render_pagination_nav(info, "/v1/eval", extra_params=extra_params or None)

    template = EVAL_DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("{{CONTENT}}", content).replace("{{PAGINATION}}", nav)
