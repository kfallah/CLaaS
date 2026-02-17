"""Eval dashboard: read eval results from disk and render HTML."""

from __future__ import annotations

import dataclasses
import html
import json
import logging
import os
from pathlib import Path
from typing import Any

from .types import (
    StepResult,
    step_result_from_dict,
)

logger = logging.getLogger(__name__)

EVAL_DASHBOARD_TEMPLATE = Path(__file__).resolve().parent.parent / "eval_dashboard.html"


def _load_json(results_dir: str, filename: str) -> dict[str, Any]:
    """Load a JSON file from the results directory, returning {} on failure."""
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_steps(results_dir: str, preference: str) -> list[StepResult]:
    """Load step results from JSONL file."""
    path = os.path.join(results_dir, preference, "steps.jsonl")
    if not os.path.exists(path):
        return []
    steps: list[StepResult] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(step_result_from_dict(json.loads(line)))
    return steps


def _load_baseline(results_dir: str, preference: str) -> dict[str, Any]:
    """Load baseline.json for a preference."""
    return _load_json(os.path.join(results_dir, preference), "baseline.json")


def _verdict_class(verdict: str | None) -> str:
    """Map verdict string to a CSS class name."""
    if verdict == "pass":
        return "verdict-pass"
    if verdict == "fail":
        return "verdict-fail"
    if verdict == "marginal":
        return "verdict-marginal"
    return ""


def _fmt(value: float | None, precision: int = 3) -> str:
    """Format a float for display, or return '-' for None."""
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _summary_table_rows(summary: dict[str, Any]) -> str:
    """Render the top-level summary table rows from summary.json data."""
    if not summary:
        return '<tr><td colspan="8">No summary data found.</td></tr>'

    rows: list[str] = []
    for pref_name, data in summary.items():
        criteria = data.get("criteria", {})
        overall = data.get("overall", "pending")
        overall_cls = _verdict_class(overall)

        criteria_parts: list[str] = []
        for key in [
            "logprob_margin_increase",
            "preference_compliance",
            "capability_retention",
            "entropy_ratio",
            "self_rouge_l",
        ]:
            v = criteria.get(key)
            if v is not None:
                label = key.replace("_", " ").title()
                cls = _verdict_class(v)
                criteria_parts.append(
                    '<span class="{cls}">{label}: {v}</span>'.format(
                        cls=html.escape(cls),
                        label=html.escape(label),
                        v=html.escape(v),
                    )
                )

        rows.append(
            """
            <tr>
              <td>{pref}</td>
              <td>{lora_id}</td>
              <td class="{overall_cls}">{overall}</td>
              <td>{margin_delta}</td>
              <td>{compliance}</td>
              <td>{cap_ratio}</td>
              <td>{criteria_html}</td>
              <td><a href="#pref-{pref}">Details</a></td>
            </tr>
            """.format(
                pref=html.escape(pref_name),
                lora_id=html.escape(data.get("lora_id", "")),
                overall_cls=html.escape(overall_cls),
                overall=html.escape(overall),
                margin_delta=_fmt(data.get("logprob_margin_delta")),
                compliance=_fmt(data.get("final_compliance")),
                cap_ratio=_fmt(data.get("capability_ratio")),
                criteria_html=" &bull; ".join(criteria_parts) if criteria_parts else "-",
            )
        )

    return "\n".join(rows)


def _step_detail_rows(steps: list[StepResult], preference: str) -> str:
    """Render step-by-step detail rows for a preference experiment."""
    if not steps:
        return '<tr><td colspan="8">No step data found.</td></tr>'

    rows: list[str] = []
    for step in steps:
        detail_id = f"step-{preference}-{step.step}"
        margin_str = _fmt(step.eval.logprob_margin.margin) if step.eval.logprob_margin else "-"
        delta_str = (
            _fmt(step.eval.logprob_margin.margin_delta_from_baseline)
            if step.eval.logprob_margin
            else "-"
        )
        compliance_str = _fmt(step.eval.preference_compliance)
        loss_str = _fmt(step.sdpo_metrics.distill_loss) if step.sdpo_metrics else "-"

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

        # SDPO metrics
        if step.sdpo_metrics:
            sdpo_json = json.dumps(dataclasses.asdict(step.sdpo_metrics), indent=2, sort_keys=True)
            sections.append(
                "<section><h3>SDPO Metrics</h3><pre>{}</pre></section>".format(
                    html.escape(sdpo_json)
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


def _preference_sections(summary: dict[str, Any], results_dir: str) -> str:
    """Render per-preference detail sections with step tables."""
    if not summary:
        return "<p>No eval results found.</p>"

    sections: list[str] = []
    for pref_name in summary:
        baseline = _load_baseline(results_dir, pref_name)
        steps = _load_steps(results_dir, pref_name)

        baseline_json = json.dumps(baseline, indent=2, sort_keys=True) if baseline else "N/A"

        step_rows = _step_detail_rows(steps, pref_name)

        sections.append(
            """
            <div id="pref-{pref}" class="pref-section">
              <h2>{pref}</h2>
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
                pref=html.escape(pref_name),
                baseline=html.escape(baseline_json),
                step_rows=step_rows,
            )
        )

    return "\n".join(sections)


def eval_dashboard_html(results_dir: str) -> str:
    """Read eval JSONs, generate HTML, substitute into template."""
    summary = _load_json(results_dir, "summary.json")
    summary_rows = _summary_table_rows(summary)
    pref_sections = _preference_sections(summary, results_dir)

    content = """
    <h2>Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Preference</th>
          <th>LoRA ID</th>
          <th>Overall</th>
          <th>Margin Delta</th>
          <th>Compliance</th>
          <th>Capability</th>
          <th>Criteria</th>
          <th>Details</th>
        </tr>
      </thead>
      <tbody>
        {summary_rows}
      </tbody>
    </table>
    {pref_sections}
    """.format(
        summary_rows=summary_rows,
        pref_sections=pref_sections,
    )

    template = EVAL_DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("{{CONTENT}}", content)
