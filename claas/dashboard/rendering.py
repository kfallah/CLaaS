"""Feedback dashboard HTML rendering.

Builds the dashboard table rows and final HTML from feedback log records.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from claas.core.types import FeedbackLogRecord

FEEDBACK_DASHBOARD_TEMPLATE = Path(__file__).resolve().parent / "feedback_dashboard.html"


def feedback_prompt_preview(prompt: str, limit: int = 140) -> str:
    """Build a single-line prompt preview for table display.

    Args:
        prompt: Full prompt text.
        limit: Maximum preview length.

    Returns:
        Prompt preview trimmed to the requested length.
    """
    normalized = " ".join(prompt.splitlines())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}…"


def feedback_dashboard_rows(records: list[FeedbackLogRecord]) -> str:
    """Render dashboard table rows for feedback records, grouped by batch.

    Each batch gets a single summary row.  Expanding it reveals per-sample
    details (prompt / response / feedback) plus the batch-level timing,
    training metrics, and orchestration info.
    """
    rows: list[str] = []
    for idx, record in enumerate(records):
        metrics_payload: dict[str, object] = {}
        if record.distill_result is not None:
            metrics_payload = record.distill_result.metadata
        timing_json = json.dumps(record.timing_ms.model_dump(mode="json"), indent=2, sort_keys=True)
        metrics_json = json.dumps(metrics_payload, indent=2, sort_keys=True)
        vllm_json = json.dumps(record.vllm.model_dump(mode="json"), indent=2, sort_keys=True)
        error_value = record.error or ""
        batch_size = len(record.batch_samples)
        detail_row_id = f"feedback-detail-{idx}"

        # -- Batch summary row --
        rows.append(
            """
            <tr>
              <td>{request_id}<br><small>{timestamp}</small></td>
              <td>{status} ({phase})</td>
              <td>{lora_id}</td>
              <td>{batch_size} sample{plural}</td>
              <td>{distill_ms}</td>
              <td>{total_ms}</td>
              <td><button type="button" onclick="toggleDetails('{detail_row_id}', this)">Expand</button></td>
            </tr>
            """.format(
                request_id=html.escape(record.request_id),
                timestamp=html.escape(record.timestamp_utc),
                status=html.escape(record.status),
                phase=html.escape(record.phase),
                lora_id=html.escape(record.lora_id),
                batch_size=batch_size,
                plural="s" if batch_size != 1 else "",
                distill_ms=record.timing_ms.distill,
                total_ms=record.timing_ms.total,
                detail_row_id=detail_row_id,
            )
        )

        # -- Expandable detail row --
        sample_sections: list[str] = []
        _raw_texts = metrics_payload.get("teacher_scored_texts") if metrics_payload else None
        teacher_scored_texts = list(_raw_texts) if isinstance(_raw_texts, list) else []
        for item_index, sample in enumerate(record.batch_samples):
            teacher_section = ""
            if item_index < len(teacher_scored_texts):
                teacher_section = (
                    "<section><h3>Teacher Scored Text</h3>"
                    "<pre>{teacher_text}</pre></section>"
                ).format(teacher_text=html.escape(str(teacher_scored_texts[item_index])))
            sample_sections.append(
                """
                <details{open_attr}>
                  <summary>Sample {item_number}/{batch_size} &mdash; {prompt_preview}</summary>
                  <div class="detail-panel">
                    <section><h3>Prompt</h3><pre>{prompt}</pre></section>
                    <section><h3>Response</h3><pre>{response}</pre></section>
                    <section><h3>Feedback</h3><pre>{feedback}</pre></section>
                    {teacher_section}
                  </div>
                </details>
                """.format(
                    open_attr=" open" if batch_size == 1 else "",
                    item_number=item_index + 1,
                    batch_size=batch_size,
                    prompt_preview=html.escape(feedback_prompt_preview(sample.prompt, limit=80)),
                    prompt=html.escape(sample.prompt),
                    response=html.escape(sample.response),
                    feedback=html.escape(sample.feedback),
                    teacher_section=teacher_section,
                )
            )

        rows.append(
            """
            <tr id="{detail_row_id}" class="detail-row">
              <td colspan="7">
                {samples}
                <div class="detail-panel" style="margin-top: 0.75rem">
                  <section><h3>Timing (ms)</h3><pre>{timing_json}</pre></section>
                  <section><h3>Training metrics</h3><pre>{metrics_json}</pre></section>
                  <section><h3>vLLM orchestration</h3><pre>{vllm_json}</pre></section>
                  <section><h3>Error</h3><pre>{error_value}</pre></section>
                </div>
              </td>
            </tr>
            """.format(
                detail_row_id=detail_row_id,
                samples="\n".join(sample_sections),
                timing_json=html.escape(timing_json),
                metrics_json=html.escape(metrics_json),
                vllm_json=html.escape(vllm_json),
                error_value=html.escape(error_value),
            )
        )

    if not rows:
        return '<tr><td colspan="7">No feedback records found.</td></tr>'
    return "\n".join(rows)


def feedback_dashboard_html(
    records: list[FeedbackLogRecord], pagination_nav: str = ""
) -> str:
    """Render feedback records into the dashboard HTML template.

    Args:
        records: Feedback records to display.
        pagination_nav: Pre-rendered pagination HTML to inject.

    Returns:
        Rendered HTML content.
    """
    template = FEEDBACK_DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    table_rows = feedback_dashboard_rows(records)
    return template.replace("{{TABLE_ROWS}}", table_rows).replace(
        "{{PAGINATION}}", pagination_nav
    )
