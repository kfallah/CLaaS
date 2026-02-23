"""Feedback log persistence and retrieval.

Read/write feedback lifecycle records to disk as JSON files.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from claas.core.types import FeedbackLogRecord

logger = logging.getLogger(__name__)


def write_feedback_log(record: dict[str, Any] | FeedbackLogRecord, log_dir: str) -> str:
    """Persist a feedback lifecycle record to disk and return its path."""
    if isinstance(record, FeedbackLogRecord):
        payload = record.model_dump(mode="json")
        request_id = record.request_id
    else:
        payload = record
        request_id = str(payload.get("request_id", ""))

    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    request_id = request_id or uuid.uuid4().hex
    path = log_root / f"{timestamp}-{request_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(path)


def read_recent_feedback_logs(
    log_dir: str, offset: int = 0, limit: int = 20
) -> tuple[list[FeedbackLogRecord], int]:
    """Load recent feedback records from disk.

    Args:
        log_dir: Directory containing feedback JSON files.
        offset: Number of records to skip (for pagination).
        limit: Maximum number of records to load.

    Returns:
        A tuple of (records, total) where *records* is ordered newest-first
        and *total* is the full count of log files on disk.
    """
    log_root = Path(log_dir)
    if not log_root.exists():
        return [], 0

    log_paths = sorted(log_root.glob("*.json"), reverse=True)
    total = len(log_paths)
    selected_paths = log_paths[offset : offset + limit]
    records: list[FeedbackLogRecord] = []
    for path in selected_paths:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        try:
            records.append(FeedbackLogRecord.model_validate(payload))
        except Exception:
            logger.warning("Skipping invalid feedback log: %s", path)
    return records, total
