"""Unified deployment entrypoint for CLaaS.

Deploy this module to publish API, worker, and teacher objects together:
    modal deploy -m claas.modal.deploy
"""

from __future__ import annotations

import modal

from claas.api import app as api_app
from claas.modal.teacher_service import app as teacher_app
from claas.modal.worker import app as worker_app

app = modal.App("claas-distill")
app.include(api_app)
app.include(worker_app)
app.include(teacher_app)
