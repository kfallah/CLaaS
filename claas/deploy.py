"""Unified deployment entrypoint for CLaaS.

Deploy this module to publish API, worker, and teacher objects together:
    modal deploy -m claas.deploy
"""

from __future__ import annotations

import modal

from .api import app as api_app
from .training.teacher_service import app as teacher_app
from .training.worker import app as worker_app

app = modal.App("claas-distill")
app.include(api_app)
app.include(worker_app)
app.include(teacher_app)
