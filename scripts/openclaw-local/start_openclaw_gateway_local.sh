#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-18789}"

exec openclaw gateway --port "$PORT" --verbose
