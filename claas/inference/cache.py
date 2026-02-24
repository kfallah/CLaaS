"""Completion cache for storing raw completions keyed by content hash."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

_CACHE_TTL_SECS = 3600  # 1 hour


class CompletionCacheEntry:
    """A single cached completion with prompt, response, token IDs, and logprobs."""

    __slots__ = ("prompt", "response", "response_token_ids", "prompt_token_ids", "response_logprobs", "system_prompt", "created_at")

    def __init__(
        self,
        prompt: str,
        response: str,
        response_token_ids: list[int],
        prompt_token_ids: list[int],
        response_logprobs: list[float] | None,
        system_prompt: str,
    ) -> None:
        self.prompt = prompt
        self.response = response
        self.response_token_ids = response_token_ids
        self.prompt_token_ids = prompt_token_ids
        self.response_logprobs = response_logprobs
        self.system_prompt = system_prompt
        self.created_at = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > _CACHE_TTL_SECS


class CompletionCache:
    """FIFO cache keyed by SHA-256 of parsed content text."""

    def __init__(self, max_size: int = 100) -> None:
        self._store: OrderedDict[str, CompletionCacheEntry] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def put(self, content_hash: str, entry: CompletionCacheEntry) -> None:
        with self._lock:
            if content_hash in self._store:
                self._store.move_to_end(content_hash)
                self._store[content_hash] = entry
            else:
                self._store[content_hash] = entry
                while len(self._store) > self._max_size:
                    self._store.popitem(last=False)

    def get(self, content_hash: str) -> CompletionCacheEntry | None:
        with self._lock:
            entry = self._store.get(content_hash)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[content_hash]
                return None
            return entry


# Module-level singleton; max_size is set at first use from config.
completion_cache = CompletionCache()
