"""Shared server-side pagination utilities."""

from __future__ import annotations

import math
import urllib.parse
from dataclasses import dataclass


@dataclass
class PaginationInfo:
    """Computed pagination state."""

    page: int
    per_page: int
    total_items: int
    total_pages: int

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.per_page

    @property
    def has_prev(self) -> bool:
        return self.page > 1

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages


def paginate(total_items: int, page: int = 1, per_page: int = 20) -> PaginationInfo:
    """Compute pagination state, clamping *page* to ``[1, total_pages]``."""
    total_pages = max(1, math.ceil(total_items / per_page))
    page = max(1, min(page, total_pages))
    return PaginationInfo(
        page=page,
        per_page=per_page,
        total_items=total_items,
        total_pages=total_pages,
    )


def render_pagination_nav(
    info: PaginationInfo,
    base_url: str,
    extra_params: dict[str, str] | None = None,
) -> str:
    """Render an HTML ``<nav>`` with Prev/Next links.

    Returns an empty string when there are no items to paginate.
    """
    if info.total_items == 0:
        return ""

    def _build_url(page: int) -> str:
        params: dict[str, str] = {}
        if extra_params:
            params.update(extra_params)
        params["page"] = str(page)
        params["per_page"] = str(info.per_page)
        qs = urllib.parse.urlencode(params)
        return f"{base_url}?{qs}"

    if info.has_prev:
        prev_link = '<a class="pagination-link" href="{url}">&laquo; Prev</a>'.format(
            url=_build_url(info.page - 1),
        )
    else:
        prev_link = '<span class="pagination-link disabled">&laquo; Prev</span>'

    if info.has_next:
        next_link = '<a class="pagination-link" href="{url}">Next &raquo;</a>'.format(
            url=_build_url(info.page + 1),
        )
    else:
        next_link = '<span class="pagination-link disabled">Next &raquo;</span>'

    status = "Page {page} of {total_pages} ({total_items} total)".format(
        page=info.page,
        total_pages=info.total_pages,
        total_items=info.total_items,
    )

    return (
        '<nav class="pagination">'
        "{prev} "
        '<span class="pagination-status">{status}</span> '
        "{next}"
        "</nav>"
    ).format(prev=prev_link, status=status, next=next_link)
