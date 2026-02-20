"""Tests for claas.pagination utilities."""

from __future__ import annotations

from claas.pagination import paginate, render_pagination_nav

# ---------------------------------------------------------------------------
# paginate() tests
# ---------------------------------------------------------------------------


class TestPaginate:
    def test_basic_math(self):
        info = paginate(total_items=50, page=1, per_page=10)
        assert info.total_pages == 5
        assert info.page == 1
        assert info.offset == 0
        assert info.has_prev is False
        assert info.has_next is True

    def test_middle_page(self):
        info = paginate(total_items=50, page=3, per_page=10)
        assert info.page == 3
        assert info.offset == 20
        assert info.has_prev is True
        assert info.has_next is True

    def test_last_page(self):
        info = paginate(total_items=50, page=5, per_page=10)
        assert info.page == 5
        assert info.offset == 40
        assert info.has_prev is True
        assert info.has_next is False

    def test_page_clamped_over(self):
        info = paginate(total_items=10, page=999, per_page=10)
        assert info.page == 1
        assert info.total_pages == 1

    def test_page_clamped_under(self):
        info = paginate(total_items=10, page=-5, per_page=10)
        assert info.page == 1

    def test_zero_items(self):
        info = paginate(total_items=0, page=1, per_page=10)
        assert info.total_pages == 1
        assert info.page == 1
        assert info.offset == 0
        assert info.has_prev is False
        assert info.has_next is False

    def test_partial_last_page(self):
        info = paginate(total_items=15, page=2, per_page=10)
        assert info.total_pages == 2
        assert info.page == 2
        assert info.offset == 10


# ---------------------------------------------------------------------------
# render_pagination_nav() tests
# ---------------------------------------------------------------------------


class TestRenderPaginationNav:
    def test_empty_when_zero_items(self):
        info = paginate(total_items=0)
        result = render_pagination_nav(info, "/v1/dashboard")
        assert result == ""

    def test_first_page_has_disabled_prev(self):
        info = paginate(total_items=50, page=1, per_page=10)
        result = render_pagination_nav(info, "/v1/dashboard")
        assert "disabled" in result
        assert "Prev" in result
        assert 'href=' in result  # next link present
        assert "Page 1 of 5" in result

    def test_last_page_has_disabled_next(self):
        info = paginate(total_items=50, page=5, per_page=10)
        result = render_pagination_nav(info, "/v1/dashboard")
        assert "Next" in result
        # Next should be disabled
        assert '<span class="pagination-link disabled">Next' in result
        # Prev should be a link
        assert "page=4" in result

    def test_middle_page_has_both_links(self):
        info = paginate(total_items=50, page=3, per_page=10)
        result = render_pagination_nav(info, "/v1/dashboard")
        assert "page=2" in result
        assert "page=4" in result
        assert "Page 3 of 5" in result
        assert "(50 total)" in result

    def test_extra_params_in_urls(self):
        info = paginate(total_items=50, page=2, per_page=10)
        result = render_pagination_nav(
            info, "/v1/eval", extra_params={"results_dir": "./data/evals"}
        )
        assert "results_dir" in result
        assert "page=1" in result
        assert "page=3" in result

    def test_single_page_both_disabled(self):
        info = paginate(total_items=5, page=1, per_page=10)
        result = render_pagination_nav(info, "/v1/dashboard")
        assert result != ""
        assert '<span class="pagination-link disabled">' in result
        assert "Page 1 of 1" in result
        # No <a> links should exist
        assert "<a " not in result
