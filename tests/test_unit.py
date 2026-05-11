"""
tests/test_unit.py

Unit tests for models.py and fetchers.py.
No network, no ANTHROPIC_API_KEY required — safe to run in CI.
"""

import pytest
from models import Conflict, Finding, Job, JobState, Scope

# ── models.py ─────────────────────────────────────────────────────────────────

class TestJobStateMachine:
    def _scoped_job(self) -> Job:
        job = Job(topic="test", depth=3)
        scope = Scope(
            topic="test",
            question_type="factual",
            source_targets=["q1", "q2"],
            completion_criteria="done",
            depth=3,
        )
        job.transition(JobState.GATHERING, scope=scope)
        return job

    def test_initial_state(self):
        job = Job(topic="test", depth=3)
        assert job.state == JobState.SCOPING

    def test_happy_path(self):
        job = Job(topic="test", depth=3)
        scope = Scope("t", "factual", ["q"], "done", 3)
        job.transition(JobState.GATHERING, scope=scope)
        job.transition(JobState.CONFLICTING)
        job.transition(JobState.SYNTHESIZING, conflicts=[])
        job.transition(JobState.DONE, synthesis="answer", confidence_scores={}, open_questions=[])
        assert job.state == JobState.DONE
        assert job.is_terminal

    def test_illegal_transition_raises(self):
        job = Job(topic="test", depth=3)
        with pytest.raises(ValueError, match="Illegal transition"):
            job.transition(JobState.DONE)

    def test_skip_state_raises(self):
        job = self._scoped_job()  # now in GATHERING
        with pytest.raises(ValueError):
            job.transition(JobState.DONE)

    def test_failed_reachable_from_any_state(self):
        for start_state in [JobState.SCOPING, JobState.GATHERING,
                             JobState.CONFLICTING, JobState.SYNTHESIZING]:
            job = Job(topic="t", depth=1)
            job.state = start_state  # force state for test
            job.transition(JobState.FAILED, error="oops")
            assert job.state == JobState.FAILED
            assert job.is_terminal

    def test_terminal_states_have_no_transitions(self):
        for state in [JobState.DONE, JobState.FAILED]:
            job = Job(topic="t", depth=1)
            job.state = state
            with pytest.raises(ValueError):
                job.transition(JobState.SCOPING)

    def test_token_budget_enforced(self):
        job = self._scoped_job()
        over_budget = Finding(
            source_url="http://x.com",
            summary="test",
            key_claims=["a"],
            token_count=job.MAX_TOKENS_PER_FINDING + 1,
        )
        with pytest.raises(ValueError, match="token budget"):
            job.add_finding(over_budget)

    def test_findings_capacity_flag(self):
        job = self._scoped_job()
        for i in range(job.MAX_FINDINGS):
            job.add_finding(Finding(f"http://{i}.com", "s", ["c"], 100))
        assert job.findings_at_capacity
        assert len(job.findings) == job.MAX_FINDINGS

    def test_total_tokens_accumulate(self):
        job = self._scoped_job()
        job.add_finding(Finding("http://a.com", "s", ["c"], 300))
        job.add_finding(Finding("http://b.com", "s", ["c"], 450))
        assert job.total_tokens_used == 750

    def test_to_dict_safe_fields(self):
        job = Job(topic="privacy check", depth=2)
        d = job.to_dict()
        assert "id" in d
        assert "state" in d
        # Raw findings list should not leak — only count
        assert "findings" not in d
        assert "findings_count" in d


# ── fetchers.py ───────────────────────────────────────────────────────────────

from fetchers import (
    FetchResult,
    MAX_CONTENT_CHARS,
    MIN_CONTENT_CHARS,
    _extract_content,
    _score_block,
)


class TestContentScoring:
    def test_nav_text_scores_zero(self):
        assert _score_block("Home About Contact Blog") == 0.0

    def test_too_short_scores_zero(self):
        assert _score_block("hi there") == 0.0

    def test_empty_scores_zero(self):
        assert _score_block("") == 0.0

    def test_body_text_scores_above_threshold(self):
        body = (
            "The Model Context Protocol defines a standardized JSON-RPC interface "
            "for AI agents to discover and invoke external tools at runtime. "
            "Anthropic introduced it in 2024 to unify fragmented tool-calling conventions."
        )
        assert _score_block(body) > 0.3


class TestContentExtraction:
    def test_article_wins_over_nav_footer(self):
        html = """<html><head><title>MCP Guide</title></head><body>
        <nav>Home | About | Pricing | Login</nav>
        <script>window._ads = true;</script>
        <article>
          <h1>What is MCP?</h1>
          <p>The Model Context Protocol is a standardized JSON-RPC interface that lets
          AI agents discover and call external tools at runtime. It was designed to unify
          fragmented tool-calling conventions across different AI frameworks.</p>
          <p>MCP servers expose typed tool schemas. Clients introspect and call them
          with structured arguments, decoupling the server from any specific client.</p>
        </article>
        <footer>Copyright 2025 · Privacy Policy · Cookie Settings</footer>
        </body></html>"""
        title, content = _extract_content(html)
        assert title == "What is MCP?"
        assert "Model Context Protocol" in content
        assert "Copyright" not in content
        assert "window._ads" not in content

    def test_boilerplate_selectors_stripped(self):
        html = """<html><body>
        <div class="cookie-banner">Accept cookies to continue.</div>
        <div class="sidebar">Related: Post 1, Post 2</div>
        <main>
          <p>FastMCP is a Python framework for building MCP servers. It handles tool
          registration, schema generation, and transport negotiation automatically.
          Developers define tools as ordinary async Python functions.</p>
        </main>
        <div class="newsletter-signup">Subscribe for updates.</div>
        </body></html>"""
        _, content = _extract_content(html)
        assert "cookie" not in content.lower()
        assert "newsletter" not in content.lower()
        assert "FastMCP" in content

    def test_content_cap_respected(self):
        long_html = (
            "<html><body><article>"
            + "<p>" + ("Word " * 50 + ". ") * 300 + "</p>"
            + "</article></body></html>"
        )
        _, content = _extract_content(long_html)
        assert len(content) <= MAX_CONTENT_CHARS

    def test_og_title_fallback(self):
        html = """<html><head>
        <meta property="og:title" content="OG Title Here">
        </head><body><main><p>Some content that is long enough to matter here.
        This paragraph has enough words to score above the threshold value.</p></main>
        </body></html>"""
        title, _ = _extract_content(html)
        assert title == "OG Title Here"


class TestFetchResult:
    def test_is_usable_happy(self):
        r = FetchResult("http://x.com", "T", "x" * MIN_CONTENT_CHARS, True)
        assert r.is_usable

    def test_is_usable_too_short(self):
        r = FetchResult("http://x.com", "T", "x" * (MIN_CONTENT_CHARS - 1), True)
        assert not r.is_usable

    def test_is_usable_failed(self):
        r = FetchResult("http://x.com", "T", "x" * 500, False, error="timeout")
        assert not r.is_usable

    def test_is_usable_empty_content(self):
        r = FetchResult("http://x.com", "T", "", True)
        assert not r.is_usable
