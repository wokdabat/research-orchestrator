from __future__ import annotations
import sys
import os
import asyncio
from typing import Any

#try:
#    from dotenv import load_dotenv
#    load_dotenv()
#except ImportError:
#    pass

import fastmcp
from fastmcp import FastMCP

# from agents import (
#     conflict_agent,
#     gap_agent,
#     scope_agent,
#     source_agent,
#     synthesis_agent,
# )
# from fetchers import search_and_fetch
# from models import Finding, Job, JobState

# ── In-memory job store (swap for Redis/SQLite for persistence) ──────────────
_jobs: dict[str, Job] = {}

mcp = FastMCP(
    name="research-orchestrator",
    instructions=(
        "A stateful research engine. Call start_research to begin, "
        "then poll get_state until DONE, then call get_synthesis for the answer. "
        "Use inject_source mid-run to add your own content. "
        "Use ask_followup to drill deeper on any sub-question."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_job(job_id: str) -> Job:
    job = _jobs.get(job_id)
    if not job:
        raise ValueError(f"Job '{job_id}' not found.")
    return job


async def _run_pipeline(job: Job) -> None:
    """
    Execute the full state machine pipeline for a job.
    Runs in the background so the caller isn't blocked.
    Each stage is a guarded transition — failures land in FAILED.
    """
    try:
        # ── SCOPING ──────────────────────────────────────────────────────────
        scope = await scope_agent(job.topic, job.depth)
        job.transition(JobState.GATHERING, scope=scope)

        # ── GATHERING ────────────────────────────────────────────────────────
        # In production: replace _mock_fetch with real web fetch + chunking
        tasks = [
            _fetch_and_summarize(query, job)
            for query in scope.source_targets[: job.MAX_FINDINGS]
        ]
        await asyncio.gather(*tasks)
        job.transition(JobState.CONFLICTING)

        # ── CONFLICTING ──────────────────────────────────────────────────────
        conflicts = await conflict_agent(job.findings)
        job.transition(JobState.SYNTHESIZING, conflicts=conflicts)

        # ── SYNTHESIZING ─────────────────────────────────────────────────────
        synthesis, scores = await synthesis_agent(job.topic, job.findings, job.conflicts)
        gaps = await gap_agent(job.topic, job.scope.completion_criteria, job.findings)
        job.transition(
            JobState.DONE,
            synthesis=synthesis,
            confidence_scores=scores,
            open_questions=gaps,
        )

    except Exception as exc:
        # Any unhandled error fails the job gracefully
        try:
            job.transition(JobState.FAILED, error=str(exc))
        except ValueError:
            job.state = JobState.FAILED
            job.error = str(exc)


async def _fetch_and_summarize(query: str, job: Job) -> None:
    """
    Search the web for a query, fetch top results, summarize each into a Finding.
    Skips results that are unusable or would exceed the job's token budget.
    """
    if job.findings_at_capacity:
        return

    fetch_results = await search_and_fetch(query, max_results=3)

    for fr in fetch_results:
        if job.findings_at_capacity:
            break
        if not fr.is_usable:
            continue
        try:
            finding = await source_agent(
                query=query,
                content=fr.content,
                source_url=fr.url,
            )
            if finding.token_count <= job.MAX_TOKENS_PER_FINDING:
                job.add_finding(finding)
        except Exception:
            # One bad source doesn't kill the whole gather phase
            continue


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
async def start_research(topic: str, depth: int = 3) -> dict[str, Any]:
    """
    Start a new research job.

    Args:
        topic: The research question or topic to investigate.
        depth: How exhaustively to research (1=quick, 5=thorough). Default 3.

    Returns:
        job_id to use in all subsequent calls, and initial state.
    """
    if not topic.strip():
        raise ValueError("topic must not be empty.")
    if not 1 <= depth <= 5:
        raise ValueError("depth must be between 1 and 5.")

    job = Job(topic=topic.strip(), depth=depth)
    _jobs[job.id] = job

    # Fire pipeline in background — caller polls get_state
    asyncio.create_task(_run_pipeline(job))

    return {
        "job_id": job.id,
        "state": job.state.value,
        "message": "Research started. Poll get_state until DONE, then call get_synthesis.",
    }


@mcp.tool()
async def get_state(job_id: str) -> dict[str, Any]:
    """
    Poll the current state of a research job.

    Args:
        job_id: The ID returned by start_research.

    Returns:
        Full job snapshot including state, findings count, and any error.
    """
    job = _get_job(job_id)
    return job.to_dict()


@mcp.tool()
async def inject_source(job_id: str, content: str, source_url: str = "injected") -> dict[str, Any]:
    """
    Inject external content into an active GATHERING job.
    Use this to add content the server can't fetch itself (e.g. internal docs).

    Args:
        job_id:     The job to inject into.
        content:    Raw text content to summarize and add as a finding.
        source_url: Label for this source (URL or descriptive name).

    Returns:
        Updated findings count.
    """
    job = _get_job(job_id)
    if job.state != JobState.GATHERING:
        raise ValueError(
            f"Can only inject during GATHERING state. Current state: {job.state.value}"
        )
    if job.findings_at_capacity:
        raise ValueError("Findings at capacity. Cannot inject more sources.")

    finding = await source_agent(
        query=job.topic,
        content=content,
        source_url=source_url,
    )
    job.add_finding(finding)
    return {"findings_count": len(job.findings), "source_url": source_url}


@mcp.tool()
async def get_conflicts(job_id: str) -> dict[str, Any]:
    """
    Return all detected contradictions for a completed or synthesizing job.

    Args:
        job_id: The job ID.

    Returns:
        List of conflict objects, each with both sides and confidence scores.
    """
    job = _get_job(job_id)
    if job.state in (JobState.SCOPING, JobState.GATHERING):
        raise ValueError("Conflicts not yet available. Job is still gathering.")
    return {
        "conflicts": [
            {
                "claim_a": c.claim_a,
                "source_a": c.source_a,
                "claim_b": c.claim_b,
                "source_b": c.source_b,
                "characterization": c.characterization,
                "confidence_a": c.confidence_a,
                "confidence_b": c.confidence_b,
            }
            for c in job.conflicts
        ],
        "count": len(job.conflicts),
    }


@mcp.tool()
async def get_synthesis(job_id: str) -> dict[str, Any]:
    """
    Return the final synthesized answer for a completed job.

    Args:
        job_id: The job ID.

    Returns:
        synthesis text, per-claim confidence scores, and open questions.
    """
    job = _get_job(job_id)
    if job.state != JobState.DONE:
        raise ValueError(
            f"Synthesis not ready. Current state: {job.state.value}. "
            "Poll get_state until DONE."
        )
    return {
        "synthesis":        job.synthesis,
        "confidence_scores": job.confidence_scores,
        "open_questions":   job.open_questions,
        "findings_count":   len(job.findings),
        "conflicts_count":  len(job.conflicts),
        "total_tokens_used": job.total_tokens_used,
    }


@mcp.tool()
async def ask_followup(job_id: str, question: str, depth: int = 2) -> dict[str, Any]:
    """
    Drill deeper on a specific angle from a completed job.
    Forks a new child job that inherits parent findings as pre-loaded context.

    Args:
        job_id:   The completed parent job to build on.
        question: The specific follow-up question.
        depth:    Research depth for the child job (default 2, shallower).

    Returns:
        New child job_id to poll independently.
    """
    parent = _get_job(job_id)
    if parent.state != JobState.DONE:
        raise ValueError("Can only ask_followup on a DONE job.")

    # Fork: child job starts with parent's synthesis as injected context
    child = Job(topic=question.strip(), depth=depth)
    _jobs[child.id] = child

    # Pre-load parent findings into child so GATHERING is cheaper
    for f in parent.findings:
        if not child.findings_at_capacity:
            child.add_finding(f)

    asyncio.create_task(_run_pipeline(child))

    return {
        "child_job_id": child.id,
        "parent_job_id": job_id,
        "state": child.state.value,
        "inherited_findings": len(child.findings),
        "message": f"Child job started with {len(child.findings)} inherited findings. Poll get_state(child_job_id).",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entrypoint for both `uv run server.py` and the `research-mcp` script."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, or set the env var directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    transport = "streamable-http" if "--http" in sys.argv else "stdio"
    port = int(os.environ.get("RESEARCH_MCP_PORT", "8000"))

    if transport == "streamable-http":
        print(f"Research MCP server running on http://localhost:{port}/mcp", flush=True)
        mcp.run(transport=transport, host="0.0.0.0", port=port)
    else:
        # Redirect stdout to stderr so FastMCP splash doesn't corrupt the JSON-RPC stream
        sys.stdout = sys.stderr
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()