from __future__ import annotations
import asyncio
import os
import sys
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import anthropic
from fastmcp import FastMCP
from agents import (
    conflict_agent,
    gap_agent,
    scope_agent,
    source_agent,
    synthesis_agent,
)
from fetchers import search_and_fetch
from models import Finding, Job, JobState
from memory import (
    preload_job_from_memory,
    persist_job_to_memory,
    get_memory_status,
    get_active_hypotheses_for_topic,
)
from hypothesis import (
    list_hypotheses,
    approve_hypothesis as _approve_hypothesis,
    reject_hypothesis as _reject_hypothesis,
    add_user_hypothesis,
    STATUS_PROPOSED,
    STATUS_ACTIVE,
)

# ── Clients + job store ───────────────────────────────────────────────────────

MODEL = os.environ.get("RESEARCH_MCP_MODEL", "claude-sonnet-4-20250514")
_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
_jobs: dict[str, Job] = {}

mcp = FastMCP(
    name="research-orchestrator",
    instructions=(
        "A stateful research engine with persistent memory. "
        "Call start_research to begin, poll get_state until DONE, "
        "then call get_synthesis for the answer. "
        "Use get_graph_summary to see accumulated knowledge. "
        "Use get_hypotheses to see tracked beliefs and approve_hypothesis to activate them. "
        "Use add_hypothesis to track your own hypotheses. "
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
    try:
        # ── Pre-load prior knowledge ──────────────────────────────────────────
        preload_summary = preload_job_from_memory(job)
        job.memory_preload = preload_summary  # store for get_state

        # ── SCOPING ───────────────────────────────────────────────────────────
        # Include active hypotheses in scope context
        active_hyps = get_active_hypotheses_for_topic(job.topic)
        hyp_context = (
            "\n".join(f"- {h['statement']}" for h in active_hyps)
            if active_hyps else ""
        )
        scope = await scope_agent(job.topic, job.depth)
        job.transition(JobState.GATHERING, scope=scope)

        # ── GATHERING ─────────────────────────────────────────────────────────
        tasks = [
            _fetch_and_summarize(query, job)
            for query in scope.source_targets[: job.MAX_FINDINGS]
        ]
        await asyncio.gather(*tasks)
        job.transition(JobState.CONFLICTING)

        # ── CONFLICTING ───────────────────────────────────────────────────────
        conflicts = await conflict_agent(job.findings)
        job.transition(JobState.SYNTHESIZING, conflicts=conflicts)

        # ── SYNTHESIZING ──────────────────────────────────────────────────────
        synthesis, scores = await synthesis_agent(job.topic, job.findings, job.conflicts)
        gaps = await gap_agent(job.topic, job.scope.completion_criteria, job.findings)
        job.transition(
            JobState.DONE,
            synthesis=synthesis,
            confidence_scores=scores,
            open_questions=gaps,
        )

        # ── PERSIST TO MEMORY ─────────────────────────────────────────────────
        memory_result = await persist_job_to_memory(job, _client, MODEL)
        job.memory_write = memory_result  # store for get_state / get_synthesis

    except Exception as exc:
        try:
            job.transition(JobState.FAILED, error=str(exc))
        except ValueError:
            job.state = JobState.FAILED
            job.error = str(exc)


async def _fetch_and_summarize(query: str, job: Job) -> None:
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
            continue


# ── Original 6 MCP Tools ──────────────────────────────────────────────────────

@mcp.tool()
async def start_research(topic: str, depth: int = 3) -> dict[str, Any]:
    """
    Start a new research job.

    Args:
        topic: The research question or topic to investigate.
        depth: How exhaustively to research (1=quick, 5=thorough). Default 3.

    Returns:
        job_id, initial state, and any prior knowledge preloaded from memory.
    """
    if not topic.strip():
        raise ValueError("topic must not be empty.")
    if not 1 <= depth <= 5:
        raise ValueError("depth must be between 1 and 5.")

    job = Job(topic=topic.strip(), depth=depth)
    _jobs[job.id] = job
    asyncio.create_task(_run_pipeline(job))

    return {
        "job_id":  job.id,
        "state":   job.state.value,
        "message": "Research started. Poll get_state until DONE, then call get_synthesis.",
    }


@mcp.tool()
async def get_state(job_id: str) -> dict[str, Any]:
    """
    Poll the current state of a research job.

    Args:
        job_id: The ID returned by start_research.

    Returns:
        Full job snapshot including state, findings count, memory preload summary, and any error.
    """
    job = _get_job(job_id)
    d = job.to_dict()
    d["memory_preload"] = getattr(job, "memory_preload", None)
    return d


@mcp.tool()
async def inject_source(job_id: str, content: str, source_url: str = "injected") -> dict[str, Any]:
    """
    Inject external content into an active GATHERING job.

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
        List of conflict objects with both sides and confidence scores.
    """
    job = _get_job(job_id)
    if job.state in (JobState.SCOPING, JobState.GATHERING):
        raise ValueError("Conflicts not yet available. Job is still gathering.")
    return {
        "conflicts": [
            {
                "claim_a":          c.claim_a,
                "source_a":         c.source_a,
                "claim_b":          c.claim_b,
                "source_b":         c.source_b,
                "characterization": c.characterization,
                "confidence_a":     c.confidence_a,
                "confidence_b":     c.confidence_b,
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
        Synthesis text, confidence scores, open questions, and memory write summary.
    """
    job = _get_job(job_id)
    if job.state != JobState.DONE:
        raise ValueError(
            f"Synthesis not ready. Current state: {job.state.value}. "
            "Poll get_state until DONE."
        )
    memory_write = getattr(job, "memory_write", {})
    return {
        "synthesis":           job.synthesis,
        "confidence_scores":   job.confidence_scores,
        "open_questions":      job.open_questions,
        "findings_count":      len(job.findings),
        "conflicts_count":     len(job.conflicts),
        "total_tokens_used":   job.total_tokens_used,
        "hypotheses_proposed": memory_write.get("hypotheses_proposed", 0),
        "hypotheses_affected": memory_write.get("hypotheses_affected", 0),
        "proposed_hypotheses": memory_write.get("proposed_hypotheses", []),
    }


@mcp.tool()
async def ask_followup(job_id: str, question: str, depth: int = 2) -> dict[str, Any]:
    """
    Drill deeper on a specific angle from a completed job.
    Forks a child job inheriting parent findings and memory context.

    Args:
        job_id:   The completed parent job to build on.
        question: The specific follow-up question.
        depth:    Research depth for the child job (default 2).

    Returns:
        New child job_id to poll independently.
    """
    parent = _get_job(job_id)
    if parent.state != JobState.DONE:
        raise ValueError("Can only ask_followup on a DONE job.")

    child = Job(topic=question.strip(), depth=depth)
    _jobs[child.id] = child

    for f in parent.findings:
        if not child.findings_at_capacity:
            child.add_finding(f)

    asyncio.create_task(_run_pipeline(child))

    return {
        "child_job_id":       child.id,
        "parent_job_id":      job_id,
        "state":              child.state.value,
        "inherited_findings": len(child.findings),
        "message": f"Child job started with {len(child.findings)} inherited findings.",
    }


# ── 4 New Memory + Hypothesis Tools ──────────────────────────────────────────

@mcp.tool()
async def get_graph_summary() -> dict[str, Any]:
    """
    Return a summary of the persistent knowledge graph and hypothesis status.

    Shows accumulated knowledge across all research sessions:
    node counts by type, top researched topics, contested claims,
    and all hypotheses grouped by status.

    Returns:
        Graph statistics, top topics, and full hypothesis breakdown.
    """
    return get_memory_status()


@mcp.tool()
async def get_hypotheses(status: str = "") -> dict[str, Any]:
    """
    List tracked hypotheses, optionally filtered by status.

    Args:
        status: Filter by status — PROPOSED, ACTIVE, CONFIRMED, REFUTED, REJECTED.
                Leave empty to return all hypotheses.

    Returns:
        List of hypothesis objects with confidence scores and evidence counts.
    """
    valid_statuses = {"PROPOSED", "ACTIVE", "CONFIRMED", "REFUTED", "REJECTED", ""}
    if status.upper() not in valid_statuses:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {valid_statuses - {''}}")

    hyps = list_hypotheses(status=status.upper() if status else None)
    return {
        "hypotheses": hyps,
        "count":      len(hyps),
        "filter":     status or "all",
    }


@mcp.tool()
async def approve_hypothesis(hypothesis_id: str) -> dict[str, Any]:
    """
    Approve a PROPOSED hypothesis, moving it to ACTIVE status.

    ACTIVE hypotheses are automatically evaluated against all future
    research jobs on related topics. Confidence scores update as
    supporting or contradicting evidence accumulates.

    Args:
        hypothesis_id: The hypothesis ID from get_hypotheses.

    Returns:
        Updated hypothesis with new status.
    """
    return _approve_hypothesis(hypothesis_id)


@mcp.tool()
async def add_hypothesis(statement: str, topic: str) -> dict[str, Any]:
    """
    Add your own hypothesis to track across future research sessions.

    User-defined hypotheses start in ACTIVE status immediately —
    no approval step required.

    Args:
        statement: A clear, falsifiable hypothesis statement.
        topic:     The research topic this hypothesis relates to.

    Returns:
        New hypothesis object with ACTIVE status.
    """
    if not statement.strip():
        raise ValueError("statement must not be empty.")
    if not topic.strip():
        raise ValueError("topic must not be empty.")
    return add_user_hypothesis(statement.strip(), topic.strip())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = "streamable-http" if "--http" in sys.argv else "stdio"
    port = int(os.environ.get("RESEARCH_MCP_PORT", "8000"))

    if transport == "streamable-http":
        print(f"Research MCP server running on http://localhost:{port}/mcp", flush=True)
        mcp.run(transport=transport, host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
