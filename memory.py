"""
memory.py — Bridge between research jobs and the knowledge graph

Two responsibilities:

  READ  (before a job runs)
    - Query graph for prior findings related to the new topic
    - Pre-load them as synthetic findings into the job
    - Pass active hypotheses to the job so agents are aware of them

  WRITE (after a job completes)
    - Persist all findings and conflicts into the graph
    - Evaluate new claims against active hypotheses
    - Propose new hypotheses from the synthesis
    - Link related topics via similarity edges

This module is the only place that touches both the job model
and the graph/hypothesis layers. server.py calls it at job
boundaries; nothing else needs to know it exists.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from models import Job

from graph import (
    init_db,
    get_prior_findings,
    store_job_findings,
    get_graph_summary,
)
from hypothesis import (
    list_hypotheses,
    evaluate_claims_against_hypotheses,
    propose_hypotheses_from_synthesis,
    STATUS_ACTIVE,
    STATUS_PROPOSED,
)
from models import Finding

# ── Ensure DB is initialised on import ───────────────────────────────────────
init_db()


# ── READ: pre-load prior knowledge into a new job ────────────────────────────

def preload_job_from_memory(job: "Job") -> dict[str, Any]:
    """
    Before GATHERING starts, query the knowledge graph for prior findings
    relevant to this job's topic and inject them as synthetic findings.

    Also collects active hypotheses so the ScopeAgent can factor them
    into the source target selection.

    Returns a summary dict describing what was preloaded.
    """
    prior_findings = get_prior_findings(job.topic)
    active_hypotheses = list_hypotheses(status=STATUS_ACTIVE)

    injected = 0
    for pf in prior_findings:
        if job.findings_at_capacity:
            break

        # Reconstruct a Finding from the graph node
        data = pf.get("data", {})
        source_url = data.get("source_url", pf.get("label", "memory"))

        # The claim node label IS the claim text — wrap it as a summary
        claim_text = pf.get("label", "")
        if not claim_text or len(claim_text) < 20:
            continue

        synthetic = Finding(
            source_url=f"[memory] {source_url}",
            summary=f"Prior finding: {claim_text}",
            key_claims=[claim_text],
            token_count=min(len(claim_text) // 4, 200),  # conservative estimate
        )

        try:
            job.add_finding(synthetic)
            injected += 1
        except ValueError:
            break  # token budget exceeded

    return {
        "prior_findings_available": len(prior_findings),
        "injected_into_job":        injected,
        "active_hypotheses":        len(active_hypotheses),
        "hypothesis_statements":    [h["statement"] for h in active_hypotheses[:5]],
    }


def get_active_hypotheses_for_topic(topic: str) -> list[dict[str, Any]]:
    """
    Return active hypotheses relevant to a topic.
    Used by server.py to include hypothesis context in agent prompts.
    """
    all_active = list_hypotheses(status=STATUS_ACTIVE)

    # Filter to hypotheses whose topic overlaps with the job topic
    from graph import _jaccard
    relevant = [
        h for h in all_active
        if _jaccard(h.get("topic", ""), topic) > 0.2
        or topic.lower() in h["statement"].lower()
    ]
    return relevant


# ── WRITE: persist completed job into the knowledge graph ─────────────────────

async def persist_job_to_memory(
    job: "Job",
    anthropic_client: Any,
    model: str,
) -> dict[str, Any]:
    """
    After a job reaches DONE state, persist everything into the graph
    and run hypothesis evaluation + proposal.

    Steps:
      1. Store all findings and conflicts in the graph
      2. Evaluate new claims against active hypotheses
      3. Propose new hypotheses from the synthesis (requires Claude call)

    Returns a summary of what was written and proposed.
    """
    # ── 1. Store findings and conflicts ──────────────────────────────────────
    findings_dicts = [
        {
            "source_url": f.source_url,
            "summary":    f.summary,
            "key_claims": f.key_claims,
            "token_count": f.token_count,
        }
        for f in job.findings
        # Skip synthetic memory findings — already in graph
        if not f.source_url.startswith("[memory]")
    ]

    conflicts_dicts = [
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
    ]

    topic_id = store_job_findings(
        job_id=job.id,
        topic_label=job.topic,
        findings=findings_dicts,
        conflicts=conflicts_dicts,
    )

    # ── 2. Evaluate claims against active hypotheses ──────────────────────────
    all_claims: list[str] = []
    for f in job.findings:
        all_claims.extend(f.key_claims)

    affected_hypotheses = evaluate_claims_against_hypotheses(
        claims=all_claims,
        topic_label=job.topic,
    )

    # ── 3. Propose new hypotheses from synthesis ──────────────────────────────
    proposed: list[dict[str, Any]] = []
    if job.synthesis:
        proposed = await propose_hypotheses_from_synthesis(
            topic=job.topic,
            synthesis=job.synthesis,
            open_questions=job.open_questions,
            conflicts=conflicts_dicts,
            anthropic_client=anthropic_client,
            model=model,
        )

    return {
        "topic_id":              topic_id,
        "findings_stored":       len(findings_dicts),
        "conflicts_stored":      len(conflicts_dicts),
        "hypotheses_affected":   len(affected_hypotheses),
        "hypotheses_proposed":   len(proposed),
        "proposed_hypotheses":   proposed,
        "affected_hypotheses":   affected_hypotheses,
    }


# ── Convenience: full graph + hypothesis status summary ───────────────────────

def get_memory_status() -> dict[str, Any]:
    """
    Return a combined summary of graph state and hypothesis status.
    Used by the get_graph_summary MCP tool.
    """
    graph_summary = get_graph_summary()

    all_hypotheses = list_hypotheses()
    by_status: dict[str, list[dict]] = {}
    for h in all_hypotheses:
        s = h["status"]
        by_status.setdefault(s, []).append(h)

    return {
        "graph":       graph_summary,
        "hypotheses": {
            "total":     len(all_hypotheses),
            "by_status": {k: len(v) for k, v in by_status.items()},
            "proposed":  [
                {"id": h["id"], "statement": h["statement"], "topic": h["topic"]}
                for h in by_status.get(STATUS_PROPOSED, [])
            ],
            "active": [
                {
                    "id":         h["id"],
                    "statement":  h["statement"],
                    "confidence": h["confidence"],
                    "for":        h["evidence_for"],
                    "against":    h["evidence_against"],
                }
                for h in by_status.get(STATUS_ACTIVE, [])
            ],
            "confirmed": [
                {"id": h["id"], "statement": h["statement"], "confidence": h["confidence"]}
                for h in by_status.get("CONFIRMED", [])
            ],
            "refuted": [
                {"id": h["id"], "statement": h["statement"], "confidence": h["confidence"]}
                for h in by_status.get("REFUTED", [])
            ],
        },
    }
