"""
hypothesis.py — Hypothesis lifecycle management

Hypotheses are tracked beliefs about research topics.
They evolve through a defined lifecycle:

  PROPOSED → ACTIVE → CONFIRMED (confidence >= 0.9)
                    → REFUTED   (confidence <= 0.1)

The server proposes hypotheses after each completed job.
The user approves, rejects, or adds their own via MCP tools.
Every subsequent related job automatically evaluates evidence
against ACTIVE hypotheses, updating confidence scores.

Confidence scoring:
  - Starts at 0.5 (neutral) when proposed
  - Each SUPPORTS edge from a claim: +0.08 (capped at 0.95)
  - Each CONTRADICTS edge from a claim: -0.10 (floored at 0.05)
  - Confidence decays toward 0.5 at 0.01/day if no new evidence
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from graph import (
    add_edge,
    add_node,
    get_nodes_by_type,
    get_node,
    get_edges,
    update_node,
    _db,
    _now,
    find_related_topics,
    build_nx_graph,
)

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_SUPPORT_DELTA    =  0.08
CONFIDENCE_CONTRADICT_DELTA = -0.10
CONFIDENCE_CONFIRMED        =  0.90
CONFIDENCE_REFUTED          =  0.10
CONFIDENCE_INITIAL          =  0.50
CONFIDENCE_MAX              =  0.95
CONFIDENCE_MIN              =  0.05

STATUS_PROPOSED  = "PROPOSED"
STATUS_ACTIVE    = "ACTIVE"
STATUS_CONFIRMED = "CONFIRMED"
STATUS_REFUTED   = "REFUTED"
STATUS_REJECTED  = "REJECTED"


# ── Core hypothesis operations ────────────────────────────────────────────────

def create_hypothesis(
    statement: str,
    topic_label: str,
    status: str = STATUS_PROPOSED,
    source: str = "server",
) -> str:
    """
    Create a new hypothesis node in the graph.
    Returns the hypothesis node id.
    Deduplicates by statement — same statement returns existing id.
    """
    data = {
        "status":      status,
        "confidence":  CONFIDENCE_INITIAL,
        "topic":       topic_label,
        "source":      source,  # "server" or "user"
        "evidence_for":     0,
        "evidence_against": 0,
        "proposed_at": _now(),
        "updated_at":  _now(),
    }
    hid = add_node("hypothesis", statement[:500], data)

    # Link to related topic nodes
    related = find_related_topics(topic_label)
    for tid, sim in related:
        add_edge(hid, tid, "RELATED_TO", weight=sim)

    return hid


def get_hypothesis(hypothesis_id: str) -> dict[str, Any] | None:
    """Fetch a single hypothesis by id."""
    node = get_node(hypothesis_id)
    if not node or node["type"] != "hypothesis":
        return None
    return _format_hypothesis(node)


def list_hypotheses(status: str | None = None) -> list[dict[str, Any]]:
    """
    List all hypotheses, optionally filtered by status.
    Sorted by confidence descending.
    """
    nodes = get_nodes_by_type("hypothesis")
    if status:
        nodes = [n for n in nodes if n["data"].get("status") == status]
    formatted = [_format_hypothesis(n) for n in nodes]
    return sorted(formatted, key=lambda h: h["confidence"], reverse=True)


def approve_hypothesis(hypothesis_id: str) -> dict[str, Any]:
    """
    Move a PROPOSED hypothesis to ACTIVE.
    ACTIVE hypotheses are evaluated against new research jobs.
    """
    node = get_node(hypothesis_id)
    if not node:
        raise ValueError(f"Hypothesis '{hypothesis_id}' not found.")
    if node["data"]["status"] != STATUS_PROPOSED:
        raise ValueError(
            f"Can only approve PROPOSED hypotheses. "
            f"Current status: {node['data']['status']}"
        )
    update_node(hypothesis_id, {"status": STATUS_ACTIVE, "updated_at": _now()})
    return get_hypothesis(hypothesis_id)


def reject_hypothesis(hypothesis_id: str) -> dict[str, Any]:
    """Move a PROPOSED hypothesis to REJECTED (will not be tracked)."""
    node = get_node(hypothesis_id)
    if not node:
        raise ValueError(f"Hypothesis '{hypothesis_id}' not found.")
    update_node(hypothesis_id, {"status": STATUS_REJECTED, "updated_at": _now()})
    return get_hypothesis(hypothesis_id)


def add_user_hypothesis(statement: str, topic_label: str) -> dict[str, Any]:
    """
    Add a user-defined hypothesis directly to ACTIVE status.
    Bypasses the proposal/approval flow.
    """
    hid = create_hypothesis(
        statement=statement,
        topic_label=topic_label,
        status=STATUS_ACTIVE,
        source="user",
    )
    return get_hypothesis(hid)


# ── Evidence evaluation ───────────────────────────────────────────────────────

def evaluate_claims_against_hypotheses(
    claims: list[str],
    topic_label: str,
) -> list[dict[str, Any]]:
    """
    Evaluate a list of claim strings against all ACTIVE hypotheses
    related to the topic. Updates confidence scores and evidence counts.

    Returns list of updated hypothesis dicts that were affected.
    """
    active = list_hypotheses(status=STATUS_ACTIVE)
    if not active:
        return []

    # Find hypotheses related to this topic
    related_topic_ids = {tid for tid, _ in find_related_topics(topic_label)}
    affected: list[dict[str, Any]] = []

    for hyp in active:
        # Check if this hypothesis is linked to a related topic
        hyp_edges = get_edges(hyp["id"], rel="RELATED_TO", direction="out")
        hyp_topic_ids = {e["dst_id"] for e in hyp_edges}

        if not hyp_topic_ids & related_topic_ids and topic_label.lower() not in hyp["statement"].lower():
            continue  # Not related to this research topic

        confidence = hyp["confidence"]
        evidence_for     = hyp.get("evidence_for", 0)
        evidence_against = hyp.get("evidence_against", 0)
        changed = False

        for claim_text in claims:
            # Check if claim supports the hypothesis (word overlap heuristic)
            support_score   = _claim_hypothesis_alignment(claim_text, hyp["statement"], mode="support")
            contradict_score = _claim_hypothesis_alignment(claim_text, hyp["statement"], mode="contradict")

            if support_score > 0.3:
                # Find or create the claim node and add SUPPORTS edge
                claim_id = add_node("claim", claim_text[:400], {})
                add_edge(claim_id, hyp["id"], "SUPPORTS", weight=support_score)
                confidence = min(CONFIDENCE_MAX, confidence + CONFIDENCE_SUPPORT_DELTA)
                evidence_for += 1
                changed = True

            elif contradict_score > 0.3:
                claim_id = add_node("claim", claim_text[:400], {})
                add_edge(claim_id, hyp["id"], "CONTRADICTS", weight=contradict_score)
                confidence = max(CONFIDENCE_MIN, confidence + CONFIDENCE_CONTRADICT_DELTA)
                evidence_against += 1
                changed = True

        if changed:
            # Determine new status
            new_status = hyp["status"]
            if confidence >= CONFIDENCE_CONFIRMED:
                new_status = STATUS_CONFIRMED
            elif confidence <= CONFIDENCE_REFUTED:
                new_status = STATUS_REFUTED

            update_node(hyp["id"], {
                "confidence":      confidence,
                "status":          new_status,
                "evidence_for":    evidence_for,
                "evidence_against": evidence_against,
                "updated_at":      _now(),
            })
            affected.append(get_hypothesis(hyp["id"]))

    return affected


def _claim_hypothesis_alignment(
    claim: str,
    hypothesis: str,
    mode: str,
) -> float:
    """
    Estimate whether a claim supports or contradicts a hypothesis.
    Uses keyword overlap + negation detection.

    mode: "support" or "contradict"
    Returns a score 0.0–1.0.
    """
    import re

    negation_words = {"not", "never", "no", "cannot", "isn't", "aren't",
                      "doesn't", "don't", "fails", "unable", "impossible"}

    claim_words  = set(re.findall(r"[a-z0-9]+", claim.lower()))
    hyp_words    = set(re.findall(r"[a-z0-9]+", hypothesis.lower()))
    stopwords    = {"a", "an", "the", "is", "are", "was", "were", "it",
                    "this", "that", "and", "or", "but", "in", "on", "at"}

    claim_words -= stopwords
    hyp_words   -= stopwords

    if not claim_words or not hyp_words:
        return 0.0

    overlap = len(claim_words & hyp_words) / len(hyp_words)

    claim_has_negation = bool(claim_words & negation_words)
    hyp_has_negation   = bool(hyp_words   & negation_words)
    negation_flip      = claim_has_negation != hyp_has_negation

    if mode == "support":
        return overlap * (0.3 if negation_flip else 1.0)
    else:  # contradict
        return overlap * (1.0 if negation_flip else 0.2)


# ── Hypothesis proposal (HypothesisAgent) ────────────────────────────────────

async def propose_hypotheses_from_synthesis(
    topic: str,
    synthesis: str,
    open_questions: list[str],
    conflicts: list[dict[str, Any]],
    anthropic_client: Any,
    model: str,
) -> list[dict[str, Any]]:
    """
    Use a scoped Claude call to propose 1–3 hypotheses from a completed job.
    Proposed hypotheses land in PROPOSED status — user approves or rejects.
    Returns list of newly created hypothesis dicts.
    """
    import re

    # Build conflict summary for the prompt
    conflict_summary = "\n".join(
        f"- {c.get('characterization', '')}" for c in conflicts
    ) if conflicts else "None detected."

    questions_summary = "\n".join(
        f"- {q}" for q in open_questions[:5]
    ) if open_questions else "None."

    system = """You are a research hypothesis specialist.
Given a research synthesis, open questions, and detected conflicts,
propose 1-3 falsifiable hypotheses worth tracking across future research sessions.

Output ONLY valid JSON — an array of hypothesis objects:
[
  {
    "statement": "<a clear, falsifiable hypothesis statement>",
    "rationale": "<one sentence: why this is worth tracking>"
  }
]

Requirements:
- Each statement must be falsifiable (can be proven true or false)
- Focus on contested or uncertain claims, not settled facts
- Keep statements under 100 words
- No preamble, no markdown, just the JSON array"""

    user = (
        f"Research topic: {topic}\n\n"
        f"Synthesis:\n{synthesis[:1500]}\n\n"
        f"Detected conflicts:\n{conflict_summary}\n\n"
        f"Open questions:\n{questions_summary}"
    )

    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        proposals = json.loads(raw)
    except Exception:
        return []

    if not isinstance(proposals, list):
        return []

    created = []
    for p in proposals[:3]:
        statement = p.get("statement", "").strip()
        if not statement:
            continue
        hid = create_hypothesis(
            statement=statement,
            topic_label=topic,
            status=STATUS_PROPOSED,
            source="server",
        )
        node = get_hypothesis(hid)
        if node:
            created.append(node)

    return created


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_hypothesis(node: dict[str, Any]) -> dict[str, Any]:
    """Return a clean, serializable hypothesis dict."""
    data = node.get("data", {})
    return {
        "id":               node["id"],
        "statement":        node["label"],
        "status":           data.get("status", STATUS_PROPOSED),
        "confidence":       round(data.get("confidence", CONFIDENCE_INITIAL), 3),
        "topic":            data.get("topic", ""),
        "source":           data.get("source", "server"),
        "evidence_for":     data.get("evidence_for", 0),
        "evidence_against": data.get("evidence_against", 0),
        "proposed_at":      data.get("proposed_at", node.get("created_at", "")),
        "updated_at":       data.get("updated_at", ""),
    }
