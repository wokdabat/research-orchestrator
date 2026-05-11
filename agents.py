"""
agents.py — Subagent runners

Each agent is a single, scoped Claude API call.
They receive exactly what they need and return structured output.
No agent knows about other agents or the overall job state.
"""

from __future__ import annotations
import json
import os
import re
import anthropic
from models import Conflict, Finding, Scope

CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
MODEL  = "claude-sonnet-4-20250514"


def _call(system: str, user: str, max_tokens: int = 1000) -> str:
    """Single Claude call — returns the text content."""
    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def _parse_json(raw: str) -> dict:
    """Strip markdown fences then parse JSON."""
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean)
    return json.loads(clean)


# ── ScopeAgent ────────────────────────────────────────────────────────────────

async def scope_agent(topic: str, depth: int) -> Scope:
    """
    Parse the research question into a structured scope.
    Returns: Scope with source_targets, question_type, completion_criteria.
    """
    system = """You are a research scoping specialist.
Given a topic and depth (1=shallow, 5=exhaustive), output ONLY valid JSON:
{
  "question_type": "factual|comparative|causal|exploratory",
  "source_targets": ["<search query 1>", "<search query 2>", ...],
  "completion_criteria": "<one sentence: what a complete answer looks like>"
}
source_targets should be 2-3 specific search queries for depth 1-2, up to 6 for depth 4-5.
No preamble, no markdown, just the JSON object."""

    user = f"Topic: {topic}\nDepth: {depth}"
    raw = _call(system, user)
    data = _parse_json(raw)

    return Scope(
        topic=topic,
        question_type=data["question_type"],
        source_targets=data["source_targets"],
        completion_criteria=data["completion_criteria"],
        depth=depth,
    )


# ── SourceAgent ───────────────────────────────────────────────────────────────

async def source_agent(query: str, content: str, source_url: str) -> Finding:
    """
    Summarize a single source into a Finding.
    content: raw text fetched from the source (caller handles fetching).
    Enforces ≤ 800 token output via max_tokens.
    """
    system = """You are a research extraction specialist.
Given source content and a research query, output ONLY valid JSON:
{
  "summary": "<concise paragraph, max 200 words>",
  "key_claims": ["<claim 1>", "<claim 2>", "<claim 3>"]
}
key_claims: 2-5 discrete, falsifiable statements from this source.
No preamble, no markdown, just the JSON object."""

    user = f"Query: {query}\n\nSource content:\n{content[:4000]}"
    raw = _call(system, user, max_tokens=600)
    data = _parse_json(raw)

    # Rough token estimate: 1 token ≈ 4 chars
    token_est = len(data["summary"]) // 4 + sum(len(c) // 4 for c in data["key_claims"])

    return Finding(
        source_url=source_url,
        summary=data["summary"],
        key_claims=data["key_claims"],
        token_count=min(token_est, 800),
    )


# ── ConflictAgent ─────────────────────────────────────────────────────────────

async def conflict_agent(findings: list[Finding]) -> list[Conflict]:
    """
    Scan all findings for contradictions.
    Returns a (possibly empty) list of Conflict objects.
    """
    if len(findings) < 2:
        return []

    # Build a compact summary of all claims for the model
    claims_block = "\n".join(
        f"[{i}] {f.source_url}\n" + "\n".join(f"  - {c}" for c in f.key_claims)
        for i, f in enumerate(findings)
    )

    system = """You are a contradiction detection specialist.
Given a numbered list of sources and their claims, identify direct contradictions.
Output ONLY valid JSON — an array of conflict objects (empty array if none):
[
  {
    "claim_a": "<exact claim text>",
    "source_a": "<source URL>",
    "claim_b": "<exact claim text>",
    "source_b": "<source URL>",
    "characterization": "<one sentence: what specifically disagrees>",
    "confidence_a": 0.0,
    "confidence_b": 0.0
  }
]
Only flag direct factual contradictions, not differences of emphasis.
No preamble, no markdown, just the JSON array."""

    user = f"Sources and claims:\n{claims_block}"
    raw = _call(system, user, max_tokens=800)
    data = _parse_json(raw)

    if not isinstance(data, list):
        return []

    return [
        Conflict(
            claim_a=d["claim_a"],
            source_a=d["source_a"],
            claim_b=d["claim_b"],
            source_b=d["source_b"],
            characterization=d["characterization"],
            confidence_a=float(d.get("confidence_a", 0.5)),
            confidence_b=float(d.get("confidence_b", 0.5)),
        )
        for d in data
    ]


# ── GapAgent ──────────────────────────────────────────────────────────────────

async def gap_agent(topic: str, completion_criteria: str, findings: list[Finding]) -> list[str]:
    """
    Identify what's still unknown or underexplored given the current findings.
    Returns a ranked list of open questions.
    """
    summaries = "\n".join(f"- {f.summary}" for f in findings)

    system = """You are a research gap analyst.
Given a research topic, its completion criteria, and current findings,
identify what is still unknown or underexplored.
Output ONLY valid JSON — an array of open questions, ranked by importance:
["<question 1>", "<question 2>", ...]
Max 5 questions. No preamble, no markdown, just the JSON array."""

    user = (
        f"Topic: {topic}\n"
        f"Completion criteria: {completion_criteria}\n\n"
        f"Current findings:\n{summaries}"
    )
    raw = _call(system, user, max_tokens=400)
    data = _parse_json(raw)
    return data if isinstance(data, list) else []


# ── SynthesisAgent ────────────────────────────────────────────────────────────

async def synthesis_agent(
    topic: str,
    findings: list[Finding],
    conflicts: list[Conflict],
) -> tuple[str, dict[str, float]]:
    """
    Build the final answer from all gathered findings + conflict map.
    Returns: (synthesis_text, confidence_scores_per_claim)
    """
    findings_block = "\n\n".join(
        f"Source: {f.source_url}\n{f.summary}"
        for f in findings
    )
    conflicts_block = (
        "\n".join(
            f"CONFLICT: {c.characterization} "
            f"(confidence A={c.confidence_a:.1f}, B={c.confidence_b:.1f})"
            for c in conflicts
        )
        if conflicts else "None detected."
    )

    system = """You are a research synthesis specialist.
Given findings and any detected conflicts, write a comprehensive answer.
Output ONLY valid JSON:
{
  "synthesis": "<full answer, 200-400 words>",
  "confidence_scores": {
    "<key claim or sub-topic>": 0.0
  }
}
confidence_scores: 0.0 (highly uncertain) to 1.0 (well supported).
Where conflicts exist, note both sides and their confidence scores.
No preamble, no markdown, just the JSON object."""

    user = (
        f"Research topic: {topic}\n\n"
        f"Findings:\n{findings_block}\n\n"
        f"Conflicts:\n{conflicts_block}"
    )
    raw = _call(system, user, max_tokens=1000)
    data = _parse_json(raw)

    return data["synthesis"], data.get("confidence_scores", {})
