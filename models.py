"""
models.py — Job dataclass + state machine transitions

The Job is the single source of truth for a research task.
States: SCOPING → GATHERING → CONFLICTING → SYNTHESIZING → DONE
        (any state) → FAILED
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


class JobState(str, Enum):
    SCOPING      = "SCOPING"
    GATHERING    = "GATHERING"
    CONFLICTING  = "CONFLICTING"
    SYNTHESIZING = "SYNTHESIZING"
    DONE         = "DONE"
    FAILED       = "FAILED"


# Valid state transitions — guards enforced in Job.transition()
TRANSITIONS: dict[JobState, list[JobState]] = {
    JobState.SCOPING:      [JobState.GATHERING,    JobState.FAILED],
    JobState.GATHERING:    [JobState.CONFLICTING,  JobState.FAILED],
    JobState.CONFLICTING:  [JobState.SYNTHESIZING, JobState.FAILED],
    JobState.SYNTHESIZING: [JobState.DONE,         JobState.FAILED],
    JobState.DONE:         [],   # terminal — ask_followup forks a new job
    JobState.FAILED:       [],   # terminal
}


@dataclass
class Finding:
    """A single summarized source."""
    source_url: str
    summary: str
    key_claims: list[str]
    token_count: int
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Conflict:
    """A contradiction detected between two findings."""
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    characterization: str       # what exactly disagrees
    confidence_a: float         # 0.0–1.0
    confidence_b: float


@dataclass
class Scope:
    """Structured interpretation of the research topic."""
    topic: str
    question_type: str          # e.g. "factual", "comparative", "causal"
    source_targets: list[str]   # search queries or URLs to hit
    completion_criteria: str    # what "done" looks like
    depth: int                  # 1–5


@dataclass
class Job:
    """
    Central state object for one research run.
    All mutations go through transition() to enforce the state machine.
    """
    id: str                      = field(default_factory=lambda: str(uuid.uuid4()))
    state: JobState              = JobState.SCOPING
    topic: str                   = ""
    depth: int                   = 3
    created_at: str              = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str              = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Populated as the machine progresses
    scope: Scope | None                         = None
    findings: list[Finding]                     = field(default_factory=list)
    conflicts: list[Conflict]                   = field(default_factory=list)
    synthesis: str | None                       = None
    confidence_scores: dict[str, float]         = field(default_factory=dict)
    open_questions: list[str]                   = field(default_factory=list)
    error: str | None                           = None

    # Context budget tracking
    total_tokens_used: int                      = 0
    MAX_TOKENS_PER_FINDING: int                 = field(default=800, repr=False)
    MAX_FINDINGS: int                           = field(default=10,  repr=False)

    def transition(self, new_state: JobState, **updates: Any) -> None:
        """
        Move to new_state if allowed, applying any keyword updates atomically.
        Raises ValueError on illegal transitions.
        """
        allowed = TRANSITIONS.get(self.state, [])
        if new_state not in allowed:
            raise ValueError(
                f"Illegal transition {self.state} → {new_state}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        for key, val in updates.items():
            if not hasattr(self, key):
                raise AttributeError(f"Job has no attribute '{key}'")
            setattr(self, key, val)
        self.state = new_state
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_finding(self, finding: Finding) -> None:
        """Add a finding, enforcing the per-finding token budget."""
        if finding.token_count > self.MAX_TOKENS_PER_FINDING:
            raise ValueError(
                f"Finding from {finding.source_url} exceeds token budget "
                f"({finding.token_count} > {self.MAX_TOKENS_PER_FINDING}). "
                "Summarize further before storing."
            )
        self.findings.append(finding)
        self.total_tokens_used += finding.token_count
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_terminal(self) -> bool:
        return self.state in (JobState.DONE, JobState.FAILED)

    @property
    def findings_at_capacity(self) -> bool:
        return len(self.findings) >= self.MAX_FINDINGS

    def to_dict(self) -> dict:
        """Serializable snapshot — safe to return to callers."""
        return {
            "id":               self.id,
            "state":            self.state.value,
            "topic":            self.topic,
            "depth":            self.depth,
            "created_at":       self.created_at,
            "updated_at":       self.updated_at,
            "findings_count":   len(self.findings),
            "conflicts_count":  len(self.conflicts),
            "synthesis":        self.synthesis,
            "confidence_scores": self.confidence_scores,
            "open_questions":   self.open_questions,
            "total_tokens_used": self.total_tokens_used,
            "error":            self.error,
        }
