"""
graph.py — Knowledge graph engine

Persistent knowledge store for the research orchestrator.
SQLite stores nodes and edges on disk.
networkx loads them at runtime for traversal, similarity, and analysis.

Node types:
  topic      — a research subject area
  claim      — a discrete, falsifiable statement from a source
  source     — a web page or injected document
  hypothesis — a tracked belief with evolving confidence

Edge types:
  SUPPORTS      — claim supports hypothesis
  CONTRADICTS   — claim contradicts hypothesis or another claim
  RELATED_TO    — topic is semantically related to another topic
  DERIVED_FROM  — claim came from a source
  PRODUCED      — job produced a claim
  BELONGS_TO    — claim belongs to a topic
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "knowledge.db"

# Minimum word-overlap similarity to consider two topics related
TOPIC_SIMILARITY_THRESHOLD = 0.25

# Max prior findings to pre-load into a new job
MAX_PRELOAD_FINDINGS = 6


# ── Database setup ────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    label       TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,
    src_id      TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    rel         TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    data        TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    FOREIGN KEY (src_id) REFERENCES nodes(id),
    FOREIGN KEY (dst_id) REFERENCES nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_type  ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_edges_src   ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst   ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_rel   ON edges(rel);
"""


@contextmanager
def _db():
    """Thread-safe SQLite connection context manager."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with _db() as conn:
        conn.executescript(SCHEMA)


# ── Similarity helpers ────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Simple word tokenizer — lowercase, strip punctuation, remove stopwords."""
    stopwords = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "how", "what", "why", "when", "where", "does", "do", "did",
    }
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in stopwords and len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two text strings."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Node operations ───────────────────────────────────────────────────────────

def add_node(
    node_type: str,
    label: str,
    data: dict[str, Any] | None = None,
    node_id: str | None = None,
) -> str:
    """
    Add a node to the graph. Returns the node id.
    If a node of the same type and label already exists, returns its id.
    """
    with _db() as conn:
        # Dedup by type + label
        row = conn.execute(
            "SELECT id FROM nodes WHERE type=? AND label=?",
            (node_type, label)
        ).fetchone()
        if row:
            return row["id"]

        nid = node_id or str(uuid.uuid4())
        conn.execute(
            "INSERT INTO nodes (id, type, label, data, created_at) VALUES (?,?,?,?,?)",
            (nid, node_type, label, json.dumps(data or {}), _now())
        )
        return nid


def update_node(node_id: str, data: dict[str, Any]) -> None:
    """Merge data into an existing node's data field."""
    with _db() as conn:
        row = conn.execute("SELECT data FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            raise ValueError(f"Node '{node_id}' not found.")
        existing = json.loads(row["data"])
        existing.update(data)
        conn.execute(
            "UPDATE nodes SET data=? WHERE id=?",
            (json.dumps(existing), node_id)
        )


def get_node(node_id: str) -> dict[str, Any] | None:
    """Fetch a single node by id."""
    with _db() as conn:
        row = conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            return None
        return {**dict(row), "data": json.loads(row["data"])}


def get_nodes_by_type(node_type: str) -> list[dict[str, Any]]:
    """Fetch all nodes of a given type."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE type=? ORDER BY created_at DESC",
            (node_type,)
        ).fetchall()
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]


# ── Edge operations ───────────────────────────────────────────────────────────

def add_edge(
    src_id: str,
    dst_id: str,
    rel: str,
    weight: float = 1.0,
    data: dict[str, Any] | None = None,
) -> str:
    """
    Add a directed edge. Deduplicates by (src, dst, rel).
    Returns the edge id.
    """
    with _db() as conn:
        row = conn.execute(
            "SELECT id FROM edges WHERE src_id=? AND dst_id=? AND rel=?",
            (src_id, dst_id, rel)
        ).fetchone()
        if row:
            # Update weight if re-asserted
            conn.execute(
                "UPDATE edges SET weight=? WHERE id=?",
                (weight, row["id"])
            )
            return row["id"]

        eid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO edges (id, src_id, dst_id, rel, weight, data, created_at) VALUES (?,?,?,?,?,?,?)",
            (eid, src_id, dst_id, rel, weight, json.dumps(data or {}), _now())
        )
        return eid


def get_edges(
    node_id: str,
    rel: str | None = None,
    direction: str = "out",
) -> list[dict[str, Any]]:
    """
    Fetch edges connected to a node.
    direction: 'out' (src=node), 'in' (dst=node), 'both'
    """
    with _db() as conn:
        if direction == "out":
            q = "SELECT * FROM edges WHERE src_id=?"
        elif direction == "in":
            q = "SELECT * FROM edges WHERE dst_id=?"
        else:
            q = "SELECT * FROM edges WHERE src_id=? OR dst_id=?"

        params: tuple = (node_id,) if direction != "both" else (node_id, node_id)
        if rel:
            q += " AND rel=?"
            params = params + (rel,)

        rows = conn.execute(q, params).fetchall()
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]


# ── networkx graph builder ────────────────────────────────────────────────────

def build_nx_graph() -> nx.DiGraph:
    """
    Load the full SQLite graph into a networkx DiGraph.
    Used for traversal, path-finding, and centrality analysis.
    """
    G = nx.DiGraph()
    with _db() as conn:
        for row in conn.execute("SELECT * FROM nodes").fetchall():
            G.add_node(
                row["id"],
                type=row["type"],
                label=row["label"],
                data=json.loads(row["data"]),
                created_at=row["created_at"],
            )
        for row in conn.execute("SELECT * FROM edges").fetchall():
            G.add_edge(
                row["src_id"],
                row["dst_id"],
                rel=row["rel"],
                weight=row["weight"],
                data=json.loads(row["data"]),
            )
    return G


# ── Topic similarity + prior finding retrieval ────────────────────────────────

def find_related_topics(topic_label: str) -> list[tuple[str, float]]:
    """
    Find existing topic nodes similar to a new topic label.
    Returns list of (topic_id, similarity_score) sorted descending.
    """
    topics = get_nodes_by_type("topic")
    scored = []
    for t in topics:
        sim = _jaccard(topic_label, t["label"])
        if sim >= TOPIC_SIMILARITY_THRESHOLD:
            scored.append((t["id"], sim))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def get_prior_findings(topic_label: str) -> list[dict[str, Any]]:
    """
    Retrieve prior claims relevant to a new topic, ranked by confidence.
    Used to pre-load context into new research jobs.
    Returns up to MAX_PRELOAD_FINDINGS claim dicts.
    """
    related = find_related_topics(topic_label)
    if not related:
        return []

    related_topic_ids = {tid for tid, _ in related}
    all_claims: list[dict[str, Any]] = []

    with _db() as conn:
        # Find claims connected to related topics via BELONGS_TO edges
        for tid in related_topic_ids:
            rows = conn.execute(
                """SELECT n.* FROM nodes n
                   JOIN edges e ON e.src_id = n.id
                   WHERE e.dst_id=? AND e.rel='BELONGS_TO' AND n.type='claim'
                   ORDER BY n.created_at DESC""",
                (tid,)
            ).fetchall()
            for row in rows:
                d = {**dict(row), "data": json.loads(row["data"])}
                all_claims.append(d)

    # Deduplicate by node id, sort by confidence descending
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for c in all_claims:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)

    unique.sort(
        key=lambda c: c["data"].get("confidence", 0.5),
        reverse=True
    )
    return unique[:MAX_PRELOAD_FINDINGS]


# ── High-level write operations ───────────────────────────────────────────────

def store_job_findings(
    job_id: str,
    topic_label: str,
    findings: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> str:
    """
    Persist a completed job's findings into the knowledge graph.

    Creates:
      - A topic node for the research subject
      - A source node per finding
      - A claim node per key claim
      - DERIVED_FROM edges: claim → source
      - BELONGS_TO edges:   claim → topic
      - PRODUCED edges:     job   → claim
      - CONTRADICTS edges:  claim → claim (from conflicts)

    Returns the topic node id.
    """
    # Job node (ensures PRODUCED edges have a valid src)
    add_node("job", job_id, {"topic": topic_label}, node_id=job_id)

    # Topic node
    topic_id = add_node("topic", topic_label, {"job_id": job_id})

    for finding in findings:
        source_url = finding.get("source_url", "unknown")
        summary    = finding.get("summary", "")
        key_claims = finding.get("key_claims", [])

        # Source node
        source_id = add_node("source", source_url, {
            "summary": summary[:500],
            "job_id":  job_id,
        })

        # Claim nodes
        for claim_text in key_claims:
            if not claim_text.strip():
                continue
            claim_id = add_node("claim", claim_text[:400], {
                "source_url": source_url,
                "job_id":     job_id,
                "confidence": finding.get("token_count", 400) / 800,  # proxy score
            })
            add_edge(claim_id, source_id, "DERIVED_FROM")
            add_edge(claim_id, topic_id,  "BELONGS_TO")
            add_edge(job_id,   claim_id,  "PRODUCED")

    # Conflict edges between claims
    for conflict in conflicts:
        claim_a_text = conflict.get("claim_a", "")
        claim_b_text = conflict.get("claim_b", "")

        # Find matching claim nodes by label
        with _db() as conn:
            row_a = conn.execute(
                "SELECT id FROM nodes WHERE type='claim' AND label LIKE ? LIMIT 1",
                (claim_a_text[:50] + "%",)
            ).fetchone()
            row_b = conn.execute(
                "SELECT id FROM nodes WHERE type='claim' AND label LIKE ? LIMIT 1",
                (claim_b_text[:50] + "%",)
            ).fetchone()

        if row_a and row_b:
            add_edge(row_a["id"], row_b["id"], "CONTRADICTS", weight=0.5, data={
                "characterization": conflict.get("characterization", ""),
                "confidence_a":     conflict.get("confidence_a", 0.5),
                "confidence_b":     conflict.get("confidence_b", 0.5),
            })

    # Link related topics
    related = find_related_topics(topic_label)
    for related_id, sim_score in related:
        if related_id != topic_id:
            add_edge(topic_id, related_id, "RELATED_TO", weight=sim_score)

    return topic_id


# ── Graph summary ─────────────────────────────────────────────────────────────

def get_graph_summary() -> dict[str, Any]:
    """
    Return a high-level summary of the knowledge graph state.
    Used by the get_graph_summary MCP tool.
    """
    with _db() as conn:
        node_counts = {}
        for row in conn.execute(
            "SELECT type, COUNT(*) as cnt FROM nodes GROUP BY type"
        ).fetchall():
            node_counts[row["type"]] = row["cnt"]

        edge_counts = {}
        for row in conn.execute(
            "SELECT rel, COUNT(*) as cnt FROM edges GROUP BY rel"
        ).fetchall():
            edge_counts[row["rel"]] = row["cnt"]

        # Top 5 topics by claim count
        top_topics = conn.execute(
            """SELECT n.label, COUNT(e.id) as claim_count
               FROM nodes n
               LEFT JOIN edges e ON e.dst_id = n.id AND e.rel = 'BELONGS_TO'
               WHERE n.type = 'topic'
               GROUP BY n.id
               ORDER BY claim_count DESC
               LIMIT 5"""
        ).fetchall()

        # Contested claims (have at least one CONTRADICTS edge)
        contested = conn.execute(
            """SELECT COUNT(DISTINCT src_id) as cnt FROM edges WHERE rel='CONTRADICTS'"""
        ).fetchone()

    return {
        "node_counts":         node_counts,
        "edge_counts":         edge_counts,
        "top_topics":          [{"label": r["label"], "claims": r["claim_count"]} for r in top_topics],
        "contested_claims":    contested["cnt"] if contested else 0,
        "total_nodes":         sum(node_counts.values()),
        "total_edges":         sum(edge_counts.values()),
    }
