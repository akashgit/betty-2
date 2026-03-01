"""Similar session search: find relevant past sessions for a new prompt.

Finds past sessions similar to a new user prompt using keyword-based
matching with optional embedding-based search. Searches project-local
first, then across all projects.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .db import UserModelDB
from .session_reader import discover_sessions, parse_session

logger = logging.getLogger(__name__)


@dataclass
class SimilarSession:
    """A past session found to be similar to the current prompt."""

    session_id: str
    goal: str
    relevance_score: float  # 0.0-1.0
    project_dir: str = ""
    outcome: str | None = None
    tools_used: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    decisions_made: list[dict[str, Any]] = field(default_factory=list)
    patterns_observed: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)

    @property
    def went_well(self) -> list[str]:
        """Indicators of success from this session."""
        indicators = []
        if self.outcome and any(
            w in self.outcome.lower()
            for w in ("success", "completed", "fixed", "done", "resolved")
        ):
            indicators.append(self.outcome)
        return indicators

    @property
    def went_wrong(self) -> list[str]:
        """Indicators of problems from this session."""
        indicators = []
        if self.outcome and any(
            w in self.outcome.lower()
            for w in ("failed", "error", "broken", "reverted", "abandoned")
        ):
            indicators.append(self.outcome)
        return indicators


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall",
        "i", "me", "my", "we", "our", "you", "your", "it", "its",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "no", "so", "if", "then",
        "this", "that", "these", "those", "there", "here",
    }
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in stopwords and len(w) > 1]


def _keyword_similarity(prompt_tokens: list[str], goal_tokens: list[str]) -> float:
    """Compute keyword overlap similarity between prompt and goal.

    Uses Jaccard-like similarity with term frequency weighting.
    """
    if not prompt_tokens or not goal_tokens:
        return 0.0

    prompt_set = set(prompt_tokens)
    goal_set = set(goal_tokens)

    intersection = prompt_set & goal_set
    union = prompt_set | goal_set

    if not union:
        return 0.0

    jaccard = len(intersection) / len(union)

    specificity_bonus = sum(
        len(word) / 20.0 for word in intersection
    ) / max(len(union), 1)

    return min(1.0, jaccard + specificity_bonus * 0.3)


def _tool_similarity(prompt_tokens: list[str], tools_used: list[str]) -> float:
    """Bonus score if the prompt mentions tools that were used in the session."""
    if not tools_used:
        return 0.0

    tool_keywords = {tool.lower() for tool in tools_used}
    matches = sum(1 for t in prompt_tokens if t in tool_keywords)
    return min(0.2, matches * 0.05)


def _file_similarity(prompt_tokens: list[str], files_touched: list[str]) -> float:
    """Bonus if prompt mentions files or directories from the session."""
    if not files_touched:
        return 0.0

    file_tokens: set[str] = set()
    for f in files_touched:
        parts = re.findall(r"[a-z0-9]+", f.lower())
        file_tokens.update(p for p in parts if len(p) > 2)

    matches = sum(1 for t in prompt_tokens if t in file_tokens)
    return min(0.15, matches * 0.03)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _session_text(session_data: dict[str, Any]) -> str:
    """Build searchable text from a session summary dict."""
    parts = []
    if session_data.get("goal"):
        parts.append(session_data["goal"])
    if session_data.get("outcome"):
        parts.append(session_data["outcome"])

    tools = session_data.get("tools_used", [])
    if isinstance(tools, list) and tools:
        parts.append(" ".join(tools))

    files = session_data.get("files_touched", [])
    if isinstance(files, list) and files:
        parts.append(" ".join(files))

    patterns = session_data.get("patterns_observed", [])
    if isinstance(patterns, list) and patterns:
        parts.append(" ".join(patterns))

    decisions = session_data.get("decisions_made", [])
    if isinstance(decisions, list):
        for d in decisions:
            if isinstance(d, dict):
                if d.get("file"):
                    parts.append(d["file"])
                if d.get("command"):
                    parts.append(d["command"])

    return " ".join(p for p in parts if p)


def _extract_key_decisions(session_data: dict[str, Any]) -> list[str]:
    """Extract human-readable key decisions from a session."""
    decisions = session_data.get("decisions_made", [])
    key = []
    if isinstance(decisions, list):
        for d in decisions[:10]:
            if isinstance(d, dict):
                if d.get("type") == "file_modification":
                    key.append(f"{d.get('tool', 'Edit')} {d.get('file', '')}")
                elif d.get("type") == "system_command":
                    key.append(d.get("command", ""))
    return key


class SessionSearcher:
    """Find past sessions relevant to a new user prompt.

    Supports two modes:
    1. Keyword-based: Jaccard similarity on tokenized text (always available)
    2. Embedding-based: cosine similarity on LLM embeddings (when LLM is available)

    Searches project-local first, then global.

    Usage:
        searcher = SessionSearcher(db)
        results = await searcher.find_similar("fix the login bug", "/my/project")
        for session in results:
            print(f"{session.goal} (score: {session.relevance_score})")
    """

    def __init__(self, db: Any, llm: Any | None = None):
        """Initialize with a connected UserModelDB instance.

        Args:
            db: A connected UserModelDB instance.
            llm: Optional LLMService for embedding-based search.
        """
        self._db = db
        self._llm = llm

    async def find_similar(
        self,
        prompt: str,
        project_dir: str | None = None,
        limit: int = 5,
    ) -> list[SimilarSession]:
        """Find past sessions similar to the given prompt.

        Searches project-local first, then global. Uses embedding
        search when available, falls back to keyword similarity.

        Args:
            prompt: The user's new prompt/task description.
            project_dir: Optional project directory to search first.
            limit: Maximum number of results to return.

        Returns:
            List of SimilarSession sorted by relevance (highest first).
        """
        prompt_tokens = _tokenize(prompt)
        if not prompt_tokens:
            return []

        # Gather candidate sessions
        project_sessions: list[dict[str, Any]] = []
        if project_dir:
            project_sessions = await self._db.get_sessions_for_project(
                project_dir, limit=50
            )

        global_sessions = await self._get_all_sessions()

        # Deduplicate: project sessions take priority
        seen_ids = {s.get("session_id") for s in project_sessions}
        global_only = [s for s in global_sessions if s.get("session_id") not in seen_ids]

        # Try embedding search first
        if self._llm is not None and hasattr(self._llm, "embed"):
            try:
                results = await self._search_by_embedding(
                    prompt, project_sessions, global_only, limit
                )
                if results:
                    return results
            except Exception as e:
                logger.debug("Embedding search failed, falling back to keywords: %s", e)

        # Keyword search
        return self._search_by_keywords(
            prompt_tokens, project_sessions, global_only, limit
        )

    async def find_similar_from_transcripts(
        self,
        prompt: str,
        project_dir: str | None = None,
        limit: int = 5,
    ) -> list[SimilarSession]:
        """Find similar sessions by parsing JSONL transcripts directly.

        Use when the database hasn't been populated yet.

        Args:
            prompt: The user's current prompt.
            project_dir: If set, prioritize sessions from this project.
            limit: Maximum number of results.

        Returns:
            List of SimilarSession objects.
        """
        prompt_tokens = _tokenize(prompt)
        if not prompt_tokens:
            return []

        project_discovered = (
            discover_sessions(project_dir=project_dir, limit=50)
            if project_dir
            else []
        )
        global_discovered = discover_sessions(project_dir=None, limit=100)

        seen_ids = {sid for sid, _ in project_discovered}
        global_only_discovered = [
            (sid, p) for sid, p in global_discovered if sid not in seen_ids
        ]

        candidates: list[SimilarSession] = []

        for discovered, is_project in [
            (project_discovered, True),
            (global_only_discovered, False),
        ]:
            for _sid, path in discovered:
                try:
                    session = parse_session(path)
                except Exception:
                    continue

                if not session.goal:
                    continue

                goal_tokens = _tokenize(session.goal)
                # Also include user messages in matching
                all_text_tokens = list(goal_tokens)
                for turn in session.user_turns[:5]:
                    all_text_tokens.extend(_tokenize(turn.content[:200]))

                score = _keyword_similarity(prompt_tokens, all_text_tokens)

                # Project boost
                if is_project:
                    score = min(1.0, score + 0.1)

                if score > 0.05:
                    decisions = []
                    for tc in session.all_tool_calls:
                        if tc.tool_name in ("Write", "Edit") and tc.file_path:
                            decisions.append(f"{tc.tool_name} {tc.file_path}")
                        elif tc.tool_name == "Bash" and tc.command:
                            cmd = tc.command.strip()
                            if len(cmd) < 100:
                                decisions.append(cmd)

                    tools = sorted({tc.tool_name for tc in session.all_tool_calls})
                    files = sorted(
                        {tc.file_path for tc in session.all_tool_calls if tc.file_path}
                    )

                    candidates.append(SimilarSession(
                        session_id=session.session_id,
                        goal=session.goal,
                        relevance_score=round(score, 3),
                        project_dir=session.project_dir,
                        tools_used=tools,
                        files_touched=files[:20],
                        key_decisions=decisions[:10],
                    ))

        candidates.sort(key=lambda s: s.relevance_score, reverse=True)
        return candidates[:limit]

    # --- Internal ---

    async def _get_all_sessions(self) -> list[dict[str, Any]]:
        """Fetch all recent sessions from the database."""
        try:
            async with self._db.db.execute(
                "SELECT * FROM session_summaries ORDER BY started_at DESC LIMIT 100"
            ) as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    r = dict(row)
                    for f in ("tools_used", "files_touched", "decisions_made", "patterns_observed"):
                        r[f] = json.loads(r[f])
                    results.append(r)
                return results
        except Exception as e:
            logger.warning("Failed to fetch global sessions: %s", e)
            return []

    def _score_session(
        self,
        session_data: dict[str, Any],
        prompt_tokens: list[str],
        is_same_project: bool = False,
    ) -> SimilarSession | None:
        """Score a session's relevance to the prompt."""
        goal = session_data.get("goal", "")
        if not goal:
            return None

        goal_tokens = _tokenize(goal)
        tools_used = session_data.get("tools_used", [])
        files_touched = session_data.get("files_touched", [])

        keyword_score = _keyword_similarity(prompt_tokens, goal_tokens)
        tool_bonus = _tool_similarity(prompt_tokens, tools_used)
        file_bonus = _file_similarity(prompt_tokens, files_touched)

        project_boost = 0.1 if is_same_project else 0.0

        total_score = min(1.0, keyword_score + tool_bonus + file_bonus + project_boost)

        if total_score < 0.05:
            return None

        return SimilarSession(
            session_id=session_data.get("session_id", ""),
            goal=goal,
            relevance_score=round(total_score, 3),
            project_dir=session_data.get("project_dir", ""),
            outcome=session_data.get("outcome"),
            tools_used=tools_used if isinstance(tools_used, list) else [],
            files_touched=files_touched if isinstance(files_touched, list) else [],
            decisions_made=session_data.get("decisions_made", []),
            patterns_observed=session_data.get("patterns_observed", []),
            key_decisions=_extract_key_decisions(session_data),
        )

    def _search_by_keywords(
        self,
        prompt_tokens: list[str],
        project_sessions: list[dict[str, Any]],
        global_sessions: list[dict[str, Any]],
        limit: int,
    ) -> list[SimilarSession]:
        """Search using keyword similarity."""
        candidates: list[SimilarSession] = []

        for s in project_sessions:
            similar = self._score_session(s, prompt_tokens, is_same_project=True)
            if similar and similar.relevance_score > 0.05:
                candidates.append(similar)

        for s in global_sessions:
            similar = self._score_session(s, prompt_tokens, is_same_project=False)
            if similar and similar.relevance_score > 0.05:
                candidates.append(similar)

        candidates.sort(key=lambda s: s.relevance_score, reverse=True)
        return candidates[:limit]

    async def _search_by_embedding(
        self,
        prompt: str,
        project_sessions: list[dict[str, Any]],
        global_sessions: list[dict[str, Any]],
        limit: int,
    ) -> list[SimilarSession]:
        """Search using embedding cosine similarity."""
        query_embedding = await self._llm.embed(prompt)
        if not query_embedding:
            return []

        scored: list[tuple[float, dict[str, Any], bool]] = []

        for session_data in project_sessions:
            text = _session_text(session_data)
            if not text.strip():
                continue
            try:
                doc_embedding = await self._llm.embed(text[:2000])
                score = _cosine_similarity(query_embedding, doc_embedding)
                scored.append((score, session_data, True))
            except Exception:
                continue

        for session_data in global_sessions:
            text = _session_text(session_data)
            if not text.strip():
                continue
            try:
                doc_embedding = await self._llm.embed(text[:2000])
                score = _cosine_similarity(query_embedding, doc_embedding)
                scored.append((score, session_data, False))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, session_data, is_project in scored[:limit]:
            if score > 0.0:
                # Apply project boost
                final_score = min(1.0, score * 1.2) if is_project else score
                results.append(SimilarSession(
                    session_id=session_data.get("session_id", ""),
                    goal=session_data.get("goal", ""),
                    relevance_score=round(final_score, 3),
                    project_dir=session_data.get("project_dir", ""),
                    outcome=session_data.get("outcome"),
                    tools_used=session_data.get("tools_used", []),
                    files_touched=session_data.get("files_touched", []),
                    decisions_made=session_data.get("decisions_made", []),
                    patterns_observed=session_data.get("patterns_observed", []),
                    key_decisions=_extract_key_decisions(session_data),
                ))

        return results
