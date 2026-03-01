"""Similar session search: find relevant past sessions for a new prompt.

Finds past sessions similar to a new user prompt using keyword-based
matching (MVP) with optional embedding-based search. Searches
project-local first, then across all projects.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

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

    # Basic Jaccard similarity
    jaccard = len(intersection) / len(union)

    # Boost for matching more specific terms (longer words)
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

    # Extract meaningful parts from file paths
    file_tokens: set[str] = set()
    for f in files_touched:
        parts = re.findall(r"[a-z0-9]+", f.lower())
        file_tokens.update(p for p in parts if len(p) > 2)

    matches = sum(1 for t in prompt_tokens if t in file_tokens)
    return min(0.15, matches * 0.03)


class SessionSearcher:
    """Find past sessions relevant to a new user prompt.

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

        Searches project-local first, then global. Uses keyword
        similarity with optional embedding boost.

        Args:
            prompt: The user's new prompt/task description.
            project_dir: Optional project directory to search first.
            limit: Maximum number of results to return.

        Returns:
            List of SimilarSession sorted by relevance (highest first).
        """
        candidates: list[SimilarSession] = []
        prompt_tokens = _tokenize(prompt)

        if not prompt_tokens:
            return []

        # Search project-local sessions first
        if project_dir:
            project_sessions = await self._db.get_sessions_for_project(
                project_dir, limit=50
            )
            for s in project_sessions:
                similar = self._score_session(s, prompt_tokens, is_same_project=True)
                if similar and similar.relevance_score > 0.05:
                    candidates.append(similar)

        # Search global sessions (different projects)
        # Note: UserModelDB doesn't have a "get all sessions" method,
        # so we skip global search if no project is given.
        # A full implementation would add that method.

        # Sort by relevance and return top results
        candidates.sort(key=lambda s: s.relevance_score, reverse=True)
        return candidates[:limit]

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

        # Compute similarity components
        keyword_score = _keyword_similarity(prompt_tokens, goal_tokens)
        tool_bonus = _tool_similarity(prompt_tokens, tools_used)
        file_bonus = _file_similarity(prompt_tokens, files_touched)

        # Same-project boost
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
        )
