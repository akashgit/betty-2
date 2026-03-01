"""Intent engine: analyze user prompts and surface clarifying questions.

The core value proposition: when a user submits a prompt to Claude Code,
Betty analyzes intent against the user model and project context, and
surfaces questions the user would miss.

Flow:
1. UserPromptSubmit hook fires with the prompt
2. IntentEngine.analyze() loads user model, policies, similar sessions
3. Calls LLM to generate clarifying questions
4. Returns IntentAnalysis with questions, context, and predictions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Question priority level."""

    HIGH = "high"  # User should definitely answer before proceeding
    MEDIUM = "medium"  # Useful but optional
    LOW = "low"  # Nice to know


@dataclass
class Question:
    """A clarifying question for the user."""

    text: str
    reason: str  # Why Betty is asking
    priority: Priority = Priority.MEDIUM
    options: list[str] = field(default_factory=list)


@dataclass
class IntentAnalysis:
    """Result of analyzing a user prompt."""

    # Clarifying questions
    questions: list[Question] = field(default_factory=list)

    # Context to inject into the session
    suggested_context: str = ""

    # Similar past sessions
    similar_sessions: list[dict[str, Any]] = field(default_factory=list)

    # Applicable policies
    applicable_policies: list[dict[str, Any]] = field(default_factory=list)

    # Betty's prediction of what the user wants
    predicted_plan: str = ""

    # Overall confidence in the analysis
    confidence: float = 0.0

    @property
    def has_questions(self) -> bool:
        return len(self.questions) > 0

    @property
    def high_priority_questions(self) -> list[Question]:
        return [q for q in self.questions if q.priority == Priority.HIGH]

    def build_context_injection(self) -> str:
        """Build context string suitable for injecting into a Claude Code session."""
        parts: list[str] = []

        if self.suggested_context:
            parts.append(self.suggested_context)

        if self.questions:
            parts.append("\n## Questions to consider:")
            for q in self.questions:
                parts.append(f"- {q.text}")
                if q.reason:
                    parts.append(f"  Reason: {q.reason}")

        if self.applicable_policies:
            parts.append("\n## Applicable policies:")
            for p in self.applicable_policies:
                rule = p.get("rule", "")
                desc = p.get("description", "")
                if rule:
                    parts.append(f"- {rule}" + (f": {desc}" if desc else ""))

        return "\n".join(parts) if parts else ""


# LLM prompt for question generation
_QUESTION_GENERATION_PROMPT = """You are Betty, a peer programming assistant. A user is about to start a task in Claude Code. Based on the context below, generate clarifying questions they should consider.

USER PROMPT: {prompt}

PROJECT: {project_dir}

USER PREFERENCES:
{user_context}

SIMILAR PAST SESSIONS:
{similar_sessions}

APPLICABLE POLICIES:
{policies}

Generate 0-3 clarifying questions as JSON. Only ask questions where evidence suggests value. Do not ask obvious or low-value questions.

{{
  "questions": [
    {{
      "text": "the question",
      "reason": "why this matters (cite evidence: past session, policy, or preference)",
      "priority": "high | medium | low",
      "options": ["option1", "option2"]
    }}
  ],
  "predicted_plan": "brief prediction of what the user wants to achieve",
  "confidence": 0.7
}}

If there are no meaningful questions to ask, return {{"questions": [], "predicted_plan": "...", "confidence": 0.5}}."""


class IntentEngine:
    """Analyze user prompts and surface clarifying questions.

    Ties together the user model, session search, and policy engine
    to provide intent amplification.

    Usage:
        engine = IntentEngine(user_model=model, searcher=searcher, db=db, llm=llm)
        analysis = await engine.analyze("add user authentication", "/my/project")
        for question in analysis.questions:
            print(f"[{question.priority.value}] {question.text}")
    """

    def __init__(
        self,
        user_model: Any | None = None,
        searcher: Any | None = None,
        db: Any | None = None,
        llm: Any | None = None,
    ):
        """Initialize the intent engine.

        Args:
            user_model: UserModel instance for preference lookup.
            searcher: SessionSearcher instance for finding similar sessions.
            db: UserModelDB instance for policy lookup.
            llm: LLMService instance for question generation.
        """
        self._user_model = user_model
        self._searcher = searcher
        self._db = db
        self._llm = llm

    async def analyze(
        self,
        prompt: str,
        project_dir: str | None = None,
    ) -> IntentAnalysis:
        """Analyze a user prompt and generate clarifying questions.

        Loads user preferences, finds similar sessions, checks policies,
        and uses LLM to generate relevant clarifying questions.

        Falls back to heuristic-only analysis if LLM is unavailable.
        """
        analysis = IntentAnalysis()

        if not prompt or not prompt.strip():
            return analysis

        # Gather context in parallel-ish (all are fast DB lookups)
        user_context = ""
        if self._user_model:
            try:
                user_context = await self._user_model.get_relevant_context(
                    prompt, project_scope=project_dir
                )
                analysis.suggested_context = user_context
            except Exception as e:
                logger.warning("Failed to get user context: %s", e)

        # Find similar sessions
        similar_sessions_text = "None found."
        if self._searcher:
            try:
                similar = await self._searcher.find_similar(
                    prompt, project_dir=project_dir, limit=3
                )
                if similar:
                    analysis.similar_sessions = [
                        {
                            "session_id": s.session_id,
                            "goal": s.goal,
                            "outcome": s.outcome,
                            "relevance": s.relevance_score,
                        }
                        for s in similar
                    ]
                    session_lines = []
                    for s in similar:
                        line = f"- {s.goal} (relevance: {s.relevance_score:.0%})"
                        if s.outcome:
                            line += f" outcome: {s.outcome}"
                        session_lines.append(line)
                    similar_sessions_text = "\n".join(session_lines)
            except Exception as e:
                logger.warning("Failed to search similar sessions: %s", e)

        # Load applicable policies
        policies_text = "None."
        if self._db:
            try:
                policies = await self._db.get_policies(project_scope=project_dir)
                if policies:
                    analysis.applicable_policies = policies
                    policy_lines = []
                    for p in policies:
                        rule = p.get("rule", "")
                        desc = p.get("description", "")
                        policy_lines.append(f"- {rule}" + (f": {desc}" if desc else ""))
                    policies_text = "\n".join(policy_lines)
            except Exception as e:
                logger.warning("Failed to load policies: %s", e)

        # Generate questions using LLM
        if self._llm:
            try:
                # Truncate context sections to keep the prompt within
                # model limits (haiku context is small, and claude -p
                # rejects overly long input).
                _MAX_SECTION = 1500  # chars per section
                _user_ctx = (user_context or "No preferences learned yet.")[:_MAX_SECTION]
                _sessions_ctx = similar_sessions_text[:_MAX_SECTION]
                _policies_ctx = policies_text[:_MAX_SECTION]

                llm_prompt = _QUESTION_GENERATION_PROMPT.format(
                    prompt=prompt[:2000],
                    project_dir=project_dir or "unknown",
                    user_context=_user_ctx,
                    similar_sessions=_sessions_ctx,
                    policies=_policies_ctx,
                )

                result = await self._llm.complete_json(
                    prompt=llm_prompt,
                    system="You are a peer programming assistant that helps users think through their tasks before starting.",
                    temperature=0.2,
                    max_tokens=1024,
                )

                # Parse questions
                raw_questions = result.get("questions", [])
                for q in raw_questions:
                    if isinstance(q, dict) and q.get("text"):
                        priority_str = q.get("priority", "medium")
                        try:
                            priority = Priority(priority_str)
                        except ValueError:
                            priority = Priority.MEDIUM

                        analysis.questions.append(Question(
                            text=str(q["text"]),
                            reason=str(q.get("reason", "")),
                            priority=priority,
                            options=q.get("options", []),
                        ))

                analysis.predicted_plan = str(result.get("predicted_plan", ""))
                analysis.confidence = float(result.get("confidence", 0.5))

            except Exception as e:
                logger.warning("LLM question generation failed: %s", e)
                # Fall through to heuristic analysis

        # Heuristic fallback: generate basic questions from context
        if not analysis.questions:
            analysis.questions = self._heuristic_questions(
                prompt, analysis.similar_sessions, analysis.applicable_policies
            )
            analysis.confidence = 0.3  # Lower confidence for heuristics

        # Sort questions by priority
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        analysis.questions.sort(key=lambda q: priority_order[q.priority])

        return analysis

    def _heuristic_questions(
        self,
        prompt: str,
        similar_sessions: list[dict[str, Any]],
        policies: list[dict[str, Any]],
    ) -> list[Question]:
        """Generate basic questions without LLM.

        Looks for patterns that commonly need clarification.
        """
        questions: list[Question] = []
        prompt_lower = prompt.lower()

        # If similar sessions exist with different outcomes
        if len(similar_sessions) >= 2:
            questions.append(Question(
                text="Similar tasks have been done before. Should I reference the approach from past sessions?",
                reason=f"Found {len(similar_sessions)} similar past sessions",
                priority=Priority.LOW,
            ))

        # If there are applicable policies
        if policies:
            questions.append(Question(
                text="There are project/org policies that may apply. Should I review them before proceeding?",
                reason=f"{len(policies)} applicable policies found",
                priority=Priority.MEDIUM,
            ))

        # Common ambiguity patterns
        if any(kw in prompt_lower for kw in ["add", "implement", "create", "build"]):
            if "test" not in prompt_lower:
                questions.append(Question(
                    text="Should tests be included for this change?",
                    reason="New feature requested without mentioning tests",
                    priority=Priority.MEDIUM,
                    options=["Yes, add tests", "No, skip tests", "Only if complex"],
                ))

        if any(kw in prompt_lower for kw in ["fix", "bug", "broken", "error"]):
            questions.append(Question(
                text="Should I also add a regression test to prevent this bug from recurring?",
                reason="Bug fix without mention of testing",
                priority=Priority.LOW,
                options=["Yes", "No"],
            ))

        return questions[:3]  # Cap at 3 questions
