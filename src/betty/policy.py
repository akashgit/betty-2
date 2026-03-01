"""Policy engine for Betty 2.0.

Loads and applies organizational/project policies to guide Claude Code behavior.
Policies come from three sources:
1. Manual config (~/.betty/policies.toml)
2. Database (org_policies table)
3. Project-level config (.betty/policies.toml in project dir)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from betty.config import BETTY_DIR

logger = logging.getLogger(__name__)

GLOBAL_POLICIES_FILE = BETTY_DIR / "policies.toml"


class PolicyType(Enum):
    """Types of policies Betty can enforce."""

    FRAMEWORK = "framework"
    DEPENDENCY = "dependency"
    TESTING = "testing"
    SECURITY = "security"
    CONVENTION = "convention"
    REVIEW = "review"


class PolicyScope(Enum):
    """Scope at which a policy applies."""

    GLOBAL = "global"
    ORG = "org"
    PROJECT = "project"


class Enforcement(Enum):
    """How strictly a policy is enforced."""

    SUGGEST = "suggest"  # Mention in context, don't block
    WARN = "warn"  # Flag violations but allow
    BLOCK = "block"  # Prevent actions that violate


@dataclass
class Policy:
    """A single policy rule."""

    policy_type: PolicyType
    rule: str
    description: str = ""
    scope: PolicyScope = PolicyScope.GLOBAL
    enforcement: Enforcement = Enforcement.SUGGEST
    source: str = ""  # Where this policy came from
    project_scope: str | None = None  # Specific project dir, or None for global
    keywords: list[str] = field(default_factory=list)

    def matches_prompt(self, prompt: str) -> bool:
        """Check if this policy is relevant to a given prompt.

        Uses keyword matching on the prompt text.
        If no keywords are defined, the policy always matches.
        """
        if not self.keywords:
            return True
        prompt_lower = prompt.lower()
        return any(kw.lower() in prompt_lower for kw in self.keywords)


def _parse_policy_dict(data: dict[str, Any], source: str = "") -> Policy:
    """Parse a single policy from a dict (e.g., from TOML)."""
    policy_type_str = data.get("type", "convention")
    try:
        policy_type = PolicyType(policy_type_str)
    except ValueError:
        policy_type = PolicyType.CONVENTION

    scope_str = data.get("scope", "global")
    try:
        scope = PolicyScope(scope_str)
    except ValueError:
        scope = PolicyScope.GLOBAL

    enforcement_str = data.get("enforcement", "suggest")
    try:
        enforcement = Enforcement(enforcement_str)
    except ValueError:
        enforcement = Enforcement.SUGGEST

    keywords = data.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    return Policy(
        policy_type=policy_type,
        rule=data.get("rule", ""),
        description=data.get("description", ""),
        scope=scope,
        enforcement=enforcement,
        source=source,
        project_scope=data.get("project_scope"),
        keywords=keywords,
    )


def load_policies_from_toml(path: Path) -> list[Policy]:
    """Load policies from a TOML file.

    Expected format:
        [[policies]]
        type = "framework"
        rule = "Use pytest for all tests"
        description = "Project standard is pytest, not unittest"
        scope = "project"
        enforcement = "warn"
        keywords = ["test", "pytest", "unittest"]
    """
    if not path.exists():
        return []

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.warning("Failed to parse policies from %s", path)
        return []

    policies_data = data.get("policies", [])
    if not isinstance(policies_data, list):
        logger.warning("Expected [[policies]] array in %s", path)
        return []

    source = str(path)
    return [_parse_policy_dict(p, source=source) for p in policies_data]


def _db_row_to_policy(row: dict[str, Any]) -> Policy:
    """Convert a database org_policies row to a Policy."""
    policy_type_str = row.get("policy_type", "convention")
    try:
        policy_type = PolicyType(policy_type_str)
    except ValueError:
        policy_type = PolicyType.CONVENTION

    return Policy(
        policy_type=policy_type,
        rule=row.get("rule", ""),
        description=row.get("description", ""),
        scope=PolicyScope.GLOBAL,
        enforcement=Enforcement.SUGGEST,
        source="database",
        project_scope=row.get("project_scope"),
    )


class PolicyEngine:
    """Loads, stores, and queries policies from all sources."""

    def __init__(self) -> None:
        self._policies: list[Policy] = []

    @property
    def policies(self) -> list[Policy]:
        """All loaded policies."""
        return list(self._policies)

    def load_policies(self, project_dir: str | None = None) -> list[Policy]:
        """Load policies from all file-based sources.

        Sources checked:
        1. Global policies file (~/.betty/policies.toml)
        2. Project-level policies (<project_dir>/.betty/policies.toml)

        Database policies should be loaded separately via load_db_policies().
        """
        self._policies = []

        # 1. Global policies
        global_policies = load_policies_from_toml(GLOBAL_POLICIES_FILE)
        self._policies.extend(global_policies)

        # 2. Project-level policies
        if project_dir:
            project_policy_file = Path(project_dir) / ".betty" / "policies.toml"
            project_policies = load_policies_from_toml(project_policy_file)
            # Tag with project scope
            for p in project_policies:
                if p.project_scope is None:
                    p.project_scope = project_dir
                if p.scope == PolicyScope.GLOBAL:
                    p.scope = PolicyScope.PROJECT
            self._policies.extend(project_policies)

        return list(self._policies)

    def load_db_policies(self, db_rows: list[dict[str, Any]]) -> None:
        """Load policies from database query results.

        Call this after querying db.get_policies() to merge DB policies
        with file-based ones.
        """
        for row in db_rows:
            self._policies.append(_db_row_to_policy(row))

    def add_policy(self, policy: Policy) -> None:
        """Add a single policy to the engine."""
        self._policies.append(policy)

    def find_applicable(
        self,
        prompt: str,
        project_dir: str | None = None,
    ) -> list[Policy]:
        """Find policies applicable to a given prompt and context.

        Filters by:
        1. Keyword matching against the prompt
        2. Project scope (global policies + matching project policies)
        """
        applicable = []
        for policy in self._policies:
            # Check project scope
            if policy.project_scope and project_dir:
                if policy.project_scope != project_dir:
                    continue
            elif policy.project_scope and not project_dir:
                # Policy is project-scoped but no project context
                continue

            # Check keyword match
            if policy.matches_prompt(prompt):
                applicable.append(policy)

        return applicable

    def find_by_type(self, policy_type: PolicyType) -> list[Policy]:
        """Get all policies of a specific type."""
        return [p for p in self._policies if p.policy_type == policy_type]

    def find_blocking(self) -> list[Policy]:
        """Get all policies with block enforcement."""
        return [p for p in self._policies if p.enforcement == Enforcement.BLOCK]

    def format_for_injection(self, policies: list[Policy] | None = None) -> str:
        """Format policies as text for injection into LLM prompts.

        Groups policies by type and formats them clearly.
        """
        if policies is None:
            policies = self._policies

        if not policies:
            return ""

        # Group by type
        by_type: dict[str, list[Policy]] = {}
        for p in policies:
            type_name = p.policy_type.value
            by_type.setdefault(type_name, []).append(p)

        lines = ["## Active Policies", ""]

        for type_name, type_policies in sorted(by_type.items()):
            lines.append(f"### {type_name.title()}")
            for p in type_policies:
                enforcement_tag = ""
                if p.enforcement == Enforcement.BLOCK:
                    enforcement_tag = " [REQUIRED]"
                elif p.enforcement == Enforcement.WARN:
                    enforcement_tag = " [WARNING]"

                lines.append(f"- {p.rule}{enforcement_tag}")
                if p.description:
                    lines.append(f"  {p.description}")
            lines.append("")

        return "\n".join(lines)
