"""Tests for betty.policy module."""

from pathlib import Path

import tomli_w

from betty.policy import (
    Enforcement,
    Policy,
    PolicyEngine,
    PolicyScope,
    PolicyType,
    _db_row_to_policy,
    _parse_policy_dict,
    load_policies_from_toml,
)


class TestPolicyDataclass:
    def test_defaults(self):
        p = Policy(policy_type=PolicyType.CONVENTION, rule="Use snake_case")
        assert p.scope == PolicyScope.GLOBAL
        assert p.enforcement == Enforcement.SUGGEST
        assert p.keywords == []
        assert p.project_scope is None

    def test_matches_prompt_no_keywords(self):
        p = Policy(policy_type=PolicyType.TESTING, rule="Use pytest")
        assert p.matches_prompt("anything at all") is True

    def test_matches_prompt_with_keywords(self):
        p = Policy(
            policy_type=PolicyType.TESTING,
            rule="Use pytest",
            keywords=["test", "pytest"],
        )
        assert p.matches_prompt("write a test for the login flow") is True
        assert p.matches_prompt("fix the database migration") is False

    def test_matches_prompt_case_insensitive(self):
        p = Policy(
            policy_type=PolicyType.SECURITY,
            rule="No hardcoded secrets",
            keywords=["secret", "password", "api_key"],
        )
        assert p.matches_prompt("Update the API_KEY rotation") is True


class TestParsePolicyDict:
    def test_basic(self):
        data = {
            "type": "framework",
            "rule": "Use React for frontend",
            "description": "Standard frontend framework",
            "scope": "project",
            "enforcement": "warn",
        }
        p = _parse_policy_dict(data, source="test")
        assert p.policy_type == PolicyType.FRAMEWORK
        assert p.rule == "Use React for frontend"
        assert p.scope == PolicyScope.PROJECT
        assert p.enforcement == Enforcement.WARN
        assert p.source == "test"

    def test_defaults_for_missing_fields(self):
        data = {"rule": "Some rule"}
        p = _parse_policy_dict(data)
        assert p.policy_type == PolicyType.CONVENTION
        assert p.scope == PolicyScope.GLOBAL
        assert p.enforcement == Enforcement.SUGGEST

    def test_invalid_enum_values_use_defaults(self):
        data = {
            "type": "nonexistent_type",
            "scope": "galaxy",
            "enforcement": "destroy",
            "rule": "test",
        }
        p = _parse_policy_dict(data)
        assert p.policy_type == PolicyType.CONVENTION
        assert p.scope == PolicyScope.GLOBAL
        assert p.enforcement == Enforcement.SUGGEST

    def test_keywords_as_list(self):
        data = {"rule": "test", "keywords": ["react", "component"]}
        p = _parse_policy_dict(data)
        assert p.keywords == ["react", "component"]

    def test_keywords_as_comma_string(self):
        data = {"rule": "test", "keywords": "react, component, hook"}
        p = _parse_policy_dict(data)
        assert p.keywords == ["react", "component", "hook"]


class TestLoadPoliciesFromToml:
    def test_nonexistent_file(self):
        result = load_policies_from_toml(Path("/nonexistent/file.toml"))
        assert result == []

    def test_valid_toml(self, tmp_path):
        policies_file = tmp_path / "policies.toml"
        data = {
            "policies": [
                {
                    "type": "framework",
                    "rule": "Use pytest for all tests",
                    "enforcement": "warn",
                    "keywords": ["test", "pytest"],
                },
                {
                    "type": "security",
                    "rule": "Never commit secrets",
                    "enforcement": "block",
                },
            ]
        }
        with open(policies_file, "wb") as f:
            tomli_w.dump(data, f)

        result = load_policies_from_toml(policies_file)
        assert len(result) == 2
        assert result[0].policy_type == PolicyType.FRAMEWORK
        assert result[0].enforcement == Enforcement.WARN
        assert result[1].policy_type == PolicyType.SECURITY
        assert result[1].enforcement == Enforcement.BLOCK

    def test_invalid_toml(self, tmp_path):
        policies_file = tmp_path / "policies.toml"
        policies_file.write_text("this is not valid [[[toml")
        result = load_policies_from_toml(policies_file)
        assert result == []

    def test_missing_policies_key(self, tmp_path):
        policies_file = tmp_path / "policies.toml"
        data = {"other_key": "value"}
        with open(policies_file, "wb") as f:
            tomli_w.dump(data, f)
        result = load_policies_from_toml(policies_file)
        assert result == []


class TestDbRowToPolicy:
    def test_basic_conversion(self):
        row = {
            "policy_type": "testing",
            "rule": "Use pytest",
            "description": "Standard test framework",
            "project_scope": "/proj",
        }
        p = _db_row_to_policy(row)
        assert p.policy_type == PolicyType.TESTING
        assert p.rule == "Use pytest"
        assert p.source == "database"
        assert p.project_scope == "/proj"

    def test_unknown_type_defaults(self):
        row = {"policy_type": "unknown", "rule": "some rule"}
        p = _db_row_to_policy(row)
        assert p.policy_type == PolicyType.CONVENTION


class TestPolicyEngine:
    def test_empty_engine(self):
        engine = PolicyEngine()
        assert engine.policies == []

    def test_load_global_policies(self, tmp_path, monkeypatch):
        import betty.policy as policy_mod

        policies_file = tmp_path / "policies.toml"
        data = {
            "policies": [
                {"type": "convention", "rule": "Use snake_case"},
            ]
        }
        with open(policies_file, "wb") as f:
            tomli_w.dump(data, f)

        monkeypatch.setattr(policy_mod, "GLOBAL_POLICIES_FILE", policies_file)

        engine = PolicyEngine()
        result = engine.load_policies()
        assert len(result) == 1
        assert result[0].rule == "Use snake_case"

    def test_load_project_policies(self, tmp_path, monkeypatch):
        import betty.policy as policy_mod

        # Empty global policies
        monkeypatch.setattr(
            policy_mod, "GLOBAL_POLICIES_FILE", tmp_path / "nonexistent.toml"
        )

        # Project-level policies
        project_dir = tmp_path / "myproject"
        betty_dir = project_dir / ".betty"
        betty_dir.mkdir(parents=True)
        policies_file = betty_dir / "policies.toml"
        data = {
            "policies": [
                {"type": "framework", "rule": "Use React", "scope": "project"},
            ]
        }
        with open(policies_file, "wb") as f:
            tomli_w.dump(data, f)

        engine = PolicyEngine()
        result = engine.load_policies(project_dir=str(project_dir))
        assert len(result) == 1
        assert result[0].rule == "Use React"
        assert result[0].project_scope == str(project_dir)
        assert result[0].scope == PolicyScope.PROJECT

    def test_load_db_policies(self):
        engine = PolicyEngine()
        rows = [
            {"policy_type": "security", "rule": "No eval()", "description": ""},
            {"policy_type": "testing", "rule": "100% coverage", "description": ""},
        ]
        engine.load_db_policies(rows)
        assert len(engine.policies) == 2

    def test_add_policy(self):
        engine = PolicyEngine()
        p = Policy(policy_type=PolicyType.CONVENTION, rule="test")
        engine.add_policy(p)
        assert len(engine.policies) == 1

    def test_find_applicable_by_keywords(self, tmp_path, monkeypatch):
        import betty.policy as policy_mod

        monkeypatch.setattr(
            policy_mod, "GLOBAL_POLICIES_FILE", tmp_path / "nonexistent.toml"
        )

        engine = PolicyEngine()
        engine.load_policies()
        engine.add_policy(
            Policy(
                policy_type=PolicyType.TESTING,
                rule="Use pytest",
                keywords=["test", "pytest"],
            )
        )
        engine.add_policy(
            Policy(
                policy_type=PolicyType.SECURITY,
                rule="No hardcoded secrets",
                keywords=["secret", "password", "key"],
            )
        )

        result = engine.find_applicable("write a test for auth")
        assert len(result) == 1
        assert result[0].rule == "Use pytest"

    def test_find_applicable_project_scope(self):
        engine = PolicyEngine()
        engine.add_policy(
            Policy(
                policy_type=PolicyType.FRAMEWORK,
                rule="Use React",
                project_scope="/proj/a",
            )
        )
        engine.add_policy(
            Policy(
                policy_type=PolicyType.FRAMEWORK,
                rule="Use Vue",
                project_scope="/proj/b",
            )
        )
        engine.add_policy(
            Policy(policy_type=PolicyType.CONVENTION, rule="Use snake_case")
        )

        result = engine.find_applicable("create a component", project_dir="/proj/a")
        assert len(result) == 2  # React + snake_case (global)
        rules = [p.rule for p in result]
        assert "Use React" in rules
        assert "Use snake_case" in rules

    def test_find_applicable_no_project_context_skips_project_scoped(self):
        engine = PolicyEngine()
        engine.add_policy(
            Policy(
                policy_type=PolicyType.FRAMEWORK,
                rule="Use React",
                project_scope="/proj/a",
            )
        )
        engine.add_policy(
            Policy(policy_type=PolicyType.CONVENTION, rule="Global rule")
        )

        result = engine.find_applicable("build a component")
        assert len(result) == 1
        assert result[0].rule == "Global rule"

    def test_find_by_type(self):
        engine = PolicyEngine()
        engine.add_policy(
            Policy(policy_type=PolicyType.TESTING, rule="Use pytest")
        )
        engine.add_policy(
            Policy(policy_type=PolicyType.SECURITY, rule="No eval()")
        )
        engine.add_policy(
            Policy(policy_type=PolicyType.TESTING, rule="Mock external APIs")
        )

        result = engine.find_by_type(PolicyType.TESTING)
        assert len(result) == 2

    def test_find_blocking(self):
        engine = PolicyEngine()
        engine.add_policy(
            Policy(
                policy_type=PolicyType.SECURITY,
                rule="No eval()",
                enforcement=Enforcement.BLOCK,
            )
        )
        engine.add_policy(
            Policy(
                policy_type=PolicyType.CONVENTION,
                rule="Use snake_case",
                enforcement=Enforcement.SUGGEST,
            )
        )

        result = engine.find_blocking()
        assert len(result) == 1
        assert result[0].rule == "No eval()"


class TestFormatForInjection:
    def test_empty_policies(self):
        engine = PolicyEngine()
        assert engine.format_for_injection([]) == ""

    def test_format_groups_by_type(self):
        engine = PolicyEngine()
        policies = [
            Policy(policy_type=PolicyType.TESTING, rule="Use pytest"),
            Policy(policy_type=PolicyType.SECURITY, rule="No eval()"),
            Policy(
                policy_type=PolicyType.TESTING,
                rule="Mock external APIs",
            ),
        ]

        result = engine.format_for_injection(policies)
        assert "## Active Policies" in result
        assert "### Testing" in result
        assert "### Security" in result
        assert "- Use pytest" in result
        assert "- No eval()" in result

    def test_format_enforcement_tags(self):
        engine = PolicyEngine()
        policies = [
            Policy(
                policy_type=PolicyType.SECURITY,
                rule="No secrets",
                enforcement=Enforcement.BLOCK,
            ),
            Policy(
                policy_type=PolicyType.CONVENTION,
                rule="Use snake_case",
                enforcement=Enforcement.WARN,
            ),
            Policy(
                policy_type=PolicyType.CONVENTION,
                rule="Add docstrings",
                enforcement=Enforcement.SUGGEST,
            ),
        ]

        result = engine.format_for_injection(policies)
        assert "[REQUIRED]" in result
        assert "[WARNING]" in result
        # SUGGEST has no tag
        assert "- Add docstrings\n" in result

    def test_format_with_descriptions(self):
        engine = PolicyEngine()
        policies = [
            Policy(
                policy_type=PolicyType.FRAMEWORK,
                rule="Use React",
                description="Standard frontend framework for all projects",
            ),
        ]

        result = engine.format_for_injection(policies)
        assert "Standard frontend framework" in result

    def test_format_none_uses_all_policies(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(policy_type=PolicyType.TESTING, rule="Use pytest"))
        engine.add_policy(Policy(policy_type=PolicyType.SECURITY, rule="No eval()"))

        result = engine.format_for_injection()
        assert "Use pytest" in result
        assert "No eval()" in result
