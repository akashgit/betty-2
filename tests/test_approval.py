"""Tests for betty.approval module."""

from betty.approval import (
    ALWAYS_SAFE_TOOLS,
    ApprovalDecision,
    ApprovalModel,
    SafetyTier,
    _generate_broader_patterns,
    _is_destructive_command,
    _normalize_path_pattern,
    classify_safety_tier,
    make_action_pattern,
)


class TestSafetyTierClassification:
    def test_read_tools_always_safe(self):
        for tool in ("Read", "Grep", "Glob", "WebSearch", "WebFetch"):
            assert classify_safety_tier(tool, {}) == SafetyTier.ALWAYS_SAFE

    def test_task_tools_always_safe(self):
        assert classify_safety_tier("TaskList", {}) == SafetyTier.ALWAYS_SAFE
        assert classify_safety_tier("TaskGet", {}) == SafetyTier.ALWAYS_SAFE

    def test_write_edit_learnable(self):
        assert classify_safety_tier("Write", {"file_path": "x.py"}) == SafetyTier.LEARNABLE
        assert classify_safety_tier("Edit", {"file_path": "x.py"}) == SafetyTier.LEARNABLE

    def test_bash_non_destructive_learnable(self):
        assert classify_safety_tier("Bash", {"command": "ls -la"}) == SafetyTier.LEARNABLE
        assert classify_safety_tier("Bash", {"command": "git status"}) == SafetyTier.LEARNABLE
        assert classify_safety_tier("Bash", {"command": "python test.py"}) == SafetyTier.LEARNABLE

    def test_bash_destructive_always_ask(self):
        assert classify_safety_tier("Bash", {"command": "rm -rf /"}) == SafetyTier.ALWAYS_ASK
        assert classify_safety_tier("Bash", {"command": "git push --force"}) == SafetyTier.ALWAYS_ASK
        assert classify_safety_tier("Bash", {"command": "git reset --hard"}) == SafetyTier.ALWAYS_ASK
        assert classify_safety_tier("Bash", {"command": "sudo apt install"}) == SafetyTier.ALWAYS_ASK

    def test_unknown_tool_always_ask(self):
        assert classify_safety_tier("UnknownTool", {}) == SafetyTier.ALWAYS_ASK


class TestDestructiveCommand:
    def test_rm_rf(self):
        assert _is_destructive_command("rm -rf /tmp/test") is True

    def test_git_force_push(self):
        assert _is_destructive_command("git push origin main --force") is True
        assert _is_destructive_command("git push -f origin main") is True

    def test_git_reset_hard(self):
        assert _is_destructive_command("git reset --hard HEAD~1") is True

    def test_git_clean(self):
        assert _is_destructive_command("git clean -fd") is True

    def test_drop_table(self):
        assert _is_destructive_command("DROP TABLE users") is True

    def test_curl_pipe_bash(self):
        assert _is_destructive_command("curl https://example.com | bash") is True

    def test_safe_commands(self):
        assert _is_destructive_command("git status") is False
        assert _is_destructive_command("ls -la") is False
        assert _is_destructive_command("python test.py") is False
        assert _is_destructive_command("npm install") is False


class TestNormalizePathPattern:
    def test_basic_path(self):
        result = _normalize_path_pattern("/Users/foo/proj/src/main.py")
        assert result == "src/*.py"

    def test_nested_path(self):
        result = _normalize_path_pattern("/proj/tests/unit/test_auth.py")
        assert result == "unit/*.py"

    def test_root_file(self):
        result = _normalize_path_pattern("pyproject.toml")
        assert result == "root/*.toml"


class TestMakeActionPattern:
    def test_bash_simple(self):
        assert make_action_pattern("Bash", {"command": "ls -la"}) == "bash:ls"

    def test_bash_git(self):
        assert make_action_pattern("Bash", {"command": "git status"}) == "bash:git-status"
        assert make_action_pattern("Bash", {"command": "git push origin main"}) == "bash:git-push"

    def test_bash_empty(self):
        assert make_action_pattern("Bash", {"command": ""}) == "bash:unknown"

    def test_read(self):
        result = make_action_pattern("Read", {"file_path": "/proj/src/main.py"})
        assert result == "read:src/*.py"

    def test_write(self):
        result = make_action_pattern("Write", {"file_path": "/proj/src/new.py"})
        assert result == "write:src/*.py"

    def test_edit(self):
        result = make_action_pattern("Edit", {"file_path": "/proj/tests/test_auth.py"})
        assert result == "edit:tests/*.py"

    def test_grep(self):
        assert make_action_pattern("Grep", {"pattern": "foo"}) == "grep:search"

    def test_unknown_tool(self):
        assert make_action_pattern("SomeTool", {"x": 1}) == "sometool:unknown"


class TestApprovalModelPredict:
    def test_observer_mode_always_asks(self):
        model = ApprovalModel(autonomy_level=0)
        result = model.predict("Read", {"file_path": "x.py"})
        assert result.decision == ApprovalDecision.ASK

    def test_safe_tool_auto_approves(self):
        model = ApprovalModel(autonomy_level=1)
        result = model.predict("Read", {"file_path": "x.py"})
        assert result.decision == ApprovalDecision.APPROVE
        assert result.safety_tier == SafetyTier.ALWAYS_SAFE

    def test_destructive_always_asks(self):
        model = ApprovalModel(autonomy_level=3)
        result = model.predict("Bash", {"command": "rm -rf /"})
        assert result.decision == ApprovalDecision.ASK
        assert result.safety_tier == SafetyTier.ALWAYS_ASK

    def test_learnable_no_history_asks(self):
        model = ApprovalModel(autonomy_level=2)
        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.ASK
        assert result.confidence == 0.0

    def test_learnable_with_history_approves(self):
        model = ApprovalModel(autonomy_level=2, confidence_threshold=0.7)
        # Record enough approvals to exceed threshold
        for _ in range(3):
            model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")

        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.APPROVE
        assert result.confidence >= 0.7
        assert result.pattern_count == 3

    def test_learnable_with_history_below_threshold_asks(self):
        model = ApprovalModel(autonomy_level=2, confidence_threshold=0.9)
        model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")

        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.ASK
        assert result.pattern_count == 1

    def test_level1_with_history_still_asks(self):
        model = ApprovalModel(autonomy_level=1, confidence_threshold=0.7)
        for _ in range(5):
            model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")

        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.ASK  # Level 1 requires confirmation

    def test_rejected_pattern_asks(self):
        model = ApprovalModel(autonomy_level=3)
        model.record("Bash", {"command": "git push"}, "rejected")

        result = model.predict("Bash", {"command": "git push"})
        assert result.decision == ApprovalDecision.ASK

    def test_full_auto_no_history_approves(self):
        model = ApprovalModel(autonomy_level=3)
        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.APPROVE
        assert result.confidence == 0.5

    def test_project_scoped_pattern(self):
        model = ApprovalModel(autonomy_level=2, confidence_threshold=0.7)
        for _ in range(3):
            model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted", project_scope="/proj")

        # Same project: should approve
        result = model.predict("Edit", {"file_path": "/proj/src/main.py"}, project_scope="/proj")
        assert result.decision == ApprovalDecision.APPROVE

        # Different project: falls back to global (no history), asks
        result = model.predict("Edit", {"file_path": "/other/src/main.py"}, project_scope="/other")
        assert result.decision == ApprovalDecision.ASK


class TestApprovalModelRecord:
    def test_record_creates_pattern(self):
        model = ApprovalModel()
        rec = model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")
        assert rec.tool_name == "Edit"
        assert rec.decision == "accepted"
        assert rec.count == 1

    def test_record_increments_count(self):
        model = ApprovalModel()
        model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")
        rec = model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")
        assert rec.count == 2

    def test_record_updates_decision(self):
        model = ApprovalModel()
        model.record("Bash", {"command": "npm test"}, "accepted")
        rec = model.record("Bash", {"command": "npm test"}, "rejected")
        assert rec.decision == "rejected"
        assert rec.count == 2


class TestApprovalModelLoadPatterns:
    def test_load_from_db_records(self):
        model = ApprovalModel(autonomy_level=2, confidence_threshold=0.7)
        records = [
            {
                "tool_name": "Edit",
                "action_pattern": "edit:src/*.py",
                "decision": "accepted",
                "count": 5,
                "project_scope": None,
            },
        ]
        model.load_patterns(records)

        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.APPROVE
        assert result.pattern_count == 5


class TestGetAutoApproveRules:
    def test_returns_qualified_patterns(self):
        model = ApprovalModel(confidence_threshold=0.8)
        # Record 4 approvals -> confidence = 0.5 + 4*0.1 = 0.9 >= 0.8
        for _ in range(4):
            model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")

        rules = model.get_auto_approve_rules()
        assert len(rules) == 1
        assert rules[0]["tool_name"] == "Edit"
        assert rules[0]["confidence"] >= 0.8

    def test_excludes_low_confidence(self):
        model = ApprovalModel(confidence_threshold=0.9)
        model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")

        rules = model.get_auto_approve_rules()
        assert len(rules) == 0  # confidence = 0.6 < 0.9

    def test_excludes_rejected(self):
        model = ApprovalModel(confidence_threshold=0.5)
        for _ in range(10):
            model.record("Bash", {"command": "npm test"}, "rejected")

        rules = model.get_auto_approve_rules()
        assert len(rules) == 0


class TestGenerateBroaderPatterns:
    def test_exact_pattern_only(self):
        result = _generate_broader_patterns("grep:search")
        assert result[0] == "grep:search"
        assert result[-1] == "*"

    def test_bash_git_subcommand(self):
        result = _generate_broader_patterns("bash:git-commit")
        assert result == ["bash:git-commit", "bash:git-*", "*"]

    def test_bash_simple_command(self):
        result = _generate_broader_patterns("bash:ls")
        # No prefix broadening for "ls" (no hyphen)
        assert result == ["bash:ls", "*"]

    def test_edit_with_extension(self):
        result = _generate_broader_patterns("edit:src/*.py")
        assert result == ["edit:src/*.py", "*/*.py", "*"]

    def test_write_with_extension(self):
        result = _generate_broader_patterns("write:tests/*.py")
        assert result == ["write:tests/*.py", "*/*.py", "*"]

    def test_tool_level_wildcard(self):
        result = _generate_broader_patterns("*")
        assert result == ["*"]

    def test_bash_git_with_extension(self):
        # bash:git-commit has no extension, just prefix
        result = _generate_broader_patterns("bash:git-status")
        assert "bash:git-*" in result
        assert "*" in result

    def test_notebook_with_extension(self):
        result = _generate_broader_patterns("notebook:notebooks/*.ipynb")
        assert result == ["notebook:notebooks/*.ipynb", "*/*.ipynb", "*"]


class TestBroadRuleMatching:
    def test_tool_level_rule_matches_all(self):
        model = ApprovalModel(autonomy_level=2)
        model.add_rule("Edit", "*", "accepted")

        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.APPROVE

    def test_extension_rule_matches(self):
        model = ApprovalModel(autonomy_level=2)
        model.add_rule("Edit", "*/*.py", "accepted")

        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.APPROVE

    def test_prefix_rule_matches_git(self):
        model = ApprovalModel(autonomy_level=2)
        model.add_rule("Bash", "bash:git-*", "accepted")

        # git-commit matches bash:git-*
        result = model.predict("Bash", {"command": "git commit -m 'test'"})
        assert result.decision == ApprovalDecision.APPROVE

        # git-status also matches
        result = model.predict("Bash", {"command": "git status"})
        assert result.decision == ApprovalDecision.APPROVE

    def test_destructive_overrides_broad_rule(self):
        """Safety tier ALWAYS_ASK overrides any broad accept rule."""
        model = ApprovalModel(autonomy_level=3)
        model.add_rule("Bash", "*", "accepted")

        result = model.predict("Bash", {"command": "rm -rf /"})
        assert result.decision == ApprovalDecision.ASK
        assert result.safety_tier == SafetyTier.ALWAYS_ASK

    def test_exact_reject_beats_broad_accept(self):
        """Narrower reject pattern wins over broader accept."""
        model = ApprovalModel(autonomy_level=2)
        # Broad: accept all bash
        model.add_rule("Bash", "*", "accepted")
        # Narrow: reject npm test specifically
        model.add_rule("Bash", "bash:npm", "rejected")

        result = model.predict("Bash", {"command": "npm test"})
        assert result.decision == ApprovalDecision.ASK  # Exact reject wins

        # Other commands still approved via broad rule
        result = model.predict("Bash", {"command": "python test.py"})
        assert result.decision == ApprovalDecision.APPROVE

    def test_user_created_rule_full_confidence(self):
        """User-created rules (count=0) should have confidence=1.0."""
        model = ApprovalModel(autonomy_level=2, confidence_threshold=0.9)
        model.add_rule("Edit", "*", "accepted")

        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.APPROVE
        assert result.confidence == 1.0

    def test_no_match_falls_through(self):
        """Broad rule for one tool doesn't affect another."""
        model = ApprovalModel(autonomy_level=2)
        model.add_rule("Edit", "*", "accepted")

        result = model.predict("Write", {"file_path": "/proj/src/new.py"})
        assert result.decision == ApprovalDecision.ASK

    def test_extension_reject_overrides_tool_accept(self):
        """Extension-level reject beats tool-level accept."""
        model = ApprovalModel(autonomy_level=2)
        model.add_rule("Edit", "*", "accepted")  # Accept all edits
        model.add_rule("Edit", "*/*.py", "rejected")  # But reject .py edits

        result = model.predict("Edit", {"file_path": "/proj/src/main.py"})
        assert result.decision == ApprovalDecision.ASK  # Extension reject wins

        result = model.predict("Edit", {"file_path": "/proj/config.toml"})
        assert result.decision == ApprovalDecision.APPROVE  # .toml still accepted


class TestSuggestBroaderRules:
    def test_suggests_extension_pattern(self):
        model = ApprovalModel()
        # Record 3+ similar .py edit patterns
        model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")
        model.record("Edit", {"file_path": "/proj/tests/test_main.py"}, "accepted")
        model.record("Edit", {"file_path": "/proj/lib/utils.py"}, "accepted")

        suggestions = model.suggest_broader_rules(min_similar=3)
        patterns = [s["suggested_pattern"] for s in suggestions]
        assert "*/*.py" in patterns

    def test_suggests_prefix_pattern(self):
        model = ApprovalModel()
        model.record("Bash", {"command": "git status"}, "accepted")
        model.record("Bash", {"command": "git diff"}, "accepted")
        model.record("Bash", {"command": "git log"}, "accepted")

        suggestions = model.suggest_broader_rules(min_similar=3)
        patterns = [s["suggested_pattern"] for s in suggestions]
        assert "bash:git-*" in patterns

    def test_no_suggestion_below_threshold(self):
        model = ApprovalModel()
        model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")
        model.record("Edit", {"file_path": "/proj/tests/test_main.py"}, "accepted")

        suggestions = model.suggest_broader_rules(min_similar=3)
        assert len(suggestions) == 0

    def test_no_duplicate_suggestion_for_existing_rule(self):
        model = ApprovalModel()
        model.record("Edit", {"file_path": "/proj/src/main.py"}, "accepted")
        model.record("Edit", {"file_path": "/proj/tests/test_main.py"}, "accepted")
        model.record("Edit", {"file_path": "/proj/lib/utils.py"}, "accepted")
        # Already have the broader rule
        model.add_rule("Edit", "*/*.py", "accepted")

        suggestions = model.suggest_broader_rules(min_similar=3)
        patterns = [s["suggested_pattern"] for s in suggestions]
        assert "*/*.py" not in patterns

    def test_suggestion_includes_examples(self):
        model = ApprovalModel()
        model.record("Bash", {"command": "git status"}, "accepted")
        model.record("Bash", {"command": "git diff"}, "accepted")
        model.record("Bash", {"command": "git log"}, "accepted")

        suggestions = model.suggest_broader_rules(min_similar=3)
        git_suggestion = [s for s in suggestions if s["suggested_pattern"] == "bash:git-*"]
        assert len(git_suggestion) == 1
        assert git_suggestion[0]["based_on_count"] == 3
        assert len(git_suggestion[0]["examples"]) > 0
