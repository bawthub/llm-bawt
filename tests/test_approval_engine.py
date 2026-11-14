"""Unit tests for the pure approval-policy engine (agent_bridge.approval).

Pure, dependency-free assertions — runnable under pytest, or standalone via
``python tests/test_approval_engine.py`` when pytest isn't installed (the
bridge/app containers don't ship it).
"""

from __future__ import annotations

from agent_bridge.approval import (
    ApprovalPolicy,
    MatcherType,
    PolicyAction,
    PolicyBundle,
    Severity,
    compute_etag,
    derive_subject,
    evaluate,
    grant_key,
    invocation_key,
    matchable_subject,
    strip_inert_heredoc_bodies,
)


def _pol(**kw) -> ApprovalPolicy:
    base = dict(id="p1", tool_name="Bash", matcher_type=MatcherType.ALWAYS)
    base.update(kw)
    # allow string enums through the tolerant coercers
    if isinstance(base.get("matcher_type"), str):
        base["matcher_type"] = MatcherType.coerce(base["matcher_type"])
    if isinstance(base.get("action"), str):
        base["action"] = PolicyAction.coerce(base["action"])
    if isinstance(base.get("severity"), str):
        base["severity"] = Severity.coerce(base["severity"])
    return ApprovalPolicy(**base)


# ---- derive_subject --------------------------------------------------------

def test_derive_subject_bash_defaults_to_command():
    assert derive_subject("Bash", {"command": "rm -rf /x"}, None) == "rm -rf /x"


def test_derive_subject_mcp_namespaced_bash():
    assert derive_subject("mcp__srv__Bash", {"command": "ls"}, None) == "ls"


def test_derive_subject_explicit_field():
    assert derive_subject("Write", {"file_path": "/etc/passwd"}, "file_path") == "/etc/passwd"


def test_derive_subject_whole_input_json():
    s = derive_subject("Write", {"b": 2, "a": 1}, "*")
    assert s == '{"a": 1, "b": 2}'  # sorted keys


def test_derive_subject_missing_field_is_empty():
    assert derive_subject("Write", {"x": 1}, "file_path") == ""


def test_derive_subject_ops_run_is_canonical_and_ignores_idempotency_key():
    subject = derive_subject(
        "mcp__bawthub__ops_run",
        {
            "idempotency_key": "tool-use-123",
            "args": {"z": 2, "a": "x"},
            "operation": "llm-bawt.restart-bridge",
        },
        None,
    )
    assert subject == 'operation=llm-bawt.restart-bridge args={"a":"x","z":2}'


def test_derive_subject_ops_run_missing_args_is_empty_object():
    assert derive_subject("ops_run", {"operation": "llm-bawt.restart-redis"}, None) == (
        "operation=llm-bawt.restart-redis args={}"
    )


def test_derive_subject_ops_run_explicit_field_still_wins():
    assert derive_subject(
        "ops_run",
        {"operation": "llm-bawt.restart-app", "args": {}},
        "operation",
    ) == "llm-bawt.restart-app"


# ---- matcher semantics -----------------------------------------------------

def test_always_matches():
    assert _pol(matcher_type=MatcherType.ALWAYS).matches_subject("anything")


def test_exact_match_trims():
    p = _pol(matcher_type=MatcherType.EXACT, pattern="rm -rf /")
    assert p.matches_subject("  rm -rf / ")
    assert not p.matches_subject("rm -rf /tmp")


def test_prefix_match():
    p = _pol(matcher_type=MatcherType.PREFIX, pattern="sudo ")
    assert p.matches_subject("sudo apt update")
    assert p.matches_subject("   sudo apt update")  # leading ws stripped
    assert not p.matches_subject("echo sudo")


def test_contains_match():
    p = _pol(matcher_type=MatcherType.CONTAINS, pattern="rm -rf")
    assert p.matches_subject("cd /tmp && rm -rf foo")


def test_glob_match():
    p = _pol(matcher_type=MatcherType.GLOB, pattern="git push*")
    assert p.matches_subject("git push origin main")
    assert not p.matches_subject("git status")


def test_regex_match():
    p = _pol(matcher_type=MatcherType.REGEX, pattern=r"\brm\b.*-[a-z]*f")
    assert p.matches_subject("rm -rf /x")
    assert not p.matches_subject("confirm something")


def test_bad_regex_never_throws_never_matches():
    p = _pol(matcher_type=MatcherType.REGEX, pattern="(unclosed")
    # must not raise, and must not match
    assert p.matches_subject("(unclosed") is False


# ---- applies_to scoping ----------------------------------------------------

def test_backend_scope_wildcard_and_specific():
    p = _pol(backend_scope="claude-code")
    assert p.applies_to("claude-code", "Bash")
    assert not p.applies_to("codex", "Bash")
    assert _pol(backend_scope="*").applies_to("codex", "Bash")


def test_tool_wildcard():
    p = _pol(tool_name="*")
    assert p.applies_to("claude-code", "Write")


def test_disabled_policy_never_applies():
    assert not _pol(enabled=False).applies_to("claude-code", "Bash")


# ---- evaluate: ordering + first-match-wins ---------------------------------

def test_default_allow_when_no_policy():
    d = evaluate([], "claude-code", "Bash", {"command": "ls"})
    assert d.action is PolicyAction.ALLOW
    assert d.policy is None


# ---- evaluate: fail-closed tools (TASK-639) --------------------------------

def test_ops_run_fails_closed_when_no_policy_matches():
    """A catalogued operation is never ungated by a missing policy row."""
    d = evaluate([], "mcp", "ops_run", {"operation": "llm-bawt.restart-redis"})
    assert d.action is PolicyAction.REQUIRE_APPROVAL
    assert d.policy is None
    assert d.severity is Severity.HIGH
    assert d.subject == 'operation=llm-bawt.restart-redis args={}'
    assert d.grant_key


def test_ops_run_fail_closed_matches_namespaced_mcp_tool_name():
    d = evaluate([], "mcp", "mcp__bawthub__ops_run", {"operation": "x"})
    assert d.action is PolicyAction.REQUIRE_APPROVAL


def test_ops_run_explicit_allow_policy_still_wins_over_fail_closed():
    """Operators can still carve out a safe operation with an allow rule."""
    pols = [_pol(
        id="allow-safe", tool_name="ops_run", matcher_type=MatcherType.PREFIX,
        pattern="operation=llm-bawt.restart-aux ", action="allow", order=5,
    )]
    d = evaluate(pols, "mcp", "ops_run",
                 {"operation": "llm-bawt.restart-aux", "args": {"service": "crawl4ai"}})
    assert d.action is PolicyAction.ALLOW
    assert d.policy is not None


def test_ops_read_only_tools_are_not_fail_closed():
    for name in ("ops_list_operations", "ops_job_status"):
        d = evaluate([], "mcp", name, {"job_id": "j1"})
        assert d.action is PolicyAction.ALLOW, name


def test_require_approval_match():
    pols = [_pol(matcher_type=MatcherType.PREFIX, pattern="rm -rf", action="require_approval")]
    d = evaluate(pols, "claude-code", "Bash", {"command": "rm -rf /x"})
    assert d.requires_approval
    assert d.subject == "rm -rf /x"
    assert d.grant_key  # populated
    assert "rm -rf /x" in d.prompt


def test_allow_rule_overrides_lower_require_rule_by_order():
    # order 10 allow carves a safe hole out of the order 100 require-all rule
    allow_rule = _pol(id="a", order=10, matcher_type=MatcherType.PREFIX,
                      pattern="git status", action="allow")
    require_rule = _pol(id="b", order=100, matcher_type=MatcherType.PREFIX,
                        pattern="git", action="require_approval")
    pols = [require_rule, allow_rule]  # deliberately out of order in the list
    d = evaluate(pols, "claude-code", "Bash", {"command": "git status -s"})
    assert d.is_allowed
    # but a different git command still trips the require rule
    d2 = evaluate(pols, "claude-code", "Bash", {"command": "git push"})
    assert d2.requires_approval


def test_deny_action():
    pols = [_pol(matcher_type=MatcherType.CONTAINS, pattern="rm -rf /", action="deny",
                 severity="critical")]
    d = evaluate(pols, "claude-code", "Bash", {"command": "rm -rf / --no-preserve-root"})
    assert d.is_denied
    assert d.severity is Severity.CRITICAL


def test_scope_filters_out_other_backend():
    pols = [_pol(backend_scope="codex", matcher_type=MatcherType.ALWAYS,
                 action="require_approval")]
    d = evaluate(pols, "claude-code", "Bash", {"command": "anything"})
    assert d.is_allowed  # codex-scoped rule doesn't apply to claude-code


# ---- grant key stability ---------------------------------------------------

def test_grant_key_stable_across_whitespace():
    k1 = grant_key("claude-code", "Bash", "rm  -rf   /x")
    k2 = grant_key("claude-code", "Bash", "rm -rf /x")
    assert k1 == k2


def test_grant_key_differs_by_command():
    assert grant_key("claude-code", "Bash", "rm -rf /x") != grant_key(
        "claude-code", "Bash", "rm -rf /y"
    )


def test_grant_key_ignores_mcp_namespacing():
    assert grant_key("claude-code", "mcp__s__Bash", "ls") == grant_key(
        "claude-code", "Bash", "ls"
    )


def test_evaluate_grant_key_matches_standalone():
    d = evaluate([_pol(action="require_approval")], "claude-code", "Bash", {"command": "x y"})
    assert d.grant_key == grant_key("claude-code", "Bash", "x y")


# ---- serialization round-trips ---------------------------------------------

def test_policy_dict_roundtrip():
    p = _pol(matcher_type=MatcherType.REGEX, pattern="rm.*", action="deny",
             severity="high", category="filesystem", approval_prompt="careful!",
             order=5, version=3)
    p2 = ApprovalPolicy.from_dict(p.to_dict())
    assert p2 == p


def test_bundle_roundtrip_and_etag_deterministic():
    pols = [_pol(id="a"), _pol(id="b", order=5)]
    etag = compute_etag(2, pols)
    b = PolicyBundle(version=2, etag=etag, policies=pols)
    b2 = PolicyBundle.from_dict(b.to_dict())
    assert b2.policies == pols
    assert b2.version == 2
    # etag stable regardless of input ordering
    assert compute_etag(2, list(reversed(pols))) == etag


def test_etag_changes_with_content():
    a = compute_etag(1, [_pol(id="a", pattern="x")])
    b = compute_etag(1, [_pol(id="a", pattern="y")])
    assert a != b


def test_tolerant_coercion_of_garbage_enum_values():
    p = ApprovalPolicy.from_dict({"id": "x", "matcher_type": "bogus",
                                  "action": "nope", "severity": "weird"})
    assert p.matcher_type is MatcherType.EXACT
    assert p.action is PolicyAction.REQUIRE_APPROVAL
    assert p.severity is Severity.MEDIUM


# ---- heredoc false positives (TASK-860) ------------------------------------

# The two live Bash rules, verbatim from tool_approval_policies.
_GIT_RULE = r"\bgit\s+(checkout\s+-b|switch\s+-c|(checkout|switch)\s+(?!-)(?!main\b)(?!master\b)\S+)"
_DOCKER_RULE = (
    r"(\bdocker(-compose|\s+compose)?\b(?!\s+exec\b)[^|&;]*?"
    r"\b(up|down|restart|stop|start|kill|rm|pause|unpause)\b)"
    r"|(\bmake\s+(?:up|down|restart|rebuild|build|run|stop|start|prod-mode|"
    r"dev-mode|snapshot|recreate|docker-dev)\b)"
)


def _live_bash_rules() -> list[ApprovalPolicy]:
    return [
        _pol(id="git", order=35, matcher_type="regex", pattern=_GIT_RULE,
             action="require_approval", severity="high"),
        _pol(id="docker", order=70, matcher_type="regex", pattern=_DOCKER_RULE,
             action="require_approval", severity="high"),
    ]


def _heredoc(sink: str, body: str, tag: str = "'EOF'") -> str:
    return f"{sink} <<{tag}\n{body}\nEOF"


def test_strip_leaves_commands_without_heredocs_untouched():
    cmd = "docker compose restart app && echo done"
    assert strip_inert_heredoc_bodies(cmd) == cmd


def test_strip_drops_a_cat_heredoc_body_but_keeps_the_frame():
    cmd = _heredoc('cat > "/tmp/x.ts"', "const a = 'docker compose restart app';")
    assert strip_inert_heredoc_bodies(cmd) == 'cat > "/tmp/x.ts" <<\'EOF\'\nEOF'


def test_strip_keeps_an_executed_heredoc_body():
    for sink in ("bash", "sh -s", "python3", "ssh nick@host bash", "/bin/bash"):
        cmd = _heredoc(sink, "docker compose down")
        assert "docker compose down" in strip_inert_heredoc_bodies(cmd), sink


def test_strip_preserves_wrapped_unquoted_heredoc():
    cmd = _heredoc("sudo tee -a /etc/hosts", "make rebuild", tag="EOF")
    assert strip_inert_heredoc_bodies(cmd) == cmd


def test_strip_keeps_an_unterminated_body():
    cmd = "cat > /tmp/x <<'EOF'\ndocker compose down"
    assert "docker compose down" in strip_inert_heredoc_bodies(cmd)


def test_strip_ignores_here_strings():
    cmd = "cat <<<'docker compose restart app'"
    assert strip_inert_heredoc_bodies(cmd) == cmd


def test_strip_handles_two_heredocs_on_one_line():
    cmd = (
        "cat > /tmp/a <<'A'\ndocker compose down\nA\n"
        "cat > /tmp/b <<'B'\nmake rebuild\nB"
    )
    # Multiple commands are not proven standalone data sinks.
    assert strip_inert_heredoc_bodies(cmd) == cmd


def test_writing_a_fixture_that_mentions_docker_no_longer_gates():
    call = {"command": _heredoc(
        'cat > "/home/bridge/dev/bawthub/x.test.ts"',
        'const cmd = "docker compose restart app";',
    )}
    decision = evaluate(_live_bash_rules(), "claude-code", "Bash", call)
    assert decision.action is PolicyAction.ALLOW
    assert decision.policy is None


def test_writing_a_fixture_that_mentions_git_switch_no_longer_gates():
    call = {"command": _heredoc("cat > /tmp/notes.md", "we ran git checkout feature/x")}
    assert evaluate(_live_bash_rules(), "claude-code", "Bash", call).action is PolicyAction.ALLOW


def test_a_real_docker_restart_still_gates():
    call = {"command": "ssh nick@172.18.0.1 'cd ~/dev/llm-bawt && docker compose restart app'"}
    decision = evaluate(_live_bash_rules(), "claude-code", "Bash", call)
    assert decision.action is PolicyAction.REQUIRE_APPROVAL
    assert decision.policy is not None and decision.policy.id == "docker"


def test_a_heredoc_piped_into_a_shell_still_gates():
    call = {"command": _heredoc("bash", "docker compose down")}
    decision = evaluate(_live_bash_rules(), "claude-code", "Bash", call)
    assert decision.action is PolicyAction.REQUIRE_APPROVAL


def test_recorded_subject_and_grant_key_stay_raw():
    body = 'const cmd = "docker compose restart app";'
    command = _heredoc("cat > /tmp/x.ts", body)
    decision = evaluate([_pol(id="all", matcher_type="always")], "claude-code", "Bash",
                        {"command": command})
    # The gate matched on the trimmed text, but the audit trail keeps the truth.
    assert decision.subject == command
    assert body in decision.subject
    assert decision.grant_key == grant_key("claude-code", "Bash", command)


def test_matchable_subject_is_a_noop_for_non_shell_tools():
    blob = '{"content": "cat <<\'EOF\'\\ndocker compose down\\nEOF"}'
    assert matchable_subject("Write", blob) == blob
    assert matchable_subject("Bash", "") == ""


def test_exact_identity_preserves_all_input_and_context():
    base = {"command": "printf 'a  b'", "timeout": 100, "nested": {"a": [1, 2]}}
    key = invocation_key("claude-code", "Bash", base, cwd="/repo")
    assert key == invocation_key("claude-code", "Bash", dict(reversed(list(base.items()))), cwd="/repo")
    for changed in (
        {**base, "command": "printf 'a b'"},
        {**base, "command": base["command"] + "\n"},
        {**base, "timeout": 101},
        {**base, "nested": {"a": [2, 1]}},
        {**base, "extra": None},
    ):
        assert key != invocation_key("claude-code", "Bash", changed, cwd="/repo")
    for backend, tool, cwd in (("codex", "Bash", "/repo"),
                               ("claude-code", "mcp__one__Bash", "/repo"),
                               ("claude-code", "Bash", "/other"),
                               ("claude-code", "Bash", None)):
        assert key != invocation_key(backend, tool, base, cwd=cwd)
    assert invocation_key("mcp", "mcp__one__Write", base) != invocation_key("mcp", "mcp__two__Write", base)


def test_exact_identity_does_not_use_policy_field_or_trimmed_heredoc():
    policies = [_pol(tool_name="Write", field="file_path")]
    original = {"file_path": "x", "content": "old"}
    changed = {**original, "content": "new"}
    first = evaluate(policies, "claude-code", "Write", original, exact_invocation=True, cwd="/repo")
    second = evaluate(policies, "claude-code", "Write", changed, exact_invocation=True, cwd="/repo")
    assert first.subject == second.subject == "x"
    assert first.grant_key != second.grant_key
    for tool, before, after in (
        ("Bash", {"command": _heredoc("cat > /tmp/x", "first")},
         {"command": _heredoc("cat > /tmp/x", "second")}),
        ("ops_run", {"operation": "x", "args": {}, "idempotency_key": "one"},
         {"operation": "x", "args": {}, "idempotency_key": "two"}),
    ):
        assert invocation_key("claude-code", tool, before) != invocation_key("claude-code", tool, after)


def test_default_backend_keys_match_subject_contract_on_all_decisions():
    for policies, tool, arguments in (
        ([], "Bash", {"command": "ls"}),
        ([], "ops_run", {"operation": "x"}),
        ([_pol()], "Bash", {"command": "x  y"}),
        ([_pol(action="deny")], "Bash", {"command": "x"}),
        ([_pol(tool_name="Write", field="file_path")], "Write", {"file_path": "x", "content": "y"}),
    ):
        decision = evaluate(policies, "claude-code", tool, arguments)
        assert decision.grant_key == grant_key("claude-code", tool, decision.subject)
        exact = evaluate(policies, "claude-code", tool, arguments, exact_invocation=True)
        assert exact.grant_key == invocation_key("claude-code", tool, arguments)
        assert exact.grant_key != decision.grant_key


def test_quoted_standalone_data_heredoc_allowlist():
    for header in ("cat > /tmp/x <<'EOF'", 'cat <<"EOF" >> "/tmp/a b"',
                   "tee -a /tmp/x <<'EOF'", "/usr/bin/cat > /tmp/x <<-'EOF'"):
        command = header + "\ndocker compose down\nEOF\n"
        assert strip_inert_heredoc_bodies(command) == header + "\nEOF\n"
        assert evaluate(_live_bash_rules(), "claude-code", "Bash", {"command": command}).is_allowed
    command = "cat > /tmp/x <<-'EOF'\n\tdocker compose down\n\tEOF"
    assert strip_inert_heredoc_bodies(command) == "cat > /tmp/x <<-'EOF'\n\tEOF"


def test_ambiguous_or_executable_heredocs_are_never_trimmed():
    commands = [
        "cat > /tmp/x <<EOF\n$(docker compose down)\nEOF",
        "cat > /tmp/x <<EOF\n`docker compose down`\nEOF",
        "cat <<'EOF' | bash\ndocker compose down\nEOF",
        "tee /tmp/x <<'EOF' | sh\ndocker compose down\nEOF",
        "cat > /tmp/x <<'EOF'\ndocker compose down\nEOF\nbash /tmp/x",
        "cat > /tmp/x <<'EOF' # comment\ndocker compose down\nEOF",
        "# cat > /tmp/x <<'EOF'\ndocker compose down\nEOF",
        "cat > /tmp/x <<'EOF'\ndocker compose down\n EOF",
        "cat > /tmp/x <<'EOF'\ndocker compose down\nEOF ",
        "cat > /tmp/x <<'EOF'\ndocker compose down\n\tEOF",
        "cat > /tmp/x <<'EOF' <<'OTHER'\ndocker compose down\nEOF\nOTHER",
        "cat > /tmp/x <<'EOF'\nEOF\ndocker compose down\nEOF",
        "cat > >(bash) <<'EOF'\ndocker compose down\nEOF",
        "env cat > /tmp/x <<'EOF'\ndocker compose down\nEOF",
        "cat > /tmp/x <<'EOF' && bash /tmp/x\ndocker compose down\nEOF",
    ]
    for command in commands:
        assert strip_inert_heredoc_bodies(command) == command, command
        assert evaluate(_live_bash_rules(), "claude-code", "Bash", {"command": command}).requires_approval, command


def _starter_policy(category: str) -> ApprovalPolicy:
    from llm_bawt.approval_defaults import _DEFAULT_POLICIES

    row = next(row for row in _DEFAULT_POLICIES if row["category"] == category)
    return ApprovalPolicy.from_dict({"id": f"starter-{category}", **row})


def test_starter_rm_gates_recursive_or_force_options():
    policy = _starter_policy("filesystem")
    for command in (
        "rm -rf /x", "rm -fr /x", "rm -R /x", "rm -f /x",
        "rm --recursive --force /x", "rm --recursive /x", "rm --force /x",
        "rm /x -rf", "rm /x /y --force", "rm /x --recursive --force",
        "rm -v /x -R", "rm -rf -- /x", "rm '/x y' --force",
        "rm /x '-rf'", 'rm "--recursive" /x',
        "cd /x && rm /y -rf", "true;rm /x --force", "pwd\nrm /x -r",
        "/bin/rm /x -rf", "/usr/bin/rm --recursive /x",
        "ssh host rm /x -rf", "ssh -p 22 host rm --recursive /x",
        "ssh host 'cd /x && rm /y -rf'", 'ssh host "rm /x --force"',
        "bash -c 'rm /x -rf'", "sudo -u root rm /x -rf",
        "env MODE=test command rm /x -rf",
    ):
        decision = evaluate([policy], "claude-code", "Bash", {"command": command})
        assert decision.requires_approval, command
        assert decision.subject == command
        assert decision.grant_key == grant_key("claude-code", "Bash", command)


def test_starter_rm_only_gates_option_tokens_before_double_dash():
    policy = _starter_policy("filesystem")
    for command in (
        "rm /x", "rm -v /x", "rm -- /x -rf", "rm /x -- --recursive --force",
        "rm '--' -rf", 'rm "--" --force', "rm /tmp/-rf",
        "rm /tmp/--recursive", "rm /x-rf", "rm --forceful /x",
        "rm -rf.txt", "rm 'a -rf'", 'rm "a --force"',
        "rm /x; echo -rf", "rm /x && printf '%s' --force",
        "echo rm -rf /x", "echo 'cd /x && rm /y -rf'",
        'printf "%s" "rm --recursive --force /x"',
        "grep 'rm /x -rf' notes", "env grep 'rm --force /x' notes",
        "printf '%s' /tmp/rm -rf", "/tmp/not-rm -rf /x",
        "ssh host 'echo rm -rf /x'", r'ssh host "echo \"rm -rf /x\""',
        "bash -c 'echo rm -rf /x'",
    ):
        assert evaluate([policy], "claude-code", "Bash", {"command": command}).is_allowed, command


def test_starter_sudo_gates_command_positions_and_wrappers():
    policy = _starter_policy("privilege")
    for command in (
        "sudo id", "  sudo\tid", "sudo", "cd /x && sudo id",
        "false || sudo id", "pwd;sudo id", "pwd\nsudo id", "true | sudo id",
        "/usr/bin/sudo id", "/bin/sudo id", "ssh host sudo id",
        "ssh -p 22 -o BatchMode=yes host sudo id",
        "ssh -p22 host sudo id", "ssh user@host-1 sudo id", "ssh host 'sudo id'",
        'ssh host "cd /x && sudo id"', "cd /x && ssh host 'sudo id'",
        "bash -c 'sudo id'", "sh -lc 'cd /x && sudo id'",
        "ssh host 'echo \"literal; sudo\" && sudo id'",
        'ssh host "echo \'literal; sudo\' && sudo id"',
        "env MODE=test sudo id", "command sudo id", "exec sudo id",
        "echo 'sudo is literal'; sudo id", "grep 'sudo' notes && sudo id",
    ):
        assert evaluate([policy], "claude-code", "Bash", {"command": command}).requires_approval, command


def test_starter_sudo_does_not_gate_literal_data_or_substrings():
    policy = _starter_policy("privilege")
    for command in (
        "echo sudo id", "echo 'cd /x && sudo id'", 'echo "sudo id; sudo id"',
        "printf '%s' sudo", "grep sudo notes", "rg 'sudo id' notes",
        "env grep 'sudo id' notes", "cat /tmp/sudo", "/tmp/not-sudo id",
        "sudoedit /x", "echo /usr/bin/sudo", "ssh sudo hostname",
        "ssh -o sudo host hostname", "ssh host echo sudo id",
        "ssh host 'echo sudo id'", 'ssh host "echo \'sudo id; sudo id\'"',
        "bash -c 'echo sudo id'", "echo 'x && sudo id'; true",
        "printf '%s' 'a\nsudo id'", r"echo x\;sudo id",
    ):
        assert evaluate([policy], "claude-code", "Bash", {"command": command}).is_allowed, command


if __name__ == "__main__":
    # Standalone runner for environments without pytest.
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed ({len(fns)} total)")
    sys.exit(1 if failed else 0)
