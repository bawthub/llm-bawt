"""Command facts for opt-in ``tool_args`` recall (TASK-1557).

Recorded commands never cross the recall boundary as text. Only recognized
``git clone`` facts with validated operands do. Every negative below asserts
that no fact is produced and no operand byte appears in the admitted text.
All fixtures are synthetic.
"""
from __future__ import annotations

import json

import pytest

from llm_bawt.recall.command_facts import command_facts, safe_path, safe_repo
from llm_bawt.recall.contracts import RecallScope
from llm_bawt.recall.postgres import PUBLIC_HISTORY_AUTHORITY, ToolArgSource, tool_arg_facts

SCOPE = RecallScope("caller", "operator", "session-a", frozenset({"excluded"}),
                    frozenset({"caller", "peer", "excluded"}))
SECRET = "fixture-pass-7"


def rendered(command: str) -> list[str]:
    return [fact.render() for fact in command_facts(command)]


@pytest.mark.parametrize("command", [
    f"curl -u demo:{SECRET} https://api.example.com/v1",
    f"curl -H 'X-Service-Access: {SECRET}' https://api.example.com/v1",
    f"python -c 'client.login(\"demo\", \"{SECRET}\")'",
    f"cat <<EOF\ngit clone https://github.com/example/r {SECRET}\nEOF",
    f"git clone https://github.com/example/r $(bw get password {SECRET})",
    f"git clone https://github.com/example/r `echo {SECRET}`",
    f"git clone https://demo:{SECRET}@github.com/example/r d",
    f"git clone https://github.com/example/r?token={SECRET} d",
    f"git clone https://github.com/example/r#{SECRET} d",
    "git clone https://github.com/example/r .env",
    "git clone https://github.com/example/r ~/.ssh/fixture",
    f"git clone --config http.extraHeader={SECRET} https://github.com/example/r",
    f"git clone https://github.com/example/r d | tee {SECRET}",
    f"git clone https://github.com/example/r d > {SECRET}",
    f"git clone https://github.com/example/r d 2>&1 {SECRET}",
    f"git clone https://github.com/example/r '{SECRET}",
    f"git clone https://github.com/example/r \\\n  {SECRET}",
    f"(git clone https://github.com/example/r {SECRET})",
    f"echo '&&' git clone https://github.com/example/r {SECRET}",
    f"git clone https://github.com/example/../r {SECRET}",
    f"git clone http://github.com/example/r {SECRET}",
    f"git clone ssh://git@github.com/example/r {SECRET}",
    f"false || git clone https://github.com/example/r {SECRET}",
    f"git clone https://github.com/example/r {SECRET} extra",
    # A comment: the shell runs nothing after '#' on the line (review B1).
    f"# note; git clone https://github.com/example/r {SECRET}",
    f"#git clone https://github.com/example/r {SECRET}",
    f"echo x # && git clone https://github.com/example/r {SECRET}",
    f"git clone https://github.com/example/r {SECRET}#x",
    # A quote joined to other characters is ONE shell word (review B1).
    f"git clone 'https://github.com/example/r'{SECRET}",
    f"git clone https://github.com/example/r'{SECRET}'",
    f"git clone \"https://github.com/example/r\"'{SECRET}'",
    f"git clone https://github.com/example/r \"\"{SECRET}",
    # A destination that resolves onto a credential path (review B2).
    "cd /tmp/.docker && git clone https://github.com/example/r config.json",
    "cd /home/u && git clone https://github.com/example/r .ssh",
    "cd / && git clone https://github.com/example/r proc/1",
    # The shell is now in a denied directory: a relative destination is under it.
    "cd /srv/.aws && git clone https://github.com/example/r fixture",
    "cd /srv && cd .ssh && git clone https://github.com/example/r fixture",
    "cd /srv/.ssh && cd keys && git clone https://github.com/example/r fixture",
    "pushd /srv/.ssh && git clone https://github.com/example/r fixture",
])
def test_unrecognized_or_unsafe_commands_project_nothing(command):
    facts = rendered(command)
    assert facts == []
    assert tool_arg_facts(json.dumps({"command": command})) is None


def test_clone_after_cd_is_one_attempted_fact_with_an_absolute_destination():
    command = "cd /tmp && git clone -q https://github.com/example/r atc"
    assert rendered(command) == ["attempted clone repo=https://github.com/example/r dest=/tmp/atc"]


def test_segments_before_the_first_unsafe_token_are_kept_and_nothing_after():
    command = (f"cd /tmp && rm -rf atc && git clone -q https://github.com/example/r atc && cd atc"
               f" && MB=$(git merge-base HEAD up/master) && git clone https://github.com/example/s {SECRET}")
    assert rendered(command) == ["attempted clone repo=https://github.com/example/r dest=/tmp/atc"]


@pytest.mark.parametrize("unsafe", ["$DEST", "${DEST}", "`pwd`", "a\\ b"])
def test_an_expansion_or_escape_in_any_word_stops_the_parse(unsafe):
    # The unsafe word is in a segment the parser would otherwise skip
    # (unrecognized command); the clone after it must still not count.
    command = (f"git clone https://github.com/example/r a && echo {unsafe}"
               f" && git clone https://github.com/example/s b")
    assert rendered(command) == ["attempted clone repo=https://github.com/example/r dest=a"]


@pytest.mark.parametrize("command, expected", [
    ("git clone --depth 1 -b main --single-branch git@github.com:example/r.git /srv/r",
     "attempted clone repo=git@github.com:example/r.git dest=/srv/r"),
    ("git clone https://github.com/example/r", "attempted clone repo=https://github.com/example/r"),
    ("cd /a\ncd b; git clone \"https://github.com/example/r\" c",
     "attempted clone repo=https://github.com/example/r dest=/a/b/c"),
    # Unknown starting directory: the destination stays as recorded.
    ("cd sub && git clone https://github.com/example/r c", "attempted clone repo=https://github.com/example/r dest=c"),
    # A directory change the parser cannot follow unsets the directory.
    ("cd /tmp && pushd /srv && git clone https://github.com/example/r c",
     "attempted clone repo=https://github.com/example/r dest=c"),
])
def test_recognized_clone_shapes(command, expected):
    assert rendered(command) == [expected]


@pytest.mark.parametrize("path, ok", [
    ("/tmp/atc", True), ("~/dev/r", True), ("a/../b", False), ("/srv/.env.local", False),
    ("/home/u/.aws/config", False), ("/x/id_ed25519", False), ("/x/server.key", False),
    ("/x/credentials.json", False), ("/proc/1/environ", False), ("a b", False), ("$HOME/r", False),
])
def test_path_operand_rules(path, ok):
    assert safe_path(path) is ok


@pytest.mark.parametrize("url, ok", [
    ("https://github.com/example/r.git", True), ("git@github.com:example/r", True),
    ("https://user@github.com/example/r", False), ("https://github.com:8443/example/r", False),
    ("https://github.com/example/r?x=1", False), ("https://github.com/example/%2e%2e/r", False),
    ("https://github.com/example/../r", False), ("file:///srv/r", False),
])
def test_repo_operand_rules(url, ok):
    assert safe_repo(url) is ok


def test_segments_before_a_comment_are_kept_and_nothing_after():
    command = f"cd /tmp && git clone https://github.com/example/r a # && git clone https://github.com/example/s {SECRET}"
    assert rendered(command) == []  # the interrupted segment holds the first clone
    command = f"git clone https://github.com/example/r a; # git clone https://github.com/example/s {SECRET}"
    assert rendered(command) == ["attempted clone repo=https://github.com/example/r dest=a"]


@pytest.mark.parametrize("command", [
    f"git clone https://github.com/example/r {SECRET}",
    f"cd /srv && git clone https://github.com/example/r {SECRET}",
    f"cd /tmp/.docker && git clone https://github.com/example/r config.json && git clone https://github.com/example/r {SECRET}",
    # An absolute destination, or a safe absolute cd, leaves the denied directory.
    "cd /srv/.ssh && git clone https://github.com/example/r /srv/r",
    "cd /srv/.ssh && cd /srv && git clone https://github.com/example/r fixture",
])
def test_every_emitted_destination_passes_the_path_rules(command):
    facts = command_facts(command)
    assert facts and all(fact.dest is None or safe_path(fact.dest) for fact in facts)
    assert all(".ssh" not in (fact.dest or "") and "config.json" not in (fact.dest or "") for fact in facts)


def test_tool_arg_facts_rejects_non_command_payloads():
    assert tool_arg_facts("{not json") is None
    assert tool_arg_facts(json.dumps(["git clone https://github.com/example/r"])) is None
    assert tool_arg_facts(json.dumps({"command": 7})) is None


class Database:
    def __init__(self, rows):
        self.rows = rows
        self.reads = []

    def read(self, sql, params):
        self.reads.append((sql, params))
        return self.rows


def call(row_id, command, *, bot="peer", at=None):
    return {"id": row_id, "arguments": json.dumps({"command": command}), "at": at,
            "source_bot": bot, "session_id": "s-1"}


@pytest.mark.asyncio
async def test_source_selects_on_admitted_facts_only_and_admits_no_command_text():
    rows = [
        # The query term appears only in a dropped segment of this call.
        call("1", f"git clone https://github.com/example/other x && echo widget {SECRET}"),
        call("2", f"cd /tmp && git clone https://github.com/example/widget w && curl -u a:{SECRET} x"),
        call("3", f"curl -u demo:{SECRET} https://widget.example.com"),
    ]
    database = Database(rows)
    evidence = await ToolArgSource(database).search("where was the widget cloned", SCOPE, ())
    assert [item.id for item in evidence] == ["2"]
    assert evidence[0].text == "attempted clone repo=https://github.com/example/widget dest=/tmp/w"
    assert (evidence[0].source, evidence[0].source_bot, evidence[0].session_id) == ("tool_args", "peer", "s-1")
    assert all(SECRET not in item.text and "curl" not in item.text for item in evidence)
    [(sql, params)] = database.reads
    assert params["bots"] == ["caller", "peer"] and "excluded" not in params["delivery_authors"]


@pytest.mark.asyncio
@pytest.mark.parametrize("command, banned", [
    (f"# note; git clone https://github.com/example/r {SECRET}", SECRET),
    (f"git clone 'https://github.com/example/r'{SECRET}", SECRET),
    ("cd /tmp/.docker && git clone https://github.com/example/r config.json", "config.json"),
])
async def test_rejected_operands_never_reach_source_evidence(command, banned):
    evidence = await ToolArgSource(Database([call("1", command)])).search(f"{banned} example", SCOPE, ())
    assert evidence == []


@pytest.mark.asyncio
async def test_source_ranks_by_fact_coverage_and_caps_records():
    rows = [call(str(i), f"git clone https://github.com/example/widget-{i} w{i}") for i in range(9)]
    rows.append(call("best", "git clone https://github.com/example/widget-gadget best"))
    evidence = await ToolArgSource(Database(rows)).search("widget gadget", SCOPE, ())
    assert evidence[0].id == "best" and len(evidence) == 6
    # A label-only match ("clone") still selects, below any operand match.
    labelled = await ToolArgSource(Database(rows)).search("which peer clone gadget", SCOPE, ())
    assert labelled[0].id == "best" and len(labelled) == 6


def test_statement_reads_only_bash_arguments_under_public_history_authority():
    statement = ToolArgSource.statement
    assert PUBLIC_HISTORY_AUTHORITY in statement
    assert "result_text" not in statement
    assert "t.tool_name = 'Bash'" in statement and "t.user_id = :user" in statement
    assert "turns.user_id = c.user_id" in statement
