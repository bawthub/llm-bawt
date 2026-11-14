"""Hermetic command selection and counts-only failure receipts."""
import json

import pytest

from scripts.memory_ownership import main


class NoConnection:
    def begin(self):
        raise AssertionError("command touched the database")


@pytest.mark.parametrize("command", ["expand", "index", "archive-down"])
def test_schema_commands_without_apply_do_not_connect(command, capsys):
    assert main([command], engine=NoConnection()) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not_applied"


@pytest.mark.parametrize("bot", ["*", "", " one ", "bot*"])
def test_wildcard_or_ambiguous_identity_refuses_before_connect(bot):
    with pytest.raises(SystemExit) as exc:
        main(["activate", "--bot", bot, "--actor", "synthetic-operator"], engine=NoConnection())
    assert exc.value.code == 2


def test_mark_only_and_attestation_refuse_before_connect():
    with pytest.raises(SystemExit) as exc:
        main(["activate", "--bot", "synthetic", "--actor", "synthetic-operator",
              "--mark-only", "--attest-user", "synthetic-user"], engine=NoConnection())
    assert exc.value.code == 2


def test_failure_does_not_print_driver_details(capsys):
    class Broken:
        def begin(self):
            raise RuntimeError("sensitive database detail")

    assert main(["status"], engine=Broken()) == 1
    output = capsys.readouterr().out
    assert "sensitive" not in output
    assert json.loads(output)["status"] == "failed"
