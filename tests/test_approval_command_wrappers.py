"""TASK-2280 review: preserve execution contexts the old rm rule caught."""
import re

import pytest

from llm_bawt.approval_defaults import _DEFAULT_POLICIES


WRAPPERS = [
    "find . | xargs {command}",
    "find . | xargs -0 -r -n 2 {command}",
    "xargs -I{{}} -P4 {command}",
    "xargs --max-args=2 --delimiter '\\n' -- {command}",
    "find . -exec {command} {{}} +",
    "find . -name '*.o' -execdir {command} {{}} \\;",
    "({command})",
    "{{ {command}; }}",
    "for f in a; do {command}; done",
    "if true; then {command}; fi",
    "if false; then echo no; else {command}; fi",
    "if false; then echo no; elif {command}; then echo yes; fi",
    "timeout 5 {command}",
    "timeout --signal TERM --kill-after=1s 5.5m {command}",
    "nice {command}",
    "nice -n 5 {command}",
    "nice --adjustment=-5 -- {command}",
    "nohup {command}",
    "time -p {command}",
    "stdbuf -oL -e 0 {command}",
    "stdbuf --output=L -- {command}",
    "FOO=1 BAR='two words' {command}",
    "FOO= {command}",
    "cd /tmp && FOO=1 timeout 5 nice -n 2 {command}",
    "ssh host 'FOO=1 timeout 5 {command}'",
    "bash -c 'find . | xargs {command}'",
]


@pytest.mark.parametrize("category,command", [("filesystem", "rm -rf x"), ("privilege", "sudo id")])
@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_execution_wrappers_are_command_positions(category, command, wrapper):
    rule = next(p for p in _DEFAULT_POLICIES if p["category"] == category)
    assert re.search(rule["pattern"], wrapper.format(command=command))


@pytest.mark.parametrize("command", [
    "echo 'find . | xargs rm -rf x'",
    "grep 'timeout 5 sudo id' fixture",
    "find . -name 'rm -rf' -print",
    "find . -name '-exec rm -rf' -print",
    "FOO='rm -rf x' echo safe",
    "xargs echo rm -rf x",
    "nice grep 'sudo id' fixture",
    "timeout 5 echo 'rm -rf x'",
    "stdbuf -oL printf '%s' 'sudo id'",
    "find . -exec echo 'rm -rf x' {} +",
    "{ echo 'rm -rf x'; }",
    "for f in a; do echo 'sudo id'; done",
    "xargs rm -- -rf",
    "FOO=1 rm ./prefix-rf",
])
def test_wrapped_arguments_are_still_data(command):
    for rule in _DEFAULT_POLICIES[:2]:
        assert re.search(rule["pattern"], command) is None
