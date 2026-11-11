"""Offline tracked-text contract guard; private literals never enter reports."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

PRIVATE_RULES = ("operator_identities", "personal_domains", "container_names", "bot_slugs")
ROOTS = ("src/", "docs/", "skills/", "tenant-skills/", "prompts/", "seeds/")
EXTRA_FILES = {"tests/test_ops_routing_reconcile.py"}
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
HOME = re.compile(r"/home/[A-Za-z0-9_.-]+|/mnt/user(?:/[A-Za-z0-9_.-]+)*")
CONTAINER = re.compile(r"^\s*container_name:\s*[\"']?([A-Za-z0-9][A-Za-z0-9_.-]*)[\"']?\s*(?:#.*)?$")
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(net) for net in
                         ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
TASK = re.compile(r"TASK-[1-9]\d*")
HASH = re.compile(r"[0-9a-f]{32}")
RULES = (*PRIVATE_RULES, "rfc1918", "host_path", "container_literal")


class ContractError(ValueError):
    pass


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if result.returncode:
        raise ContractError("Git inventory unavailable")
    return result.stdout.decode("utf-8", errors="strict")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def private_file(path: Path) -> dict:
    # A private input inside any checkout must not already be tracked.
    probe = subprocess.run(
        ["git", "-C", str(path.parent), "ls-files", "--error-unmatch", "--", path.name],
        capture_output=True,
    )
    if probe.returncode == 0:
        raise ContractError("Denylist file must be untracked")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, ValueError):
        raise ContractError("Private denylist missing or invalid JSON") from None


def load_policy(repo: Path, file: str | None = None, environ: dict | None = None) -> dict:
    env = os.environ if environ is None else environ
    common = (repo / git(repo, "rev-parse", "--git-common-dir").strip()).resolve()
    base = common
    if file or env.get("GENERAL_USE_DENYLIST_FILE"):
        path = Path(file or env["GENERAL_USE_DENYLIST_FILE"]).expanduser().resolve()
        data = private_file(path)
        base = path.parent
    elif "GENERAL_USE_DENYLIST_JSON" in env:
        try:
            data = json.loads(env["GENERAL_USE_DENYLIST_JSON"], object_pairs_hook=unique_object)
        except ValueError:
            raise ContractError("Private denylist environment is invalid JSON") from None
    else:
        data = private_file(common / "general-use-denylist.json")
    if (not isinstance(data, dict)
            or set(data) != {"version", "mode", "hmac_key_file", *PRIVATE_RULES}
            or type(data["version"]) is not int or data["version"] != 1
            or not isinstance(data["mode"], str) or data["mode"] not in {"private", "public"}):
        raise ContractError("Denylist requires version 1, mode, hmac_key_file and four literal lists")
    for rule in PRIVATE_RULES:
        values = data[rule]
        if not isinstance(values, list) or any(
            not isinstance(v, str) or not v.strip() or v != v.strip() or "\n" in v or len(v) > 256
            for v in values
        ):
            raise ContractError(f"Invalid literal list: {rule}")
    key_file = data["hmac_key_file"]
    if data["mode"] == "public":
        if any(data[rule] for rule in PRIVATE_RULES) or key_file:
            raise ContractError("Public policy must explicitly have no private literals or key")
        data["_hmac_key"] = b""
    else:
        if not isinstance(key_file, str) or not key_file:
            raise ContractError("Private policy requires hmac_key_file")
        path = (base / Path(key_file).expanduser()).resolve()
        # The key, like the denylist, must never be in tracked source.
        probe = subprocess.run(
            ["git", "-C", str(path.parent), "ls-files", "--error-unmatch", "--", path.name],
            capture_output=True,
        )
        if probe.returncode == 0:
            raise ContractError("HMAC key file must be untracked")
        try:
            encoded = path.read_text().strip()
            if not re.fullmatch(r"[0-9a-fA-F]{64}", encoded):
                raise ValueError()
            data["_hmac_key"] = bytes.fromhex(encoded)
        except (OSError, ValueError):
            raise ContractError("HMAC key file requires a private 32-byte hex key") from None
    return data


def in_scope(path: str) -> bool:
    name = PurePosixPath(path).name
    return (path.startswith(ROOTS) or path in EXTRA_FILES
            or bool(re.fullmatch(r"(?:docker-)?compose(?:[.-].*)?\.ya?ml", name)))


@dataclass(frozen=True, order=True)
class Hit:
    path: str
    rule: str
    fingerprint: str  # per-file literal count; HMAC for every private rule


def policies(policy: dict) -> tuple[list[tuple[str, str, re.Pattern]], set[str]]:
    patterns = []
    active = set()
    for rule in PRIVATE_RULES:
        for value in sorted(set(v.casefold() for v in policy[rule])):
            key = hmac.new(policy["_hmac_key"], (rule + "\0" + value).encode(),
                           hashlib.sha256).hexdigest()[:32]
            active.add(key)
            patterns.append((rule, key, re.compile(
                r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", re.IGNORECASE,
            )))
    return patterns, active


def scan(repo: Path, policy: dict) -> tuple[Counter, dict[Hit, int], int]:
    patterns, _ = policies(policy)
    hits: Counter = Counter()
    locations = {}
    scanned = 0
    for path in sorted(set(git(repo, "ls-files", "--cached", "-z").split("\0")) - {""}):
        if not in_scope(path):
            continue
        full = repo / path
        try:
            raw = os.readlink(full).encode() if full.is_symlink() else full.read_bytes()
            if b"\0" in raw:  # tracked images/assets, not source text
                continue
            content = raw.decode("utf-8")
        except (OSError, UnicodeError):
            raise ContractError(f"Cannot read tracked text: {path}") from None
        scanned += 1
        for number, line in enumerate(content.splitlines(), 1):
            found = []
            for match in IPV4.finditer(line):
                try:
                    address = ipaddress.ip_address(match[0])
                except ValueError:
                    continue
                if any(address in network for network in PRIVATE_NETWORKS):
                    found.append(("rfc1918", digest("rfc1918\0" + match[0])))
            found.extend(("host_path", digest("host_path\0" + match[0])) for match in HOME.finditer(line))
            if match := CONTAINER.fullmatch(line):
                found.append(("container_literal", digest("container_literal\0" + match[1])))
            for rule, key, pattern in patterns:
                found.extend((rule, key) for match in pattern.finditer(line))
            for rule, key in found:
                hit = Hit(path, rule, key)
                hits[hit] += 1
                locations.setdefault(hit, number)
    return hits, locations, scanned


def unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("Duplicate JSON key")
        result[key] = value
    return result


def read_baseline(path: Path) -> dict[Hit, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, ValueError):
        raise ContractError("Baseline missing or invalid JSON") from None
    if (not isinstance(data, dict) or set(data) != {"version", "rules", "files"}
            or type(data["version"]) is not int or data["version"] != 1
            or not isinstance(data["rules"], dict) or not isinstance(data["files"], dict)):
        raise ContractError("Baseline requires version 1, rules and files")
    for rule, metadata in data["rules"].items():
        if (rule not in RULES or not isinstance(metadata, dict)
                or set(metadata) != {"reason", "removal_task"}
                or not isinstance(metadata["reason"], str) or not metadata["reason"].strip()
                or not isinstance(metadata["removal_task"], str)
                or not TASK.fullmatch(metadata["removal_task"])):
            raise ContractError("Every baseline rule needs a reason and removal task")
    entries = {}
    for path, groups in data["files"].items():
        if (not isinstance(path, str) or not in_scope(path) or path.startswith("/")
                or ".." in PurePosixPath(path).parts
                or not isinstance(groups, dict) or not groups):
            raise ContractError("Invalid baseline path or empty rule group")
        for rule, counts in groups.items():
            if rule not in data["rules"] or not isinstance(counts, dict) or not counts:
                raise ContractError("Baseline hit needs a declared rule and literal counts")
            for fingerprint, count in counts.items():
                if not HASH.fullmatch(fingerprint) or type(count) is not int or count < 1:
                    raise ContractError("Invalid baseline literal fingerprint or count")
                entries[Hit(path, rule, fingerprint)] = {**data["rules"][rule], "count": count}
    return entries


def inventory(hits: Counter) -> dict:
    files: dict = {}
    for hit, count in sorted(hits.items()):
        files.setdefault(hit.path, {}).setdefault(hit.rule, {})[hit.fingerprint] = count
    return {"version": 1, "rules": {
        rule: {"reason": "", "removal_task": ""} for rule in sorted({hit.rule for hit in hits})
    }, "files": files}


def check(hits: Counter, baseline: dict, active: set[str]) -> tuple[Counter, Counter, int]:
    applicable = {hit: entry["count"] for hit, entry in baseline.items()
                  if hit.rule not in PRIVATE_RULES or hit.fingerprint in active}
    expected = Counter(applicable)
    return hits - expected, expected - hits, len(baseline) - len(applicable)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--denylist-file")
    parser.add_argument("--inventory", action="store_true",
                        help="print redacted candidates; never approves or rewrites a baseline")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        policy = load_policy(repo, args.denylist_file)
        hits, locations, scanned = scan(repo, policy)
        if args.inventory:
            print(json.dumps(inventory(hits), indent=2))
            return 0
        baseline = read_baseline(args.baseline or repo / "general-use-baseline.json")
        _, active = policies(policy)
        added, stale, inactive = check(hits, baseline, active)
        for hit, count in sorted(added.items()):
            print(f"NEW {hit.path}:{locations[hit]} {hit.rule} +{count} [{hit.fingerprint}]")
        for hit, count in sorted(stale.items()):
            print(f"STALE {hit.path} {hit.rule} -{count} [{hit.fingerprint}]; trim baseline")
        print(f"General-use contract: {scanned} text files, {sum(hits.values())} hits, "
              f"{sum(added.values())} new, {sum(stale.values())} stale, "
              f"{inactive} baseline entries inactive under this private policy")
        if inactive and policy["mode"] == "private":
            print("General-use contract: private policy drift; reconcile inactive baseline entries")
        return 1 if added or stale or (inactive and policy["mode"] == "private") else 0
    except (ContractError, UnicodeError) as exc:
        print(f"General-use contract: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
