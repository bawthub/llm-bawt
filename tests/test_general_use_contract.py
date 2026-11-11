"""The general-use ratchet needs no live services or private configuration."""
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import general_use_contract as guard


def policy(**changes):
    return {"version": 1, "mode": "private", "hmac_key_file": "test.key",
            "_hmac_key": b"a" * 32, **{key: [] for key in guard.PRIVATE_RULES}, **changes}


def serialized(value):
    return json.dumps({key: item for key, item in value.items() if key != "_hmac_key"})


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "-C", str(self.repo), "init", "-q"], check=True)
        self.private = self.root / "private.json"
        self.private.write_text(serialized(policy()))
        (self.root / "test.key").write_text((b"a" * 32).hex())

    def tracked(self, path, text):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        subprocess.run(["git", "-C", str(self.repo), "add", "--", path], check=True)
        return target

    def hits(self, text, **changes):
        self.tracked("src/example.txt", text)
        return guard.scan(self.repo, policy(**changes))[0]

    def baseline(self, hits):
        data = guard.inventory(hits)
        for entry in data["rules"].values():
            entry.update(reason="Existing fixture; replace with configuration", removal_task="TASK-123")
        path = self.repo / "general-use-baseline.json"
        path.write_text(json.dumps(data))
        return guard.read_baseline(path)

    def test_private_literals_and_boundaries(self):
        hits = self.hits(
            "EXAMPLE PERSON: a@personal.invalid sample-bot deploy-container\n"
            "sample-bot-extra notpersonal.invalidsuffix\n",
            operator_identities=["Example Person"], personal_domains=["personal.invalid"],
            bot_slugs=["sample-bot"], container_names=["deploy-container"],
        )
        self.assertEqual(Counter(hit.rule for hit in hits), Counter({
            "operator_identities": 1, "personal_domains": 1,
            "bot_slugs": 1, "container_names": 1,
        }))
        self.assertEqual(sum(hits.values()), 4)

    def test_exact_rfc1918_ranges(self):
        hits = self.hits(
            "10.0.0.1 172.16.0.1 172.31.255.255 192.168.1.1\n"
            "127.0.0.1 172.15.0.1 172.32.0.1 192.0.2.1 10.999.0.1\n"
        )
        self.assertEqual(sum(hits.values()), 4)
        self.assertEqual({hit.rule for hit in hits}, {"rfc1918"})

    def test_paths_and_literal_container_but_not_templates(self):
        hits = self.hits(
            "/home/example/project /mnt/user/appdata/project\n"
            "container_name: deployment-app # retained fixture\n"
            "container_name: ${APP_CONTAINER}\n/home/${APP_USER}/project\n/home/<user>/project\n"
        )
        self.assertEqual(Counter(hit.rule for hit in hits), Counter({
            "host_path": 2, "container_literal": 1,
        }))

    def test_shipped_image_home_and_compose_names_are_product_layout(self):
        hits = self.hits(
            "/home/bridge/dev /home/bridge /home/bridged/x /home/other/dev\n"
            "container_name: llm-bawt-app\ncontainer_name: \"bawthub-frontend\"\n"
            "container_name: bawthub-frontend-prod-1\n",
        )
        self.assertEqual(Counter(hit.rule for hit in hits), Counter({
            "host_path": 2, "container_literal": 1,
        }))

    def test_scope_and_untracked_inputs(self):
        for path in ("src/app.py", "docs/run.md", "tenant-skills/tool/SKILL.md",
                     "skills/tool.md", "prompts/system.txt", "seeds/default.txt",
                     "docker-compose.local.yml", "deploy/compose.yaml",
                     "tests/test_ops_routing_reconcile.py"):
            self.tracked(path, "10.1.2.3\n")
        self.tracked("tests/unrelated.py", "10.1.2.3\n")
        (self.repo / "src/untracked.py").write_text("10.1.2.3\n")
        hits, _, scanned = guard.scan(self.repo, policy())
        self.assertEqual(scanned, 9)
        self.assertEqual(sum(hits.values()), 9)

    def test_count_growth_new_file_and_reduction(self):
        hits = self.hits("10.1.2.3\n")
        baseline = self.baseline(hits)
        added, stale, _ = guard.check(hits, baseline, {"builtin"})
        self.assertFalse(added or stale)
        more = self.hits("10.1.2.3\n10.1.2.3\n")
        self.assertEqual(sum(guard.check(more, baseline, {"builtin"})[0].values()), 1)
        self.tracked("docs/new.md", "10.1.2.3\n")
        more = guard.scan(self.repo, policy())[0]
        self.assertEqual(sum(guard.check(more, baseline, {"builtin"})[0].values()), 2)
        self.assertEqual(sum(guard.check(Counter(), baseline, {"builtin"})[1].values()), 1)

    def test_private_baseline_only_applies_to_active_literals(self):
        original = policy(bot_slugs=["sample-bot"])
        self.tracked("docs/note.md", "sample-bot\n")
        hits = guard.scan(self.repo, original)[0]
        baseline = self.baseline(hits)
        generic, _, _ = guard.scan(self.repo, policy())
        added, stale, inactive = guard.check(generic, baseline, {"builtin"})
        self.assertFalse(added or stale)
        self.assertEqual(inactive, 1)
        other = policy(bot_slugs=["another-bot"])
        self.tracked("docs/note.md", "another-bot\n")
        changed = guard.scan(self.repo, other)[0]
        self.assertEqual(sum(guard.check(changed, baseline, guard.policies(other)[1])[0].values()), 1)

    def test_baseline_requires_metadata_and_rejects_duplicates(self):
        hits = self.hits("10.1.2.3\n")
        path = self.repo / "baseline.json"
        data = guard.inventory(hits)
        path.write_text(json.dumps(data))
        with self.assertRaises(guard.ContractError):
            guard.read_baseline(path)
        self.baseline(hits)
        data["rules"]["rfc1918"].update(reason="Documented reason", removal_task="TASK-123")
        for key, value in (("removal_task", "later"), ("reason", " ")):
            invalid = json.loads(json.dumps(data))
            invalid["rules"]["rfc1918"][key] = value
            path.write_text(json.dumps(invalid))
            with self.assertRaises(guard.ContractError):
                guard.read_baseline(path)
        fingerprint = next(iter(data["files"]["src/example.txt"]["rfc1918"]))
        invalid = json.loads(json.dumps(data))
        invalid["files"]["src/example.txt"]["rfc1918"][fingerprint] = True
        path.write_text(json.dumps(invalid))
        with self.assertRaises(guard.ContractError):
            guard.read_baseline(path)
        path.write_text('{"version":1,"version":1,"rules":{},"files":{}}')
        with self.assertRaises(guard.ContractError):
            guard.read_baseline(path)

    def test_policy_missing_invalid_and_explicit_empty(self):
        with self.assertRaises(guard.ContractError):
            guard.load_policy(self.repo, environ={})
        with self.assertRaises(guard.ContractError):
            guard.load_policy(self.repo, environ={"GENERAL_USE_DENYLIST_JSON": "bad"})
        self.assertEqual(guard.load_policy(self.repo, environ={
            "GENERAL_USE_DENYLIST_JSON": serialized(policy(
                mode="public", hmac_key_file="", _hmac_key=b"",
            )),
        }), policy(mode="public", hmac_key_file="", _hmac_key=b""))
        for change in ({"bot_slugs": [""]}, {"bot_slugs": "sample-bot"},
                       {"unknown": []}, {"mode": []}, {"version": True}):
            with self.assertRaises(guard.ContractError):
                guard.load_policy(self.repo, environ={
                    "GENERAL_USE_DENYLIST_JSON": serialized(policy(**change)),
                })

    def test_policy_cannot_be_tracked(self):
        path = self.tracked("config/private.json", serialized(policy()))
        with self.assertRaisesRegex(guard.ContractError, "untracked"):
            guard.load_policy(self.repo, str(path), {})
        self.assertEqual(guard.load_policy(self.repo, str(self.private), {}), policy())

    def test_common_git_dir_policy(self):
        path = self.repo / ".git/general-use-denylist.json"
        path.write_text(serialized(policy(bot_slugs=["sample-bot"])))
        (self.repo / ".git/test.key").write_text((b"a" * 32).hex())
        self.assertEqual(guard.load_policy(self.repo, environ={})["bot_slugs"], ["sample-bot"])

    def test_symlink_target_scanned_without_reading_external_file(self):
        outside = self.root / "secret.txt"
        outside.write_text("10.1.2.3")
        (self.repo / "src").mkdir()
        link = self.repo / "src/link"
        link.symlink_to(outside)
        subprocess.run(["git", "-C", str(self.repo), "add", "src/link"], check=True)
        self.assertFalse(guard.scan(self.repo, policy())[0])

    def test_cli_reports_no_literal_payload_and_nonzero_on_new_hit(self):
        self.private.write_text(serialized(policy(bot_slugs=["sample-bot"])))
        self.tracked("docs/note.md", "sample-bot\n")
        self.baseline(Counter())
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(out), redirect_stderr(err):
            code = guard.main(["--repo", str(self.repo), "--denylist-file", str(self.private)])
        self.assertEqual(code, 1)
        self.assertIn("NEW docs/note.md:1 bot_slugs", out.getvalue())
        self.assertNotIn("sample-bot", out.getvalue() + err.getvalue())
        baseline = guard.inventory(guard.scan(self.repo, policy(bot_slugs=["sample-bot"]))[0])
        self.assertNotIn("sample-bot", json.dumps(baseline))

    def test_private_fingerprints_are_keyed_not_dictionary_hashes(self):
        self.tracked("docs/note.md", "sample-bot\n")
        first = guard.scan(self.repo, policy(bot_slugs=["sample-bot"]))[0]
        second = guard.scan(self.repo, policy(
            bot_slugs=["sample-bot"], _hmac_key=b"b" * 32,
        ))[0]
        self.assertNotEqual(set(first), set(second))
        known_word_hash = guard.digest("bot_slugs\0sample-bot")
        self.assertNotIn(known_word_hash, json.dumps(guard.inventory(first)))

    def test_inactive_private_entries_fail_closed_except_explicit_public(self):
        self.tracked("docs/note.md", "sample-bot\n")
        self.baseline(guard.scan(self.repo, policy(bot_slugs=["sample-bot"]))[0])
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            code = guard.main(["--repo", str(self.repo), "--denylist-file", str(self.private)])
        self.assertEqual(code, 1)
        self.assertIn("private policy drift", out.getvalue())
        self.private.write_text(serialized(policy(mode="public", hmac_key_file="", _hmac_key=b"")))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(guard.main([
                "--repo", str(self.repo), "--denylist-file", str(self.private),
            ]), 0)

    def test_private_key_required_and_never_tracked(self):
        (self.root / "test.key").unlink()
        with self.assertRaisesRegex(guard.ContractError, "HMAC key"):
            guard.load_policy(self.repo, str(self.private), {})
        key = self.tracked("private.key", (b"a" * 32).hex())
        self.private.write_text(serialized(policy(hmac_key_file=str(key))))
        with self.assertRaisesRegex(guard.ContractError, "untracked"):
            guard.load_policy(self.repo, str(self.private), {})

    def test_explicit_public_policy_rejects_nonempty_private_lists(self):
        self.private.write_text(serialized(policy(
            mode="public", hmac_key_file="", bot_slugs=["sample-bot"],
        )))
        with self.assertRaisesRegex(guard.ContractError, "Public policy"):
            guard.load_policy(self.repo, str(self.private), {})

    def test_gate_runs_guard_in_snapshot_not_author_tree(self):
        root = Path(__file__).resolve().parents[1]
        makefile = (root / "Makefile").read_text()
        snapshot = (root / "scripts/test_snapshot.py").read_text()
        self.assertIn("test-all: contract-guard", makefile)
        self.assertIn('"gate": "test-all"', snapshot)
        self.assertIn("python3 scripts/general_use_contract.py", makefile)
        self.assertIn('"scripts/general_use_contract.py"', snapshot)
        self.assertIn('"general-use-baseline.json"', snapshot)


if __name__ == "__main__":
    unittest.main()
