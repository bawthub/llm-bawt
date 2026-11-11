"""Offline export contract. Stdlib-only so the source-checkout CLI is testable."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from llm_bawt.prompt_promotion import validate_promotion_item
from llm_bawt.prompt_sync import _fetch_selected, classify, drift_report, export_report, is_following, main
from llm_bawt.prompt_template_text import extract_placeholders, render_template


def item(key="portable.example", body="Use {topic}.", **changes):
    return {
        "key": key, "title": "Portable example", "category": "agents",
        "format": "plain_text", "body": body, "required_vars": ["topic"],
        "scope_type": "global", "scope_id": "*", "sync_class": "product",
        "kind": "new", **changes,
    }


class PromptPromotionContractTests(unittest.TestCase):
    def test_explicit_allowlist_and_deterministic_public_fields(self):
        rows = [
            item("portable.second", title="Second"),
            item("portable.first", title="First", metadata={"user": "not-exported"},
                 updated_by="private-writer"),
            item("unapproved", body="Do not export this private customization.", required_vars=[]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixtures"
            result = export_report({"promotion_candidates": rows}, root,
                                   approved_keys=["portable.second", "portable.first"])
            self.assertEqual(result["written"], ["portable.first", "portable.second"])
            first = {path.name: path.read_bytes() for path in root.iterdir()}
            export_report({"promotion_candidates": list(reversed(rows))}, root,
                          approved_keys=["portable.first", "portable.second"])
            self.assertEqual(first, {path.name: path.read_bytes() for path in root.iterdir()})
            manifest = json.loads(first["manifest.json"])
            self.assertEqual(list(manifest["prompts"]), ["portable.first", "portable.second"])
            self.assertNotIn("private-writer", first["manifest.json"].decode())
            self.assertNotIn("not-exported", first["manifest.json"].decode())
            self.assertFalse((root / "unapproved.txt").exists())

    def test_empty_or_missing_selection_does_not_write_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            for keys in ([], ["missing"]):
                with self.subTest(keys=keys), self.assertRaises(ValueError):
                    export_report({"promotion_candidates": [item()]}, root, approved_keys=keys)
                self.assertFalse(root.exists())

    def test_private_scopes_and_malformed_metadata_are_held(self):
        invalid = [
            {"scope_type": "bot", "scope_id": "fixture-bot"},
            {"scope_id": "another-tenant"},
            {"key": "persona.private"},
            {"key": "../escape"},
            {"category": "persona"},
            {"category": "user"},
            {"sync_class": "install"},
            {"format": "executable"},
            {"required_vars": ["different"]},
            {"required_vars": {"topic": True}},
            {"required_vars": ["topic", "topic"]},
            {"body": ""},
            {"title": ""},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                row = item(**changes)
                result = export_report({"promotion_candidates": [row]}, Path(directory),
                                       approved_keys=[row["key"]])
                self.assertEqual(result["written"], [])
                self.assertIn(row["key"], result["held"])
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_private_content_is_not_written_or_echoed_in_receipts(self):
        private = [
            "Use sk-abcdefghijklmnopqrstuvwxyz1234567890",
            "Connect to 10.42.3.4",
            "Read /home/fixture-owner/private",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            "https://fixture-user:fixture-password@example.test/path",
            '{"password": "fixture-credential-value"}',
        ]
        for text in private:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                row = item(body=text, required_vars=[])
                result = export_report({"promotion_candidates": [row]}, Path(directory),
                                       approved_keys=[row["key"]])
                self.assertEqual(result["written"], [])
                self.assertNotIn(text, json.dumps(result))
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_installation_terms_in_titles_are_also_held(self):
        with tempfile.TemporaryDirectory() as directory:
            row = item(title="FixtureOwner's rules")
            result = export_report({"promotion_candidates": [row]}, Path(directory),
                                   ["FixtureOwner"], approved_keys=[row["key"]])
            self.assertEqual(result["held"], {row["key"]: ["installation-specific term"]})

    def test_literal_json_and_one_pass_replacements_share_runtime_parser(self):
        body = 'Example: {"outer": {"inner": 1}}. Use {topic}. Escaped: {{literal}}'
        self.assertEqual(extract_placeholders(body), ["topic"])
        self.assertEqual(validate_promotion_item(item(body=body)), [])
        self.assertEqual(validate_promotion_item(item(body="Optional: {topic}", required_vars=[])), [])
        self.assertIn('{"outer": {"inner": 1}}', render_template(body, {"topic": "{not_a_second_pass}"}))
        self.assertIn("{not_a_second_pass}", render_template(body, {"topic": "{not_a_second_pass}"}))

    def test_selected_fetch_reads_only_exact_global_keys(self):
        from io import BytesIO

        calls = []

        def urlopen(url, **_kwargs):
            calls.append(url)
            key = "portable.first" if "portable.first?" in url else "portable.second"
            return BytesIO(json.dumps(item(key=key, metadata={"sync": "product"})).encode())

        with patch("urllib.request.urlopen", urlopen):
            report = _fetch_selected("https://operator-selected.example.test", ["portable.second", "portable.first"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(url.endswith("?scope_type=global&exact=true") for url in calls))
        self.assertFalse(any("/bots" in url or "/sync/drift" in url for url in calls))
        self.assertEqual([row["key"] for row in report["promotion_candidates"]], ["portable.first", "portable.second"])

    def test_export_cli_requires_key_before_any_network_call(self):
        with patch("urllib.request.urlopen") as network, redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                main(["export", "--root", "/unused"])
        network.assert_not_called()

    def test_default_lag_and_intentional_override_are_separate_states(self):
        from types import SimpleNamespace

        def row(key, body, writer, metadata):
            return SimpleNamespace(
                key=key, body=body, updated_by=writer, metadata_json=json.dumps(metadata),
                scope_type="global", scope_id="*", category="agents", title=key,
                format="plain_text", required_vars_json=None, updated_at=None,
            )

        rows = [
            row("managed", "old", "system:seed", {}),
            row("custom", "mine", "tenant", {}),
            row("explicit", "old", "system:seed", {"default_mode": "override"}),
        ]
        definitions = {key: SimpleNamespace(default_body=lambda: "new", title=key, category="agents")
                       for key in ("managed", "custom", "explicit")}
        report = drift_report(rows, definitions)
        self.assertEqual(report["drift_count"], 1)
        self.assertEqual([entry["key"] for entry in report["drift"]], ["managed"])
        self.assertEqual({entry["key"] for entry in report["overrides"]}, {"custom", "explicit"})
        self.assertEqual({entry["key"] for entry in report["promotion_candidates"]}, {"custom", "explicit"})
        self.assertFalse(is_following(rows[2]))
        rows[0].scope_type = "bot"
        rows[0].metadata_json = json.dumps({"sync": "product"})
        self.assertEqual(classify(rows[0]), "install")
        self.assertFalse(is_following(rows[0]))


if __name__ == "__main__":
    unittest.main()
