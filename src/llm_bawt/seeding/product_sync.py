"""Stage product prompt and Setup catalog changes for exact source review."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from shutil import copyfile, rmtree
from tempfile import mkdtemp

from ..prompt_sync import _fetch_selected, export_report
from ..prompt_synced_defaults import MANIFEST_NAME, SYNCED_DIR
from .bot_presets import GENERATED_CATALOG, PresetError
from .setup_live_source import is_slug, read_live_setup_source, read_live_setup_source_pgq
from .setup_role_dependencies import portable_role_prompt_revisions
from .setup_seed_sync import SetupSeedSnapshot, live_setup_snapshot, stage_catalog

CATALOG_FILE = GENERATED_CATALOG.name


def _stage_prompts(api: str, keys: Collection[str], shipped: Path, stage: Path,
                   install_terms: Collection[str]) -> tuple[dict, dict[str, str]]:
    if not keys:
        return {"status": "not_selected"}, {}
    candidate = None
    try:
        candidate = Path(mkdtemp(prefix="prompts-", dir=stage))
        manifest = shipped / MANIFEST_NAME
        if manifest.is_file():
            copyfile(manifest, candidate / MANIFEST_NAME)
        result = export_report(_fetch_selected(api, keys), candidate, install_terms, approved_keys=keys)
        for path in candidate.iterdir():
            path.chmod(0o600)
        written = set(result["written"])
        portable = {key: revision for key, revision in
                    portable_role_prompt_revisions(candidate, install_terms=install_terms).items() if key in written}
    except Exception:
        # Fetch/DB/IO exceptions may contain paths, credentials or row values.
        if candidate is not None:
            rmtree(candidate, ignore_errors=True)
        return {"status": "failed", "reason": "Selected prompt export failed"}, {}
    outcome = {"status": "held" if result["held"] else "staged",
               "written": len(written), "held": len(result["held"])}
    if written:
        outcome["directory"] = candidate.name
    else:
        rmtree(candidate)
    return outcome, portable


def sync_product(
    *, api: str, keys: Collection[str], shipped: Path, stage: Path,
    install_terms: Collection[str], read_snapshot: Callable[[Mapping[str, str]], SetupSeedSnapshot] | None,
) -> dict:
    # Reject a shared checkout or symlink as the staging root. Be private even
    # before chmod on new prompt files in their owner-only candidate directory.
    if (stage.is_symlink() or not stage.is_dir() or stage.stat().st_uid != os.geteuid()
            or stage.stat().st_mode & 0o077):
        raise ValueError("Stage must be an existing owner-only directory")
    prompts, written = _stage_prompts(api, keys, shipped, stage, install_terms)
    catalog: dict = {"status": "not_selected"}
    if read_snapshot is not None:
        try:
            shipped_revisions = portable_role_prompt_revisions(shipped, install_terms=install_terms)
            # An explicitly selected role prompt cannot silently fall back to an
            # older default when the selected change was held or failed.
            available = {k: v for k, v in shipped_revisions.items() if k not in keys} | written
        except Exception:
            catalog = {"status": "failed", "reason": "Could not read the shipped prompt defaults"}
        else:
            try:
                snapshot = read_snapshot(available)
            except PresetError as held:
                catalog = {"status": "held", "reason": str(held)}
            except Exception:
                catalog = {"status": "failed", "reason": "Could not read the eligible Setup source"}
            else:
                try:
                    target = stage / CATALOG_FILE
                    if target.is_symlink() or (target.exists() and not target.is_file()):
                        raise OSError("Not a regular catalog file")
                    if target.exists():
                        target.chmod(0o600)
                    held: dict[str, list[str]] = {}
                    changed = stage_catalog(lambda: snapshot, target, install_terms=install_terms, held=held)
                    catalog = {"status": "staged" if changed else "unchanged", "file": CATALOG_FILE,
                               "bots": sorted(set(snapshot.eligible_slugs) - held.keys())}
                    if held:
                        # Reported by slug and kind of content; the bot is left out until its text is fixed.
                        catalog["held_bots"] = {slug: held[slug] for slug in sorted(held)}
                except PresetError as held:
                    catalog = {"status": "held", "reason": str(held)}
                except Exception:
                    catalog = {"status": "failed", "reason": "Could not stage the Setup catalog"}
    if catalog["status"] in {"held", "failed"}:
        # Retained bytes are recovery evidence, not a verified current catalog.
        catalog["previous_catalog_reusable"] = False
    return {"prompts": prompts, "catalog": catalog}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True, help="Existing private candidate directory (0700)")
    parser.add_argument("--api", help="Approved global prompt source API (required with --key)")
    parser.add_argument("--key", action="append", default=[], help="Explicitly approved prompt key; repeatable")
    parser.add_argument("--setup-catalog", action="store_true",
                        help="Generate the Setup catalog from the live, eligible bot lineup")
    parser.add_argument("--excluded-bots",
                        help="Bots to leave out of the catalog, comma-separated; explicit empty if none")
    parser.add_argument("--pgq", action="store_true", help="Read the lineup through pgq instead of app config")
    parser.add_argument("--install-term", action="append", default=[], help="Additional install-specific term; repeatable")
    args = parser.parse_args(argv)
    if not (args.key or args.setup_catalog):
        parser.error("select --key and/or --setup-catalog")
    if args.key and not args.api:
        parser.error("--key requires --api")
    read_snapshot = None
    if args.setup_catalog:
        if args.excluded_bots is None:
            parser.error("--setup-catalog requires --excluded-bots (explicitly empty if none)")
        excluded = [slug.strip().lower() for slug in args.excluded_bots.split(",") if slug.strip()]
        if not all(is_slug(slug) for slug in excluded):
            parser.error("--excluded-bots contains an invalid slug")

        def read_snapshot(revisions):
            if args.pgq:
                source = read_live_setup_source_pgq(excluded=excluded)
            else:
                from ..utils.config import Config

                source = read_live_setup_source(Config(), excluded=excluded)
            return live_setup_snapshot(source, revisions)

    terms = [t.strip() for t in (*args.install_term, *os.environ.get("PROMPT_SYNC_INSTALL_TERMS", "").split(","))
             if t.strip()]
    if args.setup_catalog:
        # No pattern recognizes a bare host or owner name, so the operator
        # names them. An excluded bot named in another bot's prompt is held too.
        if not terms:
            parser.error("--setup-catalog requires --install-term or PROMPT_SYNC_INSTALL_TERMS"
                         " (host, owner and other installation names)")
        terms += sorted({form for slug in excluded for form in (slug, slug.capitalize(), slug.upper())})
    try:
        result = sync_product(api=args.api or "", keys=args.key, shipped=SYNCED_DIR, stage=args.stage,
                              install_terms=terms, read_snapshot=read_snapshot)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    # A held bot is reported and needs attention even though the catalog staged.
    return int(any(stage["status"] in {"failed", "held"} for stage in result.values())
               or bool(result["catalog"].get("held_bots")))


if __name__ == "__main__":
    raise SystemExit(main())
