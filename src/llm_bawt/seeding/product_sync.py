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
from .setup_portable_input import is_slug, load_portable_input
from .setup_role_dependencies import portable_role_prompt_revisions
from .setup_seed_sync import SetupSeedSnapshot, approved_setup_snapshot, stage_catalog
from .setup_structure import read_setup_structure

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
                    changed = stage_catalog(lambda: snapshot, target, install_terms=install_terms)
                    catalog = {"status": "staged" if changed else "unchanged", "file": CATALOG_FILE}
                except PresetError as held:
                    catalog = {"status": "held", "reason": str(held)}
                except Exception:
                    catalog = {"status": "failed", "reason": "Could not stage the Setup catalog"}
    if catalog["status"] in {"held", "failed"}:
        # Retained bytes are recovery evidence, not a verified current catalog.
        catalog["previous_catalog_reusable"] = False
    return {"prompts": prompts, "catalog": catalog}


def _read_configured_source(revisions: Mapping[str, str], *, path: Path, sha: str,
                            excluded: Collection[str], denied_roles: Collection[str]) -> SetupSeedSnapshot:
    approved = load_portable_input(path, sha)
    from ..utils.config import Config

    structure = read_setup_structure(Config(), excluded=excluded, allowed_roles=approved.roles.keys(),
                                     denied_roles=denied_roles)
    return approved_setup_snapshot(structure, approved, revisions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True, help="Existing private candidate directory (0700)")
    parser.add_argument("--api", help="Approved global prompt source API (required with --key)")
    parser.add_argument("--key", action="append", default=[], help="Explicitly approved prompt key; repeatable")
    parser.add_argument("--setup-catalog", action="store_true", help="Read and stage the complete eligible roster")
    parser.add_argument("--portable-input", type=Path, help="Independently reviewed portable Setup source file")
    parser.add_argument("--portable-sha256", help="Approved SHA-256 of the exact portable source bytes")
    parser.add_argument("--excluded-bots", help="Reviewed bot exclusions, comma-separated; explicit empty if none")
    parser.add_argument("--excluded-roles", help="Reviewed role exclusions, comma-separated; explicit empty if none")
    parser.add_argument("--install-term", action="append", default=[], help="Additional install-specific term; repeatable")
    args = parser.parse_args(argv)
    if not (args.key or args.setup_catalog):
        parser.error("select --key and/or --setup-catalog")
    if args.key and not args.api:
        parser.error("--key requires --api")
    if args.setup_catalog and "PROMPT_SYNC_INSTALL_TERMS" not in os.environ and not args.install_term:
        parser.error("configure reviewed PROMPT_SYNC_INSTALL_TERMS or --install-term for the catalog")
    read_snapshot = None
    if args.setup_catalog:
        if (args.portable_input is None or not args.portable_sha256 or args.excluded_bots is None
                or args.excluded_roles is None):
            parser.error("--setup-catalog requires --portable-input, --portable-sha256, --excluded-bots and --excluded-roles")

        def exclusions(value, flag):
            slugs = [slug.strip().lower() for slug in value.split(",") if slug.strip()]
            if not all(is_slug(slug) for slug in slugs):
                parser.error(f"{flag} contains an invalid slug")
            return slugs

        excluded = exclusions(args.excluded_bots, "--excluded-bots")
        denied_roles = exclusions(args.excluded_roles, "--excluded-roles")

        def read_snapshot(revisions):
            return _read_configured_source(revisions, path=args.portable_input, sha=args.portable_sha256,
                                           excluded=excluded, denied_roles=denied_roles)

    terms = [t.strip() for t in (*args.install_term, *os.environ.get("PROMPT_SYNC_INSTALL_TERMS", "").split(","))
             if t.strip()]
    try:
        result = sync_product(api=args.api or "", keys=args.key, shipped=SYNCED_DIR, stage=args.stage,
                              install_terms=terms, read_snapshot=read_snapshot)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return int(any(stage["status"] in {"failed", "held"} for stage in result.values()))


if __name__ == "__main__":
    raise SystemExit(main())
