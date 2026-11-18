"""Catalog-owned approval text, enriched from the immutable operation snapshot."""
from __future__ import annotations

DEFAULT_PROMPT = "Run this operator-configured operation?"


def approval_text(snapshot: dict) -> str:
    prefix = snapshot["operation"].get("approval_prompt_prefix") or DEFAULT_PROMPT
    prompt = prefix
    if snapshot.get("release_source"):
        source, spec = snapshot["release_source"], snapshot["spec"]
        prompt = (
            f"{prefix}\n"
            f"Source: {spec['github_repository']}@{source['expected_sha']}\n"
            f"Target: {spec['deploy_operation']} (currently {source.get('expected_current_image_id', 'unknown')})\n"
            "Deploys only if the build, release identity and production baseline verify. "
            "Unknown or changed outcomes stop for inspection; rollback remains separate."
        ) if source.get("deploy_authorization") == "after_verified_build" else (
            f"{prefix}\nBawtHub {spec['github_repository']}@{source['expected_sha']}\n"
            "Production deployment requires a separate approval."
        )
    if snapshot.get("routing"):
        routing, spec = snapshot["routing"], snapshot["spec"]
        prompt = (
            f"{prefix}\nTarget: {spec['container_name']} ({routing['container_id']})\n"
            f"Unchanged image: {routing['image_ref']}\n"
            f"Configuration fingerprint: {routing['config_hash']}\n"
            f"Before: {routing['before_rule']}\nAfter: {routing['after_rule']}\n"
            "No build, pull, environment, mount or permission changes. "
            "Verifies the hostname-routed health report; restores the previous container "
            "if verification fails. Changed target/configuration cancels execution."
        )
    if snapshot.get("host_key"):
        from . import host_key

        prompt = host_key.approval_text(snapshot["spec"], snapshot["host_key"], prefix)
    if snapshot.get("remove"):
        from . import container_remove

        prompt = container_remove.approval_text(snapshot["remove"], prefix)
    if snapshot.get("config"):
        from . import app_config_reconcile, config_reconcile, dev_config_reconcile, tagged_config_reconcile

        if snapshot["spec"].get("action") == app_config_reconcile.ACTION:
            prompt = app_config_reconcile.snapshot_approval_text(snapshot, prefix)
        elif snapshot["spec"].get("action") == tagged_config_reconcile.ACTION:
            prompt = tagged_config_reconcile.approval_text(snapshot["spec"], snapshot["config"], prefix)
        elif snapshot["spec"].get("action") == dev_config_reconcile.ACTION:
            prompt = dev_config_reconcile.approval_text(snapshot["spec"], snapshot["config"], prefix)
        else:
            prompt = config_reconcile.approval_text(snapshot["spec"], snapshot["config"], prefix)
    if snapshot.get("plan"):
        from . import plan_approval

        prompt = plan_approval.approval_text(snapshot, prefix)
    if snapshot.get("sequence"):
        from . import sequence_approval

        prompt = sequence_approval.approval_text(snapshot, prefix)
    from .execution_mode import with_card_lines
    from .operation_revision import with_safety_lines

    return with_safety_lines(with_card_lines(prompt, snapshot), snapshot)
