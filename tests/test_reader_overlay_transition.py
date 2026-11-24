"""TASK-2590: the app reconcile adopts and detaches the reader broker overlay.

The app's file list comes from its own Compose label, never from ``.env``.
Forward: an overlay binding with ``adopt_files`` = the four-file baseline.
Inverse: a binding with no overlay and ``adopt_files`` = the enabled label,
which renders from exactly the baseline. Pure functions, no Docker.
"""
from __future__ import annotations

import pytest

from llm_bawt.ops.app_config_reconcile import approval_text, validate_label_compose_config  # noqa: F401
from llm_bawt.ops.compose_render import ComposeError, compose_source, overlay_files_valid

PROJECT = "/srv/dev/llm-bawt"
ROOT = "/srv/dev/deployment/llm-bawt/reader-broker"
OVERLAY = f"{ROOT}/reader-broker.yml"
BASELINE = [f"{PROJECT}/{n}" for n in ("docker-compose.yml", "docker-compose.dev.yml",
                                         "docker-compose.gpu.yml", "docker-compose.ops.yml")]
ENABLED = [*BASELINE, OVERLAY]
RENDERER = "docker@sha256:" + "a" * 64


def _spec(**config):
    config = {"renderer_image": RENDERER, **config}
    spec = {"compose_config": config}
    validate_label_compose_config(spec)
    return spec


FORWARD = dict(overlay_root=ROOT, overlay_files=["reader-broker.yml"], adopt_files=BASELINE)
INVERSE = dict(adopt_files=ENABLED)


def _labelled(files):
    return {"Config": {"Labels": {"com.docker.compose.project.working_dir": PROJECT,
                                  "com.docker.compose.project.config_files": ",".join(files)}}}


def _source(spec, files):
    source = compose_source(spec, _labelled(files))
    assert overlay_files_valid(spec["compose_config"], source["project_dir"], source["compose_files"])
    return source["compose_files"]


def test_default_binding_keeps_the_four_file_baseline():
    assert _source(_spec(), BASELINE) == BASELINE


def test_default_binding_refuses_the_enabled_label():
    with pytest.raises(ComposeError):
        _source(_spec(), ENABLED)


def test_forward_adopts_the_baseline_then_stays_enabled():
    spec = _spec(**FORWARD)
    assert _source(spec, BASELINE) == ENABLED  # first reconcile adopts
    assert _source(spec, ENABLED) == ENABLED   # later reconciles


def test_inverse_detaches_to_exactly_the_baseline_then_stays_there():
    spec = _spec(**INVERSE)
    assert _source(spec, ENABLED) == BASELINE  # first reconcile detaches
    assert _source(spec, BASELINE) == BASELINE  # later reconciles: ordinary label check
    assert "Detaches a container labelled" in approval_text.__globals__["_overlay_text"](spec)


@pytest.mark.parametrize("config", [{}, FORWARD, INVERSE])
def test_an_unrelated_fifth_file_is_still_refused(config):
    spec = _spec(**config)
    for label in ([*BASELINE, f"{PROJECT}/docker-compose.extra.yml"],
                  [*BASELINE, "/srv/elsewhere/extra.yml"],
                  [*BASELINE, OVERLAY, "/srv/elsewhere/extra.yml"]):
        with pytest.raises(ComposeError):
            _source(spec, label)


def test_detach_refuses_a_list_that_is_not_project_files_plus_an_overlay():
    for adopt in (BASELINE, [OVERLAY, *BASELINE]):
        with pytest.raises(ComposeError):
            _source(_spec(adopt_files=adopt), adopt)
