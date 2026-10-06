"""Structural regression coverage for the split history route facade."""

from llm_bawt.service.routes import history


def test_history_facade_preserves_route_order_and_names() -> None:
    routes = [
        (route.path, tuple(sorted(route.methods or ())), route.name)
        for route in history.router.routes
    ]

    assert routes == [
        ("/v1/history", ("GET",), "get_history"),
        ("/v1/history/around", ("GET",), "get_history_around"),
        ("/v1/history/timeline", ("GET",), "get_history_timeline"),
        ("/v1/history/timeline/anchors", ("POST",), "get_history_timeline_anchors"),
        (
            "/v1/history/timeline/prompts",
            ("GET",),
            "get_history_timeline_prompts",
        ),
        ("/v1/history/search", ("POST",), "search_history"),
        ("/v1/history/search", ("GET",), "search_history_get"),
        ("/v1/history/search_all", ("POST",), "search_all_history"),
        ("/v1/history", ("DELETE",), "clear_history"),
        ("/v1/history/{message_id}", ("DELETE",), "delete_message"),
        (
            "/v1/history/summarize/preview",
            ("GET",),
            "preview_summarizable_sessions",
        ),
        ("/v1/history/summarize", ("POST",), "summarize_history"),
        (
            "/v1/history/summarize/rebuild",
            ("POST",),
            "rebuild_history_summaries",
        ),
        ("/v1/history/summaries", ("GET",), "list_summaries"),
        ("/v1/history/context-seed", ("GET",), "get_context_seed"),
        (
            "/v1/history/summary/{summary_id}",
            ("DELETE",),
            "delete_summary",
        ),
    ]


def test_history_facade_keeps_service_imports_stable() -> None:
    from llm_bawt.service.routes import history_pages, history_seed

    assert history.build_context_seed is history_seed.build_context_seed
    assert history.maybe_build_session_seed is history_seed.maybe_build_session_seed
    assert history._clear_history_direct is history_pages._clear_history_direct
