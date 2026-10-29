"""History summarization orchestration and summary-management routes."""

from fastapi import APIRouter, HTTPException, Query

from ...memory.summarization_limits import resolve_summarization_limits
from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import (
    DeleteSummaryResponse,
    ListSummariesResponse,
    SummarizableSession,
    SummarizePreviewResponse,
    SummarizeResponse,
    SummaryInfo,
)

management_router = APIRouter()
delete_router = APIRouter()
log = get_service_logger(__name__)


def _resolve_summarization_limits(service, model_alias: str | None) -> tuple[int, int]:
    """Return (max_prompt_tokens, max_chunk_tokens) tuned for the active model.

    TASK-858: the budget rules moved to ``memory.summarization_limits`` so the
    background job and the ``/new`` pre-seed size their prompts the same way
    this route always has. This stays as the service-shaped adapter.
    """
    return resolve_summarization_limits(service.config, model_alias)


def _invalidate_bot_history_cache(service, bot_id: str) -> None:
    """Invalidate in-memory history views for a bot after summary writes."""
    for (_model, cached_bot_id, _user), llm_bawt in service._llm_bawt_cache.items():
        if cached_bot_id != bot_id:
            continue
        invalidate = getattr(llm_bawt, "invalidate_history_cache", None)
        if callable(invalidate):
            invalidate()


def _build_summary_callable(service, bot_id: str, user_id: str = "system", model: str | None = None):
    """Create a summarization callable using the service's model resolution path."""
    from ...memory.summarization import (
        _extract_json_object,
        compress_structured_summary_text,
        format_session_for_summarization,
        is_summary_low_quality,
    )
    from ...models.message import Message
    from ...prompt_registry import PromptResolver

    from ...runtime_settings import resolve_job_model

    # TASK-610: summarization model from the ONE global Tier-1 job dict
    # (model=None => inherit maintenance_model, as before).
    requested_model = (
        model
        or service.config.resolve_summarization_job().get("model")
        or resolve_job_model(service.config, "maintenance_model")
    )
    model_alias = None
    client = None
    max_prompt_tokens = 6000
    max_chunk_tokens = 4000
    quality_retry_enabled = bool(getattr(service.config, "SUMMARIZATION_QUALITY_RETRY", False))

    try:
        # TASK-857: summarization is a GLOBAL job (TASK-522) — resolve its
        # model with no bot identity. This used to route through
        # ``_resolve_request_model(bot_id)`` + ``_get_llm_bawt(bot_id=...)``,
        # which on an all-agent deployment handed back the *bot's* agent
        # backend client (harness-typed ``claude-code``) instead of a plain
        # API client, so every route-driven rebuild fell through to the
        # heuristic. ``_get_background_client`` is the one job-model resolver.
        client, model_alias = service._get_background_client(
            model_override=requested_model, job="history_summary_rebuild",
            allow_fallback=not model,
        )
        if client is None:
            raise RuntimeError(
                f"no background job client for model '{requested_model}'"
            )
        max_prompt_tokens, max_chunk_tokens = _resolve_summarization_limits(service, model_alias)
        log.info("History summarization route using model: %s", model_alias)
        log.info(
            "History summarization limits: prompt<=%s tokens, chunk<=%s tokens",
            max_prompt_tokens,
            "disabled" if max_chunk_tokens <= 0 else max_chunk_tokens,
        )
    except Exception as e:
        log.error(f"Failed to resolve/load model for history summarization route: {e}")
        client = None

    prompt_resolver = PromptResolver(service.config)

    def summarize_with_loaded_client(session) -> str | None:
        if not client:
            return None

        conversation_text = format_session_for_summarization(session)
        prompt = prompt_resolver.render(
            key="history.summarization.single",
            variables={"messages": conversation_text},
        )
        estimated_tokens = len(prompt) // 4
        log.debug(
            "Per-session summarization: %s messages, ~%s estimated prompt tokens",
            session.message_count,
            f"{estimated_tokens:,}",
        )
        if estimated_tokens > max_prompt_tokens:
            log.warning(
                "Session too large (%s estimated tokens > %s), needs chunking",
                estimated_tokens,
                max_prompt_tokens,
            )
            return None

        try:
            messages = [
                Message(
                    role="system",
                    content="You are a helpful assistant that summarizes conversations concisely.",
                ),
                Message(role="user", content=prompt),
            ]
            response = client.query(
                messages=messages,
                max_tokens=520,
                temperature=0.3,
                plaintext_output=True,
                stream=False,
            )
            if not response:
                return None
            error_indicators = ["Error:", "exception occurred", "exceed context window", "tokens exceed"]
            if any(indicator.lower() in response.lower() for indicator in error_indicators):
                log.error(f"LLM returned error: {response[:100]}...")
                return None
            response_text = compress_structured_summary_text(response.strip(), source_session=session)
            if quality_retry_enabled and is_summary_low_quality(response_text, source_session=session):
                revision_prompt = (
                    "Your previous summary was too vague. Rewrite with concrete specifics from source material only. "
                    "Explicitly name entities, actions taken, tools/searches used, and unresolved items. "
                    "Keep required sections including Key Details.\n\n"
                    f"Original conversation:\n{conversation_text}\n\n"
                    f"Previous summary:\n{response_text}"
                )
                revised = client.query(
                    messages=[
                        Message(
                            role="system",
                            content="You are a precise summarizer. Avoid generic phrasing and preserve concrete facts.",
                        ),
                        Message(role="user", content=revision_prompt),
                    ],
                    max_tokens=620,
                    temperature=0.1,
                    plaintext_output=True,
                    stream=False,
                )
                if revised and not is_summary_low_quality(revised, source_session=session):
                    return compress_structured_summary_text(revised.strip(), source_session=session)
            return response_text
        except Exception as e:
            log.error(f"LLM summarization failed: {e}")
            return None

    def summarize_batch_with_loaded_client(sessions) -> dict[int, str] | None:
        if not client or not sessions:
            return None

        try:
            session_lines: list[str] = []
            for idx, session in enumerate(sessions, start=1):
                conversation_text = format_session_for_summarization(session)
                session_lines.append(
                    "\n".join(
                        [
                            f"### SESSION {idx}",
                            f"SESSION_INDEX: {idx}",
                            f"MESSAGE_COUNT: {session.message_count}",
                            "CONVERSATION:",
                            conversation_text,
                        ]
                    )
                )

            sessions_blob = "\n\n".join(session_lines)
            prompt = prompt_resolver.render(
                key="history.summarization.batch",
                variables={"sessions_blob": sessions_blob},
            )
            estimated_tokens = len(prompt) // 4
            log.info(
                "Batch summarization: %s sessions, ~%s estimated prompt tokens (limit %s)",
                len(sessions),
                f"{estimated_tokens:,}",
                f"{max_prompt_tokens:,}",
            )
            if estimated_tokens > max_prompt_tokens:
                log.warning(
                    "Batch too large (%s estimated tokens > %s), using per-session fallback",
                    estimated_tokens,
                    max_prompt_tokens,
                )
                return None

            model_max_tokens = int(service.config.get_model_max_tokens(model_alias) or 4096)
            requested_output_tokens = max(900, min(model_max_tokens, 320 * len(sessions)))

            # xAI OpenAI-compatible endpoint is typically most reliable with json_object mode.
            response_format = {"type": "json_object"}

            messages = [
                Message(
                    role="system",
                    content="You are a helpful assistant that summarizes conversations with strict JSON output.",
                ),
                Message(role="user", content=prompt),
            ]
            response = client.query(
                messages=messages,
                max_tokens=requested_output_tokens,
                temperature=0.2,
                plaintext_output=True,
                stream=False,
                response_format=response_format,
            )
            if not response:
                return None

            payload = _extract_json_object(response)
            if not payload:
                log.error("Batch summarization returned non-JSON output; retrying without response_format")
                retry_response = client.query(
                    messages=messages,
                    max_tokens=requested_output_tokens,
                    temperature=0.1,
                    plaintext_output=True,
                    stream=False,
                )
                payload = _extract_json_object(retry_response or "")
                if not payload:
                    preview = (retry_response or response or "")[:200].replace("\n", " ")
                    log.error("Batch summarization still non-JSON after retry: %s", preview)
                    return None

            summaries = payload.get("summaries")
            if not isinstance(summaries, list):
                log.error("Batch summarization JSON missing 'summaries' list")
                return None

            mapped: dict[int, str] = {}
            for item in summaries:
                if not isinstance(item, dict):
                    continue
                try:
                    session_index = int(item.get("session_index"))
                except (TypeError, ValueError):
                    continue
                if session_index < 1 or session_index > len(sessions):
                    continue

                summary = str(item.get("summary", "")).strip()
                key_details = str(item.get("key_details", "")).strip()
                intent = str(item.get("intent", "")).strip()
                tone = str(item.get("tone", "")).strip()
                open_loops = str(item.get("open_loops", "")).strip()
                cross_links = str(item.get("cross_session_links", "")).strip()
                if cross_links:
                    open_loops = f"{open_loops} Cross-session links: {cross_links}".strip()

                mapped[session_index] = compress_structured_summary_text(
                    "\n".join(
                        [
                            f"Summary: {summary}".rstrip(),
                            f"Key Details: {key_details}".rstrip(),
                            f"Intent: {intent}".rstrip(),
                            f"Tone: {tone}".rstrip(),
                            f"Open Loops: {open_loops}".rstrip(),
                        ]
                    ).strip(),
                    source_session=sessions[session_index - 1],
                )

            # Drop vague batch items so they get heuristic fallback instead.
            pre_filter_count = len(mapped)
            mapped = {
                idx: txt
                for idx, txt in mapped.items()
                if not is_summary_low_quality(txt, source_session=sessions[idx - 1] if idx <= len(sessions) else None)
            }
            if pre_filter_count != len(mapped):
                log.info(
                    "Batch quality filter: %s/%s sessions passed (%s filtered)",
                    len(mapped),
                    pre_filter_count,
                    pre_filter_count - len(mapped),
                )

            return mapped if mapped else None
        except Exception as e:
            log.error(f"Batch LLM summarization failed: {e}")
            return None

    return summarize_with_loaded_client, summarize_batch_with_loaded_client, max_chunk_tokens

@management_router.get("/v1/history/summarize/preview", response_model=SummarizePreviewResponse, tags=["History"])
def preview_summarizable_sessions(
    bot_id: str = Query(None, description="Bot ID"),
):
    """Preview sessions that would be summarized (dry run)."""
    from datetime import datetime
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        # Compute context budget for budget-driven eligibility
        # TASK-609: single Tier-2 budget authority
        _, _, max_context_tokens = service.config.resolve_context_budget(None)

        summarizer = HistorySummarizer(
            service.config,
            effective_bot_id,
            max_context_tokens=max_context_tokens,
        )
        sessions = summarizer.preview_summarizable_sessions()

        session_infos = []
        total_messages = 0

        for session in sessions:
            total_messages += session.message_count

            # Get first and last user messages for preview
            user_msgs = [m for m in session.messages if m.get("role") == "user"]
            first_msg = user_msgs[0].get("content", "")[:100] if user_msgs else ""
            last_msg = user_msgs[-1].get("content", "")[:100] if len(user_msgs) > 1 else first_msg

            session_infos.append(SummarizableSession(
                start_timestamp=session.start_timestamp,
                end_timestamp=session.end_timestamp,
                start_time=datetime.fromtimestamp(session.start_timestamp).strftime("%Y-%m-%d %H:%M"),
                end_time=datetime.fromtimestamp(session.end_timestamp).strftime("%H:%M"),
                message_count=session.message_count,
                first_message=first_msg,
                last_message=last_msg,
            ))

        return SummarizePreviewResponse(
            bot_id=effective_bot_id,
            sessions=session_infos,
            total_messages=total_messages,
        )
    except Exception as e:
        log.error(f"Failed to preview summarizable sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@management_router.post("/v1/history/summarize", response_model=SummarizeResponse, tags=["History"])
def summarize_history(
    bot_id: str = Query(None, description="Bot ID"),
    use_heuristic: bool = Query(False, description="Fall back to heuristic if LLM fails"),
    model: str | None = Query(None, description="Optional model alias override"),
):
    """Summarize eligible history sessions."""
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        summarize_with_loaded_client, summarize_batch_with_loaded_client, max_chunk_tokens = _build_summary_callable(
            service=service,
            bot_id=effective_bot_id,
            user_id="system",
            model=model,
        )
        # Compute context budget from model
        resolved_model = model or getattr(service, '_default_model', None)
        max_prompt_tokens, max_chunk_tokens_resolved = _resolve_summarization_limits(service, resolved_model)
        if max_chunk_tokens == 0:
            max_chunk_tokens = max_chunk_tokens_resolved
        # TASK-609: single Tier-2 budget authority
        _, _, max_context_tokens = service.config.resolve_context_budget(resolved_model)

        summarizer = HistorySummarizer(
            service.config,
            effective_bot_id,
            summarize_fn=summarize_with_loaded_client,
            summarize_batch_fn=summarize_batch_with_loaded_client,
            max_context_tokens=max_context_tokens,
        )
        result = summarizer.summarize_eligible_sessions(
            use_heuristic_fallback=use_heuristic,
            max_tokens_per_chunk=max_chunk_tokens,
        )
        _invalidate_bot_history_cache(service, effective_bot_id)

        return SummarizeResponse(
            success=True,
            sessions_summarized=result.get("sessions_summarized", 0),
            messages_summarized=result.get("messages_summarized", 0),
            sessions_targeted=result.get("sessions_targeted"),
            summaries_replaced=result.get("summaries_replaced"),
            errors=result.get("errors", []),
        )
    except Exception as e:
        log.error(f"Failed to summarize history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@management_router.post("/v1/history/summarize/rebuild", response_model=SummarizeResponse, tags=["History"])
def rebuild_history_summaries(
    bot_id: str = Query(None, description="Bot ID"),
    sessions: int = Query(5, ge=0, le=2000, description="How many recent eligible sessions to rebuild (0 = all eligible)"),
    use_heuristic: bool = Query(False, description="Fall back to heuristic if LLM fails"),
    start_ts: float | None = Query(None, description="Optional Unix start timestamp filter"),
    end_ts: float | None = Query(None, description="Optional Unix end timestamp filter"),
    purge_existing: bool = Query(False, description="Delete existing historical summaries in range before rebuild"),
    model: str | None = Query(None, description="Optional model alias override"),
):
    """Rebuild summaries for the last N eligible sessions from original history."""
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        summarize_with_loaded_client, summarize_batch_with_loaded_client, max_chunk_tokens = _build_summary_callable(
            service=service,
            bot_id=effective_bot_id,
            user_id="system",
            model=model,
        )
        # Compute context budget from model
        resolved_model = model or getattr(service, '_default_model', None)
        # TASK-609: single Tier-2 budget authority
        _, _, max_context_tokens = service.config.resolve_context_budget(resolved_model)

        summarizer = HistorySummarizer(
            service.config,
            effective_bot_id,
            summarize_fn=summarize_with_loaded_client,
            summarize_batch_fn=summarize_batch_with_loaded_client,
            max_context_tokens=max_context_tokens,
        )
        result = summarizer.rebuild_recent_sessions(
            session_limit=sessions,
            use_heuristic_fallback=use_heuristic,
            max_tokens_per_chunk=max_chunk_tokens,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            purge_existing=purge_existing,
        )
        _invalidate_bot_history_cache(service, effective_bot_id)
        return SummarizeResponse(
            success=True,
            sessions_summarized=result.get("sessions_summarized", 0),
            messages_summarized=result.get("messages_summarized", 0),
            sessions_targeted=result.get("sessions_targeted"),
            summaries_replaced=result.get("summaries_replaced"),
            summaries_purged=result.get("summaries_purged"),
            errors=result.get("errors", []),
        )
    except Exception as e:
        log.error(f"Failed to rebuild history summaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@management_router.get("/v1/history/summaries", response_model=ListSummariesResponse, tags=["History"])
def list_summaries(
    bot_id: str = Query(None, description="Bot ID"),
):
    """List existing history summaries."""
    from datetime import datetime
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        summarizer = HistorySummarizer(service.config, effective_bot_id)
        summaries = summarizer.list_summaries()

        summary_infos = []
        for summ in summaries:
            start_ts = summ.get("session_start")
            end_ts = summ.get("session_end")

            summary_infos.append(SummaryInfo(
                id=summ.get("id", ""),
                content=summ.get("content", ""),
                timestamp=summ.get("timestamp", 0),
                session_start_time=datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M") if start_ts else None,
                session_end_time=datetime.fromtimestamp(end_ts).strftime("%H:%M") if end_ts else None,
                message_count=summ.get("message_count", 0),
                method=summ.get("method", "unknown"),
            ))

        return ListSummariesResponse(
            bot_id=effective_bot_id,
            summaries=summary_infos,
            total_count=len(summary_infos),
        )
    except Exception as e:
        log.error(f"Failed to list summaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@delete_router.delete("/v1/history/summary/{summary_id}", response_model=DeleteSummaryResponse, tags=["History"])
def delete_summary(
    summary_id: str,
    bot_id: str = Query(None, description="Bot ID"),
):
    """Delete a summary and restore the original messages."""
    from ...memory.summarization import HistorySummarizer

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        summarizer = HistorySummarizer(service.config, effective_bot_id)
        result = summarizer.delete_summary(summary_id)

        if result.get("success"):
            return DeleteSummaryResponse(
                success=True,
                summary_id=result.get("summary_id"),
                messages_restored=result.get("messages_restored", 0),
            )
        else:
            return DeleteSummaryResponse(
                success=False,
                detail=result.get("error", "Failed to delete summary"),
            )
    except Exception as e:
        log.error(f"Failed to delete summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
