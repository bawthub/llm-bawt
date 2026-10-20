"""Native chat-tool streaming, separate from agent transport and SSE coordination.

Owns tool schema/executor setup and the text/tool/result continuation loop. The
worker retains queue consumption, bridge dispatch, cancellation and finalization.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from .chat_stream_worker import put_queue_item_threadsafe
from .logging import get_service_logger
from .turn_stream_context import TurnStreamContext

log = get_service_logger(__name__)


class NativeToolStream:
    """Per-turn native-tool iteration using the worker's shared context."""

    def __init__(self, ctx: TurnStreamContext, *, publish_event_direct: Callable[[dict], None]):
        self.ctx = ctx
        self.publish_event_direct = publish_event_direct

    def stream(self, messages, gen_kwargs):
        # Preserve eager tool setup and lazy provider execution from the worker.
        ctx = self.ctx
        llm_bawt = ctx.llm_bawt
        cancel_event = ctx.cancel_event
        loop, chunk_queue = ctx.loop, ctx.chunk_queue
        turn_log_id, trigger_message_id = ctx.turn_log_id, ctx.trigger_message_id
        bot_id, user_id = ctx.bot_id, ctx.user_id
        tool_call_details_holder = ctx.tool_call_details_holder
        tool_context_holder = ctx.tool_context_holder
        _publish_event_direct = self.publish_event_direct
        from ..tools.executor import ToolExecutor
        from ..tools.formats import get_format_handler
        from ..models.message import Message as Msg

        log.debug("Using native streaming with tools")

        tool_definitions = llm_bawt._get_tool_definitions() if llm_bawt.bot.uses_tools else []
        handler = get_format_handler(llm_bawt.tool_format)
        tools_schema = handler.get_tools_schema(tool_definitions)

        # TASK-215: the virtual trigger_animation tool has been
        # removed. The classifier picks the animation post-hoc.

        # Log tool names for debugging
        tool_names = [t.get("function", {}).get("name", "?") for t in tools_schema]
        ha_tools = [n for n in tool_names if n.startswith("Hass") or n in ("GetLiveContext",)]
        log.info(f"📋 {len(tools_schema)} tools in schema ({len(ha_tools)} HA: {', '.join(ha_tools[:5])}{'...' if len(ha_tools) > 5 else ''})")

        executor = ToolExecutor(
            memory_client=llm_bawt.memory,
            profile_manager=llm_bawt.profile_manager,
            search_client=llm_bawt.search_client,
            home_client=llm_bawt.home_client,
            ha_native_client=llm_bawt.ha_native_client,
            model_lifecycle=llm_bawt.model_lifecycle,
            config=llm_bawt.config,
            user_id=llm_bawt.user_id,
            bot_id=llm_bawt.bot_id,
        )

        def native_stream_with_tool_loop():
            """Stream with native tool support, handling tool calls inline."""
            import json as _json
            from ..tools.parser import ToolCall, strip_tool_result_tags

            current_msgs = list(messages)
            max_iterations = 3 if llm_bawt._ha_mode else 5
            has_executed_tools = False

            for iteration in range(max_iterations):
                if cancel_event.is_set():
                    return
                # After first tool execution, keep tools_schema
                # but force tool_choice="none" so the model
                # generates a text response.  Dropping the schema
                # entirely causes some providers (xAI) to ignore
                # function_call_output items and return nothing.
                if has_executed_tools:
                    current_tools = tools_schema
                    current_tool_choice = "none"
                else:
                    current_tools = tools_schema
                    current_tool_choice = "auto"

                # TASK-216: stream text tokens immediately instead
                # of buffering until we know whether a tool will
                # fire. The original buffer was a workaround for
                # models that emit preamble before tool calls
                # ("I don't have live access, let me check…").
                # Modern Claude / GPT-5 / Grok rarely do that, but
                # the buffering imposed 700-2700ms TTFT on every
                # tool-capable bot — which is what made voice mode
                # feel laggy (see scripts/probe_voice_latency.py).
                #
                # Trade-off: in the rare case a model does emit
                # preamble before a tool call, those tokens will
                # leak to the client. For voice that's harmless
                # (TTS speaks naturally); for chat the preamble
                # text shows up alongside the tool result, which
                # is acceptable.
                #
                # We still track yielded text in `yielded_text` so
                # downstream code that previously consumed
                # text_buffer (e.g. the no-tool-call early return)
                # can be reasoned about.
                yielded_text: list[str] = []

                for item in llm_bawt.client.stream_with_tools(
                    current_msgs,
                    tools_schema=current_tools,
                    tool_choice=current_tool_choice,
                    **gen_kwargs,
                ):
                    if cancel_event.is_set():
                        return
                    if isinstance(item, str):
                        # Stream immediately — no buffering.
                        yielded_text.append(item)
                        yield item
                    elif isinstance(item, dict) and "tool_calls" in item:
                        tool_calls = item["tool_calls"]

                        if not tool_calls:
                            # Empty tool_calls list — treat as pure text response.
                            # Text was already streamed above; nothing more to flush.
                            return

                        # TASK-215: trigger_animation interception removed.
                        # All tool calls in this stream are real.
                        real_tool_calls = tool_calls

                        # Tools follow streamed text. The text has
                        # already been delivered to the client; we
                        # can't take it back. (No-op kept here so
                        # downstream diff is small.)
                        yielded_text.clear()

                        # Emit tool calls as OpenAI delta.tool_calls, then execute
                        tool_results = []
                        for idx, tc in enumerate(real_tool_calls):
                            if cancel_event.is_set():
                                return
                            func = tc.get("function", {})
                            name = func.get("name", "")
                            call_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                            args_str = func.get("arguments", "{}")
                            try:
                                args = _json.loads(args_str) if args_str else {}
                            except _json.JSONDecodeError:
                                args = {}

                            log.info(f"🔧 {name}({args})")
                            # Push OpenAI-format tool_call delta to queue
                            put_queue_item_threadsafe(
                                loop,
                                chunk_queue,
                                {
                                    "_type": "tool_call_delta",
                                    "index": idx,
                                    "id": call_id,
                                    "name": name,
                                    "arguments": args_str,
                                },
                            )
                            # Publish tool_start directly to Redis
                            # (bypasses chunk_queue so events reach
                            # Redis even if the SSE generator is dead)
                            _publish_event_direct({
                                "_type": "tool_event",
                                "event": "tool_start",
                                "turn_id": turn_log_id,
                                "trigger_message_id": trigger_message_id,
                                "bot_id": bot_id,
                                "user_id": user_id,
                                "tool_name": name,
                                "arguments": args,
                                "call_id": call_id,
                                "iteration": iteration + 1,
                                "ts": time.time(),
                            })

                            tool_call_obj = ToolCall(name=name, arguments=args, raw_text="")
                            raw_result = executor.execute(tool_call_obj)
                            # Strip <tool_result> XML tags — native tool path
                            # sends results as structured function_call_output
                            # items, not inline text.
                            result = strip_tool_result_tags(raw_result)
                            tool_results.append({
                                "tool_call_id": tc.get("id", ""),
                                "content": result,
                            })
                            # Publish tool_end directly to Redis
                            _publish_event_direct({
                                "_type": "tool_event",
                                "event": "tool_end",
                                "turn_id": turn_log_id,
                                "trigger_message_id": trigger_message_id,
                                "bot_id": bot_id,
                                "user_id": user_id,
                                "tool_name": name,
                                "arguments": args,
                                "call_id": call_id,
                                "result": result or "",
                                "iteration": iteration + 1,
                                "ts": time.time(),
                            })

                            # Persist for TurnLog.tool_calls_json
                            tool_call_details_holder.append({
                                "tool": name,
                                "arguments": args,
                                "call_id": call_id,
                                "result": result or "",
                                "iteration": iteration + 1,
                            })

                        has_executed_tools = True

                        # Signal finish_reason: "tool_calls" to consumer
                        put_queue_item_threadsafe(
                            loop,
                            chunk_queue,
                            {"_type": "tool_calls_finish"},
                        )

                        # Store tool context
                        tool_context_holder[0] = "\n\n".join(
                            f"[{tc['function']['name']}]\n{tr['content']}"
                            for tc, tr in zip(real_tool_calls, tool_results)
                        )

                        # Build continuation messages — empty content so the
                        # follow-up pass is grounded on tool results, not the
                        # discarded pre-tool fallback text.
                        assistant_msg = Msg(
                            role="assistant",
                            content="",
                            tool_calls=[
                                {"id": tc.get("id"), "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                                for tc in real_tool_calls
                            ],
                        )
                        current_msgs.append(assistant_msg)

                        for tc, tr in zip(real_tool_calls, tool_results):
                            current_msgs.append(Msg(
                                role="tool",
                                content=tr["content"],
                                tool_call_id=tr["tool_call_id"],
                            ))

                        # Ground the synthesis pass on tool results.
                        # Without this, stale history summaries (e.g.
                        # "assistant lacks live access") can override
                        # the actual retrieved data in the model's response.
                        current_msgs.append(Msg(
                            role="system",
                            content="You have retrieved live data using tools above. Answer the user's question directly and confidently using those results. Do not claim to lack live access.",
                        ))

                        # Continue to next iteration (will stream the follow-up)
                        # Yield paragraph break so post-tool text is
                        # separated from any prior content.  This flows
                        # into full_response_holder (persisted to DB)
                        # and the SSE stream.
                        yield "\n\n"
                        break
                else:
                    # Stream finished without a tool_calls dict
                    # (pure content response). TASK-216: tokens
                    # were already yielded inline above, so
                    # nothing left to flush.
                    return

            log.warning(f"Tool loop: max iterations ({max_iterations}) reached")

        return native_stream_with_tool_loop()
