"""Native chat-tool streaming, separate from agent transport and SSE coordination.

Owns tool schema/executor setup and the text/tool/result continuation loop. The
worker retains queue consumption, bridge dispatch, cancellation and finalization.
"""
from __future__ import annotations

import time
import uuid

from .logging import get_service_logger
from .turn_stream_context import TurnStreamContext

log = get_service_logger(__name__)

# An elapsed-time budget stops new tool work; it cannot safely undo an
# already-running tool. The final text-only synthesis is a separate last pass.
TOOL_LOOP_BUDGET_SECONDS = 60.0
HA_TOOL_LOOP_BUDGET_SECONDS = 20.0


class NativeToolStream:
    """Per-turn native-tool iteration using the worker's shared context."""

    def __init__(self, ctx: TurnStreamContext):
        self.ctx = ctx

    def stream(self, messages, gen_kwargs):
        # Preserve eager tool setup and lazy provider execution from the worker.
        ctx = self.ctx
        llm_bawt = ctx.llm_bawt
        cancel_event = ctx.cancel_event
        turn_log_id, trigger_message_id = ctx.turn_log_id, ctx.trigger_message_id
        tool_context_holder = ctx.tool_context_holder
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
            trigger_message_id=trigger_message_id,
            turn_id=turn_log_id,
        )

        def native_stream_with_tool_loop():
            """Stream with native tool support, handling tool calls inline."""
            import json as _json
            from ..tools.parser import (
                ToolCall, format_tool_result, strip_tool_result_tags, tool_result_is_error,
            )

            current_msgs = list(messages)
            max_iterations = 3 if llm_bawt._ha_mode else 5
            budget = HA_TOOL_LOOP_BUDGET_SECONDS if llm_bawt._ha_mode else TOOL_LOOP_BUDGET_SECONDS
            deadline = time.monotonic() + budget
            limit_reason = None

            for iteration in range(max_iterations + 1):
                if cancel_event.is_set():
                    return
                if iteration >= max_iterations:
                    limit_reason = "tool-round limit"
                elif time.monotonic() >= deadline:
                    limit_reason = "tool time budget"
                final_pass = limit_reason is not None
                if final_pass:
                    log.warning(f"Tool loop: {limit_reason} reached; final text synthesis")
                    current_msgs.append(Msg(
                        role="system",
                        content=(
                            f"The {limit_reason} has been reached. No more tools may run. "
                            "Give a final answer using only the results already available. "
                            "Clearly state any unfinished work; do not promise another tool call "
                            "or claim an action succeeded without a successful result."
                        ),
                    ))
                # Keep the schema even for final synthesis: xAI can ignore
                # function_call_output when its schema disappears. Keeping it
                # does not require forcing text after every successful round.
                current_tools = tools_schema
                current_tool_choice = "none" if final_pass else "auto"

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

                emitted_text = False
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
                        emitted_text = emitted_text or bool(item.strip())
                        yield item
                    elif isinstance(item, dict) and "tool_calls" in item:
                        tool_calls = item["tool_calls"]

                        if not tool_calls:
                            # Empty tool_calls list — treat as pure text response.
                            # Text was already streamed above; nothing more to flush.
                            if final_pass and not emitted_text:
                                yield "I reached the tool limit before completing the request."
                            return
                        if final_pass:
                            # A provider ignoring tool_choice must never run an
                            # extra tool after either limit.
                            if not emitted_text:
                                yield "I reached the tool limit before completing the request."
                            return

                        # TASK-215: trigger_animation interception removed.
                        # All tool calls in this stream are real.
                        real_tool_calls = tool_calls

                        # Emit tool calls as OpenAI delta.tool_calls, then execute
                        tool_results = []
                        for tc in real_tool_calls:
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

                            log.debug(f"🔧 {name}({args})")
                            # Same item contract as agent bridges: the
                            # ToolEventInterceptor flushes preceding text, stamps
                            # text_offset, persists and publishes tool_start, and
                            # the SSE layer emits delta.tool_calls.
                            yield {
                                "event": "tool_call",
                                "name": name,
                                "arguments": args,
                                "tool_use_id": call_id,
                                "trigger_message_id": trigger_message_id,
                            }

                            tool_call_obj = ToolCall(name=name, arguments=args, raw_text="")
                            if time.monotonic() >= deadline:
                                limit_reason = "tool time budget"
                                raw_result = format_tool_result(
                                    name, None, error=(
                                        "Not executed: the tool time budget expired "
                                        "before this call could start."
                                    ),
                                )
                            else:
                                raw_result = executor.execute(tool_call_obj)
                            # Read the status before stripping its wrapper tag.
                            is_error = tool_result_is_error(raw_result)
                            # Strip <tool_result> XML tags — native tool path
                            # sends results as structured function_call_output
                            # items, not inline text.
                            result = strip_tool_result_tags(raw_result)
                            tool_results.append({
                                "tool_call_id": tc.get("id", ""),
                                "content": result,
                            })
                            yield {
                                "event": "tool_result",
                                "name": name,
                                "arguments": args,
                                "result": result or "",
                                "is_error": is_error,
                                "tool_use_id": call_id,
                                "trigger_message_id": trigger_message_id,
                            }

                        # Retain every round for the persisted tool-results history.
                        round_context = "\n\n".join(
                            f"[{tc['function']['name']}]\n{tr['content']}"
                            for tc, tr in zip(real_tool_calls, tool_results)
                        )
                        tool_context_holder[0] = "\n\n".join(
                            part for part in (tool_context_holder[0], round_context) if part
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

                        # Ground the next decision on results without forcing
                        # an answer before dependent tool work is complete.
                        current_msgs.append(Msg(
                            role="system",
                            content=(
                                "Use the tool results above. Call additional tools if needed "
                                "to complete the request, or answer when you have enough "
                                "information. Do not claim that an action succeeded unless "
                                "its tool result confirms success."
                            ),
                        ))

                        # Continue to the next iteration. The interceptor adds
                        # the counted tool→text paragraph break.
                        break
                else:
                    # Stream finished without a tool_calls dict
                    # (pure content response). TASK-216: tokens
                    # were already yielded inline above, so
                    # nothing left to flush.
                    if final_pass and not emitted_text:
                        yield "I reached the tool limit before completing the request."
                    return

                if final_pass:
                    return

        return native_stream_with_tool_loop()
