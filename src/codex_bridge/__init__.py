"""Codex bridge — translates OpenAI Codex SDK events to the shared AgentEvent protocol.

Reads chat.send commands from the shared Redis ``agent:commands`` stream,
runs them through the OpenAI Codex SDK (which spawns the codex Rust binary),
and publishes AgentEvent-compatible events back to Redis run streams.

The main app sees identical events regardless of which agent bridge handled
the request.
"""
