"""Claude Code bridge — translates Agent SDK events to the shared AgentEvent protocol.

Reads chat.send commands from the shared Redis ``agent:commands`` stream,
runs them through the Claude Agent SDK (which spawns the Claude Code binary),
and publishes AgentEvent-compatible events back to Redis run streams.

The main app sees identical events regardless of which agent bridge handled
the request.
"""
