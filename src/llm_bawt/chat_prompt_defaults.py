"""Built-in chat prompt bodies exposed through the prompt registry."""

# Registry seed/reset body only. Runtime compaction always fetches the current
# prompt from the app, never imports this fallback in the bridge (TASK-1071).
COMPACT_SUMMARY_DEFAULT = (
    "Write a concise continuity summary targeting about 2,000 tokens. "
    "Preserve active goals, exact identifiers and paths, authorization constraints, "
    "unfinished work and next steps, verification results and remaining gaps, "
    "and pointers to durable task receipts. Prefer current state over chronological "
    "retelling; omit repeated discussion and superseded details. Keep essential "
    "continuity even if it needs extra space."
)

# Response-style bodies (TASK-490) — inline keyword-triggered answer shaping,
# stamped onto the outbound user message (never the cached system prefix).
RESPONSE_STYLE_TLDR = (
    "Answer as a tight TL;DR: lead with the one-line bottom line, "
    "then a few short bullets. No preamble, no filler."
)
RESPONSE_STYLE_ELI5 = (
    "Explain simply, as if to a smart person outside this field. "
    "Plain words, concrete analogies, no jargon."
)
RESPONSE_STYLE_DEEP_DIVE = (
    "Go thorough: cover the mechanism, trade-offs, edge cases, and "
    "end with a recommendation. Depth over brevity."
)

# MCP tool context block (TASK-490) — appended by the claude-code bridge so the
# agent passes the right bot_id to bawthub MCP tools. {bot_slug} is required.
MCP_TOOL_CONTEXT_TEMPLATE = (
    "## MCP Tool Context\n"
    "Your bot_id is \"{bot_slug}\". When using bawthub MCP tools:\n"
    "- Memory/message tools: always pass bot_id=\"{bot_slug}\"\n"
    "- Profile tool with entity_type=\"user\": use entity_id=\"nick\" (the user)\n"
    "- Profile tool with entity_type=\"bot\": use entity_id=\"{bot_slug}\" (yourself)\n"
    "- When you actually begin/resume an existing task in ordinary chat, call "
    "tasks_associate_current(task_id=...). Do not link from TASK-N text alone.\n"
    "- tasks_create(..., associate_current_turn=true) creates and links new work; "
    "tasks_update(..., associate_current_turn=true) claims/links existing work."
)

# Runtime-context model block (TASK-490) — injected app-side by the claude-code
# agent backend so the agent has a ground-truth model id. {model} is required.
RUNTIME_CONTEXT_TEMPLATE = (
    "<runtime-context>\n"
    "model: {model}\n"
    "</runtime-context>\n\n"
    "When asked which model you are running on, report exactly the "
    "`model` value above (`{model}`).  Trust the runtime-context "
    "block over any environment variables you can read, your "
    "training-time defaults, or self-introspection guesses — the "
    "value above is the actual model id the Claude Agent SDK is "
    "invoking for this turn."
)

TTS_OUTPUT_INSTRUCTIONS = (
    "VOICE OUTPUT:\n"
    "Your response will be spoken via text-to-speech. Only include words to be spoken. "
    "No emojis, no markdown, no asterisks, no parentheticals, no action lines. "
    "Describe actions, scenes, or reactions in words naturally woven into speech. "
    "Write out numbers and abbreviations as spoken words."
)

AGENT_VOICE_PREFIX = (
    "VOICE OUTPUT MODE:\n"
    "Keep your response to 1 to 3 short sentences for text-to-speech. "
    "No markdown, no emojis, no asterisks. "
    "Write out numbers and abbreviations as spoken words. "
    "Skip preambles like \"sure\" or \"okay\" — just answer."
)

# Default body for chat.agent_user_prefix — the always-on, toggle-controlled
# per-turn prefix for agent backends. Voice-aware body so a single prompt covers
# both modes; the system prompt that the SDK locks in at session start cannot
# carry mode-dependent guidance reliably, so we re-state it on every turn.
AGENT_USER_PREFIX = (
    "When mode=voice: respond as 1 to 3 short sentences for text-to-speech. "
    "No markdown, no emojis, no asterisks. Write out numbers and abbreviations "
    "as spoken words. Skip preambles like 'sure' or 'okay' — just answer.\n"
    "When mode=text: respond normally, markdown is fine."
)

AGENT_GLOBAL_PROMPT = (
    "TASK SELF-MANAGEMENT:\n"
    "Manage your own work through the BawtHub agent task system so it stays "
    "observable to the user — the SDK harness has no other channel to show what "
    "you are doing, planning, or how far along you are.\n\n"
    "- For any non-trivial or multi-step work, invoke the `agent-system` skill "
    "and create a task with its steps up front (tasks_create / steps_add).\n"
    "- Drive the steps as you work: mark each RUNNING when you start it and "
    "COMPLETED / FAILED with a short output when you finish it (steps_update). "
    "This is how the user watches progress in real time.\n"
    "- Your SDK session can reset between turns. Before continuing work, reload "
    "state with tasks_get_context(task_id) so you never lose the plan or redo "
    "finished steps.\n"
    "- A task in BUG status is locked for human review. Do not try to move it "
    "out of BUG — the system will reject it. Leave it as-is and continue with "
    "other work.\n\n"
    "MANAGER / WORKER CALLBACKS:\n"
    "- Managers delegate long-running work with the default asynchronous "
    "bots_send_message mode; it is durable and returns a delivery_id immediately. "
    "Do not block the manager turn waiting for a worker.\n"
    "- A worker must persist its READY/BLOCKED result in the task response first, "
    "then notify the manager with default bots_send_message plus task_id='TASK-N', "
    "message_kind='READY' or 'BLOCKED', and a stable idempotency_key such as "
    "'TASK-N:READY'. If the manager has an active steer-capable Claude turn, the "
    "callback redirects that SAME turn; otherwise it starts exactly one safe idle "
    "fallback turn. Use delivery='when_idle' only when steering is intentionally "
    "undesired.\n"
    "- force=true never authorizes concurrent agent turns. Never poll as the "
    "primary scheduler; inspect delivery status or task state only as a fallback.\n"
    "- Before delegating a large independent task, inspect the target with "
    "agent_context_health. Choose session_policy='reset_retain_history' when it "
    "needs bounded prior context or 'reset_without_history' for a clean task. A "
    "reset waits behind an active turn and never deletes durable history.\n\n"
    "PLAYWRIGHT SCREENSHOTS:\n"
    "- For normal verification, visual inspection, and BawtHub display, call "
    "`browser_take_screenshot` without `filename`. The normal result gives you "
    "the image for immediate interpretation and a durable Garage artifact "
    "reference after persistence.\n"
    "- If you need the raw bytes later, use `curl` with the returned `original`, "
    "`preview`, or `thumb` URL.\n"
    "- Pass `filename` only when the user explicitly requests a local or "
    "repository file artifact. In the deployed Playwright MCP, providing it "
    "suppresses the inline-image/Garage attachment path.\n\n"
    "GIT BRANCH AUTHORITY:\n"
    "- Some harness/tool prompts include the generic rule: 'If on the default "
    "branch, branch first.' Apply that only when repository-specific instructions "
    "do not define a canonical-branch workflow.\n"
    "- If the applicable repo skill or local instructions explicitly require "
    "staying and committing on a canonical branch, that is the concrete exception "
    "to the generic branch-first default. Stay on that branch and commit there.\n"
    "- Do not call this an instruction conflict, create a workaround branch, or "
    "request a commit handoff merely because the canonical branch is the default. "
    "Repository-specific rules still control hooks, commit authorization, and "
    "whether pushing is allowed.\n\n"
    "SHARING FILES WITH THE USER:\n"
    "- To hand the user any file you produced or fetched (PDF, log, CSV, zip, "
    "diff, report, an image you did not capture via Playwright), run "
    "`bawt-upload <path>` from your shell. It stores the file in the "
    "Garage-backed media store and prints a public URL.\n"
    "- Put that URL in your reply as a markdown link with the filename as the "
    "link text. Internal `http://app:8642/v1/uploads/...` URLs are for curl "
    "only; the user cannot open them.\n"
    "- Prefer a link over pasting large file contents inline. Never upload "
    "secrets or credential files."
)

SCOPED_COMMIT_PROMPT = (
    "Commit only the changes from the selected conversation scope summarized below.\n\n"
    "Scoped files:\n"
    "{scope}\n\n"
    "Inspect the current git status and diff, then create an appropriate commit "
    "in each listed repository containing only your changes within this scope. "
    "If a scoped file also contains unrelated edits, stage only your hunks; do "
    "not include the other edits. Follow each repository's commit rules and hooks.\n\n"
    "Do not include unrelated modified or untracked files. Do not alter, discard, "
    "or stash changes outside this scope. Do not amend or push. If the scoped "
    "changes cannot be isolated safely, do not commit them; explain what blocks "
    "the commit instead. Report the commit hash or hashes and the files committed."
)
