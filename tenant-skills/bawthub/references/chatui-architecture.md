# ChatUI Architecture Reference

Detailed map of `frontend/src/app/chat/` after the TASK-193 decomposition
(commits `5f9c204` → `76b831d`, June 2026). Read this before any chat-module
work. The in-repo source of truth is `frontend/src/app/chat/ARCHITECTURE.md` —
if the two disagree, trust the repo and update this file.

## The core idea

`ChatUI.tsx` was a 3,061-line monolith; it is now a **~400-effective-line pure
orchestrator**. It instantiates hooks, threads their returns into child props,
and renders layout. It contains **no business logic**: no `useEffect` with
logic, no `fetch`, no SSE handlers. Every concern lives in a focused module
under a concern directory.

This is enforced, not aspirational: ESLint `max-lines` (error at 500,
`skipBlankLines`/`skipComments`) applies to every file in `src/app/chat/**`.
ChatUI.tsx has **no exemption** — if your change grows it past the budget,
lint fails the build. Legacy files not yet decomposed carry `TODO(TASK-193)`
exemptions in `eslint.config.mjs`; **new files never get one**. If you hit
500 lines, split by concern.

## Decision table — where does new code go?

| You are adding… | It goes in… |
|---|---|
| History loading, pagination, payload normalization, turn-log→message mapping | `history/` |
| Pending agent turns, `turn_complete` SSE, abort/stop behavior | `turns/` |
| Tool-call activity fetching, merging, dedup, per-message activity derivation | `activity/` |
| HTTP stream lifecycle, stream→store projection, resume polling, send path | `stream/` |
| Scroll, keyboard inset, viewport/touch behavior | `viewport/` |
| A new visual element (bubble, card, popup, button) | A new `.tsx` component at chat root (≤500 lines) |
| A cross-cutting concern that fits none of the above | A NEW directory with its own `use*.ts` hook — never ChatUI.tsx |

File kinds: pure logic → `.ts` module · stateful logic → `use*.ts` hook ·
visual chrome → `.tsx` component. Don't mix kinds in one file.

## Module map

```
frontend/src/app/chat/
├── ARCHITECTURE.md            # in-repo decision table + rules (source of truth)
├── ChatUI.tsx                 # orchestrator only (~400 effective lines, hard lint cap)
│
├── history/                   # loading & pagination of message history
│   ├── useChatHistory.ts      # THE history hook: loadRecent/Older/Newer/AroundMessage
│   │                          #   + scope-change effect + anchor-landing state
│   ├── useDeleteMessage.ts    # delete-message handler
│   ├── normalize.ts           # payload/message/attachment normalization (pure)
│   ├── turnLogMessages.ts     # turn-log → message mapping + response LRU cache (pure)
│   ├── optimistic.ts          # optimistic-ID fingerprint reconciliation (pure)
│   ├── resumeScan.ts          # resume-detection scan over loaded history
│   ├── turnLogBackfill.ts     # turn-log backfill for gaps
│   ├── constants.ts           # page sizes etc.
│   ├── AnchorLandingCurtain.tsx, HistoryPrimingSpinner.tsx
│   └── normalize.test.ts      # 16 characterization tests — run before/after edits
│
├── turns/                     # pending agent-turn tracking + turn_complete + abort
│   ├── usePendingAgentTurns.ts        # ref-based Map<botId, Map<turnId, userMsgId>>
│   │                                  #   + useSyncExternalStore snapshot
│   ├── useTurnComplete.ts             # turn_complete SSE listener (useEffectEvent)
│   ├── useAbortTurns.ts               # abort single/all turns, orphaned-turn query
│   └── useReconcileOptimisticActivity.ts
│
├── activity/                  # tool-call activity (fetch, merge, derive)
│   ├── useToolActivityFetch.ts        # fetchToolActivity + catchUpLiveToolActivity
│   ├── useActivityForMessage.ts       # identity-stable per-message activity memo
│   ├── useOrphanedActivityReKey.ts    # turn-id → message-id re-keying sweep
│   ├── agentActivity.ts               # activity parsing (pure, has parser test)
│   └── toolCallDerivation.ts          # tool-call derivation (pure, has test)
│
├── stream/                    # HTTP stream lifecycle and projection
│   ├── ChatStreamContext.tsx          # stream provider: SSE parse, per-stream state
│   ├── useChatStreamSync.ts           # ChatStreamContext → Zustand store projection
│   │                                  #   (optimistic upsert, assistant commit, errors)
│   ├── useSendMessage.ts              # handleSend: optimistic insert, upload-first
│   │                                  #   image flow, startStream delegation
│   └── useResumeController.ts         # resume polling for interrupted turns
│
├── viewport/                  # scroll + keyboard + device
│   ├── useChatScrollLock.ts (+test)   # scroll lock / follow state machine
│   ├── useAutoFollow.ts               # ResizeObserver grow-only follow-to-bottom
│   ├── useKeyboardInset.ts            # visualViewport tracking, nav-gap reclaim
│   ├── useTouchDevice.ts              # pointer:coarse media query
│   └── ScrollToBottomButton.tsx
│
└── (root)                     # shared components & small hooks
    ├── MessageList.tsx, Composer.tsx, AgentActivityRow.tsx,
    │   ClaudeToolCallCard.tsx, InFlightBubble.tsx, …   # visual components
    ├── useChatStore.ts        # Zustand store (per-bot messages/activity/flags)
    ├── useChatBotState.ts     # per-bot store selectors for ChatUI
    ├── useMirrorRef.ts        # generic value→ref mirror (replaces 7 ad-hoc pairs)
    ├── useDismissComposerActions.ts   # click-outside/Escape for composer popups
    └── useImageAttachments.ts, useChatUiStore.ts, …
```

Legacy files still over budget (have `TODO(TASK-193)` exemptions):
`AgentActivityRow.tsx`, `ClaudeToolCallCard.tsx`, `MessageList.tsx`,
`useChatStore.ts`. Shrink-only — don't grow them.

## Development patterns (the new idioms)

**1. Hooks own their state; ChatUI threads parameters.**
Each hook declares an `Args` interface (inputs from other hooks/ChatUI) and
returns an `Api` object (e.g. `useChatHistory(args): ChatHistoryApi`).
Cross-cluster needs are passed as explicit parameters — never via shared
mutable refs declared at ChatUI scope. If hook B needs something hook A owns,
ChatUI passes A's return into B's args.

**2. Ref mirrors via `useMirrorRef`.**
When an effect/callback needs the latest value without re-subscribing, use
`const xRef = useMirrorRef(x)` instead of hand-rolling `useRef` + `useEffect`
sync pairs.

**3. Ref-based maps must bump.**
Pending agent turns live in a ref `Map` (no re-render on mutation). Every
`.set/.delete/.clear` MUST be followed by `bumpPendingTurns()` — that's what
notifies the `useSyncExternalStore` snapshot. Forgetting the bump = stale UI
with no error.

**4. `useEffectEvent` for SSE/async handlers.**
The `turn_complete` handler and activity re-keying use `useEffectEvent` to
read fresh messages/activity without stale closures and without effect
re-subscription. When moving or extending these, keep the wrapper.

**5. Identity-stable memos are load-bearing.**
`activity/useActivityForMessage` reuses array identities for unchanged
buckets — this prevents AgentActivityRow re-mount flash during streaming.
Same for the store's message-array reuse. Do not "simplify" these into fresh
arrays per render.

**6. Characterization tests before moving pure logic.**
`history/normalize.test.ts` exists to lock behavior during extraction. If you
refactor pure helpers, run (and extend) the tests first.

**7. No behavior piggy-backing.**
Structural refactors land separately from behavior changes. A bug found
mid-refactor gets its own commit/task.

## Data flow (one turn, end to end)

```
Composer → useSendMessage.handleSend
  ├─ optimistic user message → useChatStore (id: local-user-<ts>)
  ├─ image uploads first (if any), then startStream → ChatStreamContext
ChatStreamContext (SSE parse, per-stream state incl. warnings[])
  → useChatStreamSync projects into useChatStore:
      streaming flags, assistant commit on terminal status,
      error/warning surfacing (stream_interrupted renders a bubble — don't drop)
Agent turns: turn ids tracked in usePendingAgentTurns
  → turn_complete SSE → useTurnComplete resolves turn → message
  → useReconcileOptimisticActivity / useOrphanedActivityReKey re-key
      activity from optimistic/turn ids to server message ids
Tool activity: useToolActivityFetch (history fetch + live catch-up)
  → useActivityForMessage merges store + pending per message (identity-stable)
History: useChatHistory loads recent on scope change; Load-older button
  paginates with scroll-position preservation; ?message=<id> deep link loads
  around the anchor and lands via AnchorLandingCurtain
```

## Task ↔ conversation binding (where it lives, and where it doesn't)

The task-conversation architecture (TASK-700–706) is deliberately
server-owned. **The chat module does NOT own task-binding decisions.**

- **Ordinary-chat trusted association** — happens in llm-bawt
  (`chat_streaming.py` mint site) and the MCP server. The chat client
  sends `/api/chat` normally; when an agent calls
  `tasks_associate_current` (or the `associate_current_turn=true` flag on
  `tasks_update` / `tasks_create`), the association write happens
  server-side via a bearer-authenticated internal listener. There is no
  chat-UI concern here — do not add per-message task-association state
  to `useChatStore` or a hook under `chat/`.
- **User-driven attach / link / unlink / set-primary** — lives on the
  **Task detail** view (`(app)/(dashboard)/tasks/…`), calling
  `/api/tasks/tasks/:ref/conversation-context*` routes. Not in `chat/`.
- **Task-bound chat continuation** — opening a task's conversation from
  the Task detail view navigates to the chat page with the linked
  session preselected. The chat module treats this as an ordinary
  explicit-session load; it does not need to know a task drove it.
- **`bawthub/frontend/src/app/chat/ARCHITECTURE.md` "Externally initiated
  turns" section** — this DOES apply to the chat module and already
  covers the TASK-709 session-scoping rules for events that arrive on
  the bot-scoped SSE. Keep those rules aligned when session-scoping
  changes.

**Canonical cross-system reference:**
[`agent-skills/agent-system/task-conversation-architecture.md`](../../agent-system/task-conversation-architecture.md)
— cross-DB ER diagram, five sequence diagrams, 16 verified invariants,
and the deferred ledger (including the mid-turn thread-switch gap and
split-view same-bot hydrate limits, which are chat-module concerns).

## Validation checklist for chat changes

1. `cd frontend && npx tsc --noEmit` — clean.
2. Tests: `npx -y tsx --test src/app/chat/multiBot.test.ts src/app/chat/useChatStore.dedupe.test.ts src/app/chat/history/normalize.test.ts src/app/chat/activity/agentActivity.parser.test.ts src/app/chat/activity/toolCallDerivation.test.ts src/app/chat/viewport/useChatScrollLock.test.ts` (53 tests).
3. `npx eslint src/app/chat` — 0 NEW problems. Baseline (June 2026): 10
   pre-existing set-state-in-effect errors + ~8 warnings in legacy files.
4. Manual smoke on `dev.bawthub.com` for timing-sensitive paths:
   load-older scroll preservation, deep-link `?message=` landing, send/stream
   round-trip, abort mid-turn.
5. Once reviewed, pushed work is ready, bots file the enabled `bawthub.release-prod` ops card (see `prod-release.md`); a different bot whose tags grant `approve_ops`, or Nick, approves. Never substitute local Make rebuild.

## Chat hot facts (moved from SKILL.md, TASK-1640)

- Zustand store per bot (`useChatStore`); HTTP streams via shared `ChatStreamContext`; SSE (`turn_complete`, `tool_activity`) via one unified event stream.
- Pending agent turns live in a ref-based Map — **call `bumpPendingTurns()` after every `.set()`/`.delete()`/`.clear()`** (ref mutations don't re-render).
- Agent bots = `!!agent_backend && agent_backend !== "default"` (multi-turn, SSE activity); non-agent = single-turn HTTP stream.
- The unified composer popup duplicates compact (mobile) vs desktop code — apply changes to BOTH.
