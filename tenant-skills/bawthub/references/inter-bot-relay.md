# Inter-bot relay feedback (TASK-1106 / TASK-1108 / TASK-1111)

Read before changing bot-message pills or bot-bar send/receive effects. Paths below are relative to `frontend/src/app/` in BawtHub.

## One event source, two views

`chat/useUnifiedEventStream.ts` routes delivery events through `_liveHandlers.interBotDelivery` to `chat/unifiedInterBotDeliveryHandler.ts::handleInterBotDelivery`. That handler drives BOTH:

- The notification pill / event-log receipt through `notify` and `notifications/useNotificationStore.ts`.
- The avatar effect through `notifications/useBotRelayPulses.ts`; `notifications/BotRelayPulse.tsx` renders it inside `(app)/FloatingBotList.tsx` next to `BotPickerAvatar`.

Do not add another SSE subscription, infer sends from tool text, or trigger the effect from component mounts. Ordinary-chat deliveries take a separate early-return branch: inline queue badges and failure handling, **not** bot relay pills or avatar pulses.

## Lifecycle and deduplication

| Backend status | Relay phase | Meaning / live pill lifetime |
|---|---|---|
| `QUEUED` | `queued` | Waiting; `RELAY_QUEUED_MS` = 12s, visible countdown |
| `STEERING` | `sending` | Waiting for the active turn's next tool boundary; `RELAY_IN_FLIGHT_MS` = 120s |
| `DISPATCHING` | `handoff` for async; `working` for waited | Separate target turn accepted. Async sender's exchange ends after a 4s blue send/"Handed off" confirmation, not a green delivery check; waited sender's graphic holds up to 120s for the reply. |
| `DELIVERED` | `sent` | Steer accepted OR separate turn completed. For async dedicated turns already shown as handed off, log the completion but do not replay the live pill; waited turns transition to success (4s). |
| `FAILED` | `failed` | Automatic retries stopped; sticky (`durationMs: null`) |
| `CANCELLED` | `cancelled` | Cancelled; warning default = 8s |

`DELIVERED` is **not universally a completion receipt**: check `delivery_mode`. A separate turn can take minutes; its receive effect belongs at `DISPATCHING`, not at completion. The server-owned `reply_mode` rides the delivery event: `async` ends the sender's live exchange at `DISPATCHING`; `waited` keeps it open until `DELIVERED`. Events from older producers without `reply_mode` retain the previous in-flight behavior rather than guessing that a waited send was async.

The live pill uses `inter-bot-delivery:${delivery_id}` with `replace: true`. Queued pills expire instead of occupying the screen indefinitely; a subsequent transition brings the pill back. An async dedicated-turn handoff uses a short 4s info lifetime rather than tracking the target's ongoing work; its later completion stays in notification history without resurrecting the pill. Failures still surface visibly. Notification history retains receipts separately from the live stack.

The handler remembers up to 200 delivery states, comparing `[status, attempt_count, last_error, delivery_mode]`. Retry timestamps alone do not count as changes. Identical requeue heartbeats do not repeat feedback. `deliveryLifecycleStatus()` exposes the last remembered status; this is bounded client memory, not durable delivery history.

### Avatar pulse rules

`pulseBotBar(event, previousStatus)` requires both party IDs:

- **Send:** first lifecycle event observed for that delivery (`previousStatus === null`). This is an observation signal, not proof of successful delivery.
- **Receive:** changed status is `DISPATCHING`, or is `DELIVERED` and the immediately previous remembered status was not `DISPATCHING` (steer path).
- Same-status attempt/error updates do not replay arrival. Failed/cancelled events never trigger receive.

`useBotRelayPulses` keeps one latest `{kind, peer, id}` per bot slug. A monotonic ID keys the DOM effect so a new pulse replays it. **The store owns expiry** (`BOT_RELAY_PULSE_MS = 3400`), not the component: a remounted rail must not replay expired feedback. Timeout callbacks delete only their own ID, never a newer pulse. `reset()` clears slots without recycling IDs.

## Presentation and shared identity

- `notifications/InterBotRelay.tsx::RelayLane`: `[sender BotChip] ››› [target BotChip + status badge]`. `BotChip.avatarAdornment` keeps the badge attached to the avatar; `avatarVariant="rail"` shares the homepage's dark embossed avatar bubble. `compact` is the static event-log receipt; `floating` uses the shared glass surface.
- `useRelayParty()` resolves names, avatars and palettes through existing bot hooks. **Reuse it** for both views; do not duplicate bot lookup/color logic. Use `palette.accentColor` (light-on-dark), not potentially near-black `hex`.
- `NotificationToast.tsx` renders relay-specific captions and dismiss controls; `NotificationItem.tsx` renders the compact receipt. TASK-1111 adds `RelayDeliveryControls.tsx` to the event-log row (not the mobile ticker): one plain-language status, failure reason/attempt/next retry, and 44px actions. Queued offers Cancel; failed normal callbacks offer Retry; all offer Open chat. Old receipts retain history, but only the latest event for a delivery has active actions, and a retry receipt disables the parent retry action.
- The Retry button POSTs the exact failed delivery ID to `/api/chat/proxy/v1/inter-bot-deliveries/{delivery_id}/retry`. llm-bawt creates a **new** durable row under a deterministic `manual-retry:{old_id}` idempotency key; a repeat click/ambiguous HTTP response never starts another turn. It rejects in-flight/successful, ordinary-chat, prompt/maintenance, and session-reset rows. The original failure remains in the log; the new delivery has `retry_of_delivery_id` in its SSE lifecycle. Cancel targets the existing queued-only endpoint. Do not publish raw Redis bridge commands from the UI or add a second event listener; REST confirms the action, existing SSE drives lifecycle.
- Retry may re-run side-effecting tools if the old failed attempt partly executed. The event log warns explicitly; no automatic retry is added beyond the existing dispatcher policy. Missing/invalid REST receipts never claim success; 409 stays visible as an actionable error.
- `NotificationHost.tsx` overlays the **top navigation bar** at every breakpoint, using measured `--app-nav-height`; opening the log drops it below the nav. Rail clearance is applied **symmetrically** (`max(left, right)`) so the centred group sits on the true centre. Do not place the ticker over chat or introduce transformed ancestors around backdrop-filter surfaces.
- `NotificationTicker.tsx` shows live notices side by side, oldest → newest, centred as one group (a newer notice adds on the right; an updated notice keeps its slot). Only when the lane runs out of width do the oldest fold; an overlaid `+N` badge on the first visible pill shows the count without occupying row width or shifting the centred pill. That pill opens the log. Fit is measured, never breakpoint-coded: a layout effect reads each shown item's natural width, the CSS gap and the lane width, and `tickerFit.ts::foldedTickerCount` (pure, unit-tested) decides how many fold; a `ResizeObserver` re-fits on lane resize. Items must therefore keep their natural width (`shrink-0`, capped `max-w-full`) — a growing (`flex-1`) or shrinking item corrupts the measurement. Live notices are capped at `MAX_STACK` (5) in the store.
- Send avatar effect: outward rings in the sender's own color plus a three-dot CSS speech bubble. Receive: inward ring and inner glow in the **sender's** color. `BotRelayPulse` is decorative (`aria-hidden`, no pointer events); the notification host supplies the live announcement.

## CSS and geometry constraints

All motion lives in `globals.css`: `.relay`, `.relay-arrows`, `.relay-ring`, `.relay-badge`, `.relay-caption`, `.relay-floating`; `.bot-pulse`, `.bot-pulse-ring`, `.bot-pulse-bubble` and `[data-kind]`.

**Static visible frames are the base.** Animations are layered inside `prefers-reduced-motion: no-preference`; reduced-motion users still see feedback until store expiry. Keep the expiry at least as long as the longest delayed animation.

Current avatar timing: send rings 1300ms × 2 with a 650ms delayed second ring; bubble 3200ms; typing dots 700ms × 3. Receive ring 1100ms × 2, glow 1600ms after a 1500ms delay. Total slot lifetime is 3400ms. Shorter initial timing was too easy to miss.

The rail's scrollport is **44px wide**, with 6px right padding around a 32px avatar, and clips overflow. Keep effects within roughly 5px of the avatar; do not assume overflow-visible on a child defeats the scrollport. Corner ownership: pin/unread top-left, activity pip bottom-right, speech bubble top-right (`top: -5px; right: -3px`). On the 4px mobile gap the bubble overlaps the status pip of the bot above; it must paint over it, so `.bot-pulse` has no z-index (no stacking context), the ring sits at z-15 and the bubble at z-30, above every badge (z-20). At this size a generic speech icon reads as a circle/dot; use the existing CSS bubble and typing dots.

## Verification and extension rules

- Handler regression suite: `chat/unifiedInterBotDeliveryHandler.test.ts`. Reset both `resetInterBotDeliveryTracking()` and `useBotRelayPulses.getState().reset()` between scenarios. Cover queued → dispatching → delivered, steering → delivered, duplicate heartbeats, same-status attempt changes, ordinary-chat exclusion, failed/cancelled, and missing party IDs.
- Run client typecheck and targeted tests from `frontend/` using the normal host workflow:
  ```sh
  npx tsc --noEmit
  node --experimental-test-module-mocks --import tsx --test src/app/chat/unifiedInterBotDeliveryHandler.test.ts
  ```
- Dev-only visual fixture: find the **loaded HMR-timestamped module URL** for `useBotRelayPulses.ts` in `performance.getEntriesByType('resource')`, then dynamically import that exact URL and call the store's `pulse(slug, kind, peer)`. Importing a guessed bare URL can create a different store instance after HMR. Use real non-personal bot slugs already visible in the rail; never send real messages just to manufacture a screenshot.
- To inspect a frame, pause the relevant animations from `document.getAnimations()` and set their `currentTime`; this does **not** pause the 3400ms store expiry. Inspect promptly or use a temporary dev-only store fixture, then reset it. Do not ship fixtures/debug triggers.
- Check actual elapsed visibility and clipping, both rail edges, mobile and desktop widths, reduced motion, rapid overlapping pulses, and route/collapse remounts. Browser emulation is not physical-device evidence. A synthetic store pulse verifies presentation, not live SSE delivery.
