# Mobile-only behavior: capture evidence on the actual device

Use when iPhone/Safari/PWA keyboard, scroll, focus, or layout behavior cannot be
reproduced on desktop. A narrow Playwright viewport is NOT a native keyboard
reproduction. Collect one bounded device trace before guessing at another fix.

## Existing touch-Send viewport trace

Source of truth in the BawtHub repo:
- `frontend/src/app/chat/viewport/submitViewportTrace.ts` — recorder and upload.
- `frontend/src/app/chat/viewport/useKeyboardInset.ts` — installs the recorder.
- `frontend/server/routes/remaining.ts` — `/api/clog` transport/ring buffer.

1. Confirm these files and the installer are present on the dev working surface.
   This is **dev-only** (`import.meta.env.DEV`), not enabled by a production URL.
2. Give the user a direct link to his affected chat with `viewportTrace=1` appended:
   `https://<dev-host>/chat/<bot>?viewportTrace=1` (example; preserve an existing
   query with `&viewportTrace=1`). Use the affected, permitted bot; never test a
   protected personal bot. No console, cable, developer mode, or broad logging
   switch is required on the phone.
3. Open that URL as a page load so the installer sees the flag. With the composer
   focused and keyboard open, tap Send once and leave the page foregrounded for
   about four seconds. **Send is real, not a dry run**; don't generate extra
   messages autonomously merely to test telemetry.
4. Read the dev `/api/clog` buffer immediately. Confirm the target-device UA and
   fresh capture timestamp, not just that some trace exists. If a useful capture
   arrived, tell the user you have it; don't ask for repetitive reproductions.
5. Return to the ordinary unflagged URL after diagnosis. The flag is not stored
   in preferences; a fresh unflagged page load installs no recorder.

### Capture contract and limits

- Trigger: `bawthub:touch-message-submit`, accepted only for the focused textarea.
  Desktop Enter or sending with an already-blurred composer is not equivalent.
- Sampling: rAF for ~3 seconds, 3.2-second fallback timer; unchanged geometry is
  omitted unless an event occurred. Max 240 samples and three captures per
  installer mount. Cleanup stops and uploads an active capture.
- Data: numeric/boolean/null viewport, shell, transcript, composer, natural-content,
  anchor and focus geometry, plus event labels. `ms` is elapsed from submission.
- Upload: POST `/api/clog`, entries prefixed `[submit-viewport]`, eight samples per
  part, common `traceId` and `ts` (capture-start epoch milliseconds). `part` starts
  at zero; `dropped` reports samples omitted by the cap. No idle recording.
- Privacy: no draft/message text, message IDs, or real chat URL in trace payloads.
  The URL is the literal `/chat (viewport trace)`. User agent is included; the
  server may also record forwarded IP. Treat logs as diagnostic data, not public
  artifacts. Do not enable global console forwarding as a shortcut.
- The chosen DOM anchor is followed by identity without exporting its ID.
  `anchorTop: null` means it was absent/unmounted, NOT that it moved to zero.
- Backgrounding/throttling, network failure, and missing DOM selectors can prevent
  a complete capture. Upload is best-effort; there is no durable client queue.

## Read and assemble the trace

Command from the bridge, with `HOST` set to the Traefik host's LAN address (host routing matters):

```bash
curl -fsS -H 'Host: <dev-host>' \
  "http://$HOST/api/clog?limit=1000" | python3 -c '
import json, sys
prefix = "[submit-viewport] "
groups = {}
for entry in json.load(sys.stdin)["entries"]:
    text = entry.get("text", "")
    if not text.startswith(prefix) or "iPhone" not in entry.get("ua", ""):
        continue
    try:
        part = json.loads(text[len(prefix):])
    except json.JSONDecodeError:
        print("Incomplete/truncated trace entry", entry.get("ts"), file=sys.stderr)
        continue
    group = groups.setdefault(part["traceId"], {
        "startedAt": entry["ts"], "ua": entry.get("ua"), "parts": {}
    })
    group["parts"][part["part"]] = part
fields = ("viewportHeight", "viewportTop", "windowY", "shellTop", "shellHeight",
          "bodyTop", "clientHeight", "scrollTop", "scrollHeight", "naturalHeight",
          "topPadding", "composerTop", "composerHeight", "anchorTop", "focused")
for trace_id, group in sorted(groups.items(), key=lambda item: item[1]["startedAt"]):
    parts = group["parts"]
    print("TRACE", trace_id, "startedAt", group["startedAt"], "parts", sorted(parts),
          "dropped", max(p.get("dropped", 0) for p in parts.values()), "UA", group["ua"])
    for number in sorted(parts):
        for sample in parts[number]["samples"]:
            print(sample["ms"], ",".join(sample["events"]),
                  json.dumps({key: sample["geometry"].get(key) for key in fields}))
'
```

Change the UA filter for the actual device (Android/iPad may differ). Group by
`traceId` and sort by `part`; never splice two submissions into one timeline.
Missing part zero/gaps mean an incomplete capture; contiguous parts alone don't
prove a full three seconds elapsed. Save only the relevant sanitized evidence
in the task receipt before it rolls out of the buffer.

`/api/clog` is a **shared in-memory 1,000-entry ring**, not durable storage.
GET defaults to 500, caps at 1,000; optional `since=<epoch-ms>` filters by server
`receivedAt`, not device `ts`. Dev and prod have separate buffers. Hono restarts
clear the ring; don't restart to obtain diagnostics or DELETE the shared logs.
Entries are truncated at 8,000 characters and also emitted to server console.

## Interpret before changing code

| Observed movement | Investigate |
|---|---|
| `viewportHeight` / `viewportTop` / `windowY` change | Native keyboard, browser panning, visual viewport events |
| Viewport settles but `shellTop` / `shellHeight` lag | Shell viewport synchronization |
| Shell steady, `composerHeight` changes | Textarea resizing, attachments, composer controls |
| Shell/composer steady, `naturalHeight` / `scrollHeight` grow then shrink | Transient transcript UI: queued status, banners, live-reply admission, row measurement |
| Content height steady but `scrollTop` / `anchorTop` shift | Follow/lock controller, native anchoring, focus-driven scroll |
| `topPadding` changes | Reserved nav space / keyboard inset reclamation |

Correlate multiple fields; one ResizeObserver event doesn't identify its source.
Trace geometry narrows the subsystem; identify/reproduce the actual component
before claiming causality. Keep `viewport/useChatScrollLock.ts` the sole scroll
writer; diagnostics must never set scrollTop, resize nodes, or change focus.

**Worked case — TASK-1097 (2026-10-16):** the iPhone keyboard and composer had
settled, but transcript height rose 48px, fell 48px, then grew for the live reply.
The ordinary queue status contributed a 44px Cancel button plus 4px margin.
A browser fixture reproduced the same +48/-48px flash. Delaying only queue
presentation 300ms avoided fast-admission flashes while retaining cancellation
for real queues. The user subsequently confirmed the fix on his iPhone. This is a
case study, NOT evidence that every mobile jitter is a queued-status bug.

## Adapt the pattern for other mobile-only symptoms

The existing recorder is **not a general JavaScript execution trace**. For another
interaction, add narrowly scoped dev-only instrumentation around its real event:
explicit URL opt-in, a short bounded capture, numeric geometry/sanitized state
labels, correlation ID and monotonic timing, batched POSTs to existing `/api/clog`,
and complete listener/observer/timer cleanup. Capture lifecycle labels when
geometry alone cannot identify what mounted. Never upload DOM text, credentials,
message contents/identities, or arbitrary console arguments. Don't persist an
always-on diagnostic preference or add another log service for a one-off probe.

Verify opt-out produces zero uploads and opt-in reaches the server. Test caps,
cleanup, and serialization; run client typecheck/lint. Use isolated browser
fixtures for regression coverage without sending real chat or mutating backend
history. Mock all relevant history/tool sources together—mixing synthetic rows
with live refreshes can manufacture unrelated jumps. Desktop simulation is useful
for the fix but not proof of native resolution; finish with device evidence or
the user's explicit confirmation.

If the link shows a blank page, inspect `[RootErrorBoundary]` entries and the UA
first; see [blank-screen triage](pitfalls.md). A failed lazy import is a separate
loading bug, not a viewport measurement. Never prescribe cache/site-data wipes,
reinstallation, incognito, or repeated refreshes as the fix.
