# Ops, media and search

## Operations

Before service actions read `llm-bawt/reference/ops-actions.md` and discover the
live catalog with ops_list_operations. It lists compact rows (slug, title,
summary, risk, category, argument names with `*` = required);
ops_list_operations(slug=...) returns one operation's description, args_schema,
defaults and target. include_disabled=true helps investigation but does not
enable execution.

`ops_run(operation, args, idempotency_key)` validates args against the catalog.
Use a stable key for one logical request. Unknown/disabled operations, invalid
arguments and executor failures are errors, not invitations to use Bash/SSH.
Never retry an approval-required call: wait for its persisted result.

Keep job_id. Approval-gated jobs deliver their terminal receipt as a new turn;
do not poll (llm-bawt/reference/ops-actions.md step 3). Verify state/exit_code
plus target health from that receipt; ops_job_status is a one-shot read; the reconciler records progress.
output_tail_bytes adds bounded output. Queued or approved is not completed.
Bridge/Redis restarts and production release need an approved card per run.
Scheduling or forwarding a prompt cannot grant that authorization.

### Delegated ops approval (`approve_ops`, TASK-1609)

A bot holding the `approve_ops` grant tag (shown in gold as "Ops approver")
may decide other bots' pending ops_run approval cards:

- ops_approvals_list(): the pending ops_run cards you may decide, oldest
  first, with sensitive args masked. Cards your own bot filed, cards from
  private bots and other users' cards are never listed.
- ops_approval_resolve(request_id, decision="approve"|"deny", reason): the
  reason is required and shown on the card as "Approved/Denied by <you>". The
  approved job then runs and delivers its receipt exactly as for a person's
  approval.

You act only as yourself (the signed turn decides who you are). Refusal codes:
grant_required (you don't hold approve_ops), self_approval_refused (your own
bot's card: another holder or the user decides it), not_ops_card (only ops_run
cards; other approvals stay with a person), not_found, already_resolved (with
the stored decision; first decision wins, the user's included), unauthorized /
turn_capability_required / turn_capability_invalid (no valid signed turn).

The grant moves only through bots_grant_set(bot_id, tag, granted, reason), or
through a signed-in person on BawtHub Bot Tags. Only bots holding the
grant-manager role may call it (runtime setting
ops_grant_manager_capability; unset means persons only). Never your own grant,
never a private bot. Refusals: not_grant_manager, self_grant_refused,
access_denied. Every other tag writer (bots_profile_patch, profile PATCH, the
tag routes) refuses grant tags with grant_locked, and bots_profile_patch
refuses grant-manager role tags with tag_locked. The grant changes who may
decide a card, not the authorization rules in
`llm-bawt/reference/ops-actions.md`: a holder may approve every card class,
including production release, deploy, rollback and bridge/Redis restarts.

## Media library and downloads

Start with media_lookup(query, kind): kind=auto searches movie+series, or choose
movie/series. Query can be a title or external ID (tmdb:NNN for movies,
tvdb:NNN for series, imdb:ttNNN where supported). Resolve ambiguity and existing
library status before adding.

`media_add(kind, external_id=...)` uses an exact lookup ID. query is supported
but chooses the first match, so is unsuitable for ambiguous titles.
Either query or external_id is required. Existing titles return an
already_in_library flag instead of duplicating them.

- quality_profile: a service profile name; default HD-1080p if present, otherwise
  the first available profile. Do not invent profile names.
- monitored defaults true; search_now defaults true and triggers an indexer search.
- Series monitor: all, future, missing, existing, firstSeason, latestSeason, none.
- Movie minimum_availability: announced, inCinemas, released, preDB.

media_pipeline_status traces one title across library, queue and history.
media_queue reads Sonarr/Radarr/SAB queues, bounded per service.
media_history kind=all|movie|series, event=grabbed|imported|failed or omit.
media_library_stats is the compact overview. Check per-service errors even when
some of the response succeeded.

## Image generation

`generate_image(prompt, provider="grok"|"openai")` generates a durable asset plus
an inline image. Grok is the compatibility default; optional model/aspect_ratio/
resolution must be supported by the selected provider. user_id scopes ownership.
Read the returned image before deciding whether to iterate. Reuse its asset_id
as reference_asset_id with a change prompt; use returned URLs for links.
Inspect error payloads rather than claiming an image was generated.

## Playground

`playground_send(bot_id, title, note=None, asset_id=None | content=..., format="html"|"markdown"|"text")`
sends the user one artifact for Bots → Playground (`/bots/playground`): an existing `ma_…` asset (any
media; bytes are copied) or inline content up to 1 MB. The user can view it once;
closing it deletes it. Signed caller only, sent as that bot. A bell entry links to
the page. Use it for something worth looking at, not routine results.

## Speech and home audio

`home_audio_devices()` lists allowlisted Home Assistant targets and a voice-catalog
URL. Entity IDs matter: the Basement Speaker currently has the legacy entity ID
`media_player.bathroom_speaker`. Never infer a target or broadcast to All by default.

- `speech_generate(text, bot_id, voice?)` renders a durable PCM WAV without playing
  it. Omitted voice uses that bot's configured default; missing/unavailable voices
  fail rather than silently switching voices. Returns asset_id and duration.
- `home_audio_enqueue(target, bot_id, idempotency_key, text=..., voice=...)` renders
  and plays asynchronously; alternatively pass `asset_id` from speech_generate.
  Exactly one of text or asset_id. Reuse a key for retries of the SAME request.
  Same key with different input fails. Text limit 4,000 characters / five minutes.
- `home_audio_status(job_id)` reads durable state. queued/preparing/playing are
  not completion. completed requires observing this media URL playing then ending.
  failed/interrupted may have played partly; do not automatically retry them.
- `home_audio_cancel(job_id)` cancels queued jobs only. It never stops a user's
  music or an active announcement; check the returned status.

One global FIFO serializes bot announcements, including Cast groups. Busy/paused
players are left alone; queued work expires five minutes after enqueue. Unavailable
players fail. No volume changes, automatic music resume, or emergency interruption.
After ambiguous dispatch/crash, the lane is held for up to 330 seconds to prevent
an overlapping group announcement. External HA automations are not owned by this
queue and can still interrupt it. Never modify the garage-door safety automation.

Announcement playback reuses one Garage object and one asset row via the existing
upload routine's named-replacement option. Job `asset_id` identifies the current
playback slot, not a saved copy of that job's audio; never reuse it as clip input.
Use enqueue's text mode for disposable announcements. `speech_generate` explicitly
creates a separate saved clip under normal media-retention rules. Each playback
gets a fresh query version and no-store response to avoid cached speech.

Configuration: global DB runtime setting `home_audio` supports `tts_url`,
`media_base_url`, and `targets` (allowlist). Credentials reuse existing HA config.
The media base URL must be reachable by the actual speakers without login; Docker
service names and OAuth-gated frontend URLs will not work. See repository
`docs/HOME_AUDIO.md` for activation and verification. Source registration alone is
not live activation; don't announce that this works until a real playback test.

## Phone notifications

`notify_user(bot_id, message, title=None, urgency="normal", link=None, dedupe_key=None)`
pushes to the user's phone through Home Assistant `notify.<service>`.

- **When:** an approval waiting on the user, a release/deploy that finished or
  failed, or a decision that blocks work. Never routine progress, never "done"
  for ordinary tasks; the task board covers that.
- `urgency="high"` is time-sensitive (iOS) / high priority (Android) and is the only
  level that passes quiet hours. Use it for failures and blocking approvals only.
- `link`: an app-relative path (`/tasks/tasks/TASK-N`) or a URL on the public
  BawtHub origin; anything else is rejected. Tapping the push opens it.
- `dedupe_key`: reuse the same key for the same event (e.g. `deploy:<sha>`) so a
  retry or a second bot reporting it collapses into one push.
- Needs a signed caller (your turn capability) and notifies only as that bot;
  unsigned clients and private bots are refused. Off-origin links are refused.

Receipt `result`: `sent`; `partial:ha_error:<status>` (some phones got it; don't
retry); `suppressed:quiet_hours|dedupe|rate_limit`; `not_configured`;
`ha_error:<status|unreachable|not_connected|...>`; `refused:<reason>`. A
suppressed or refused push is not an error to work around: don't rephrase, change
`dedupe_key` or switch to `high` to get it through. Each call writes a
`notify_log` audit row (bot, message hash, result).

Configuration: global DB runtime setting `notify` = `{"ha_services":
["mobile_app_<device>"], "quiet_hours": {"start": "22:00", "end": "07:00",
"timezone": "<IANA zone>"}, "per_bot_interval_seconds": 600, "global_per_hour": 6,
"dedupe_window_seconds": 3600}`; only `ha_services` is required. HA URL and token
reuse the Home Assistant integration record.

## Web search

web_search uses configured local search providers; omit provider for parallel
fan-out or select brave/reddit/tavily/duckduckgo. max_results is PER provider,
not a total output cap. Results include source attribution. An empty result plus
error can mean no provider is configured. Use crawl4ai tools to fetch a specific
page rather than treating snippets as the full document.

### X public-post search

`x_search(query, max_results=10, start_time?, end_time?, next_token?, sort_order="recency", include_authors=false)`
and `x_counts(query, granularity="hour", start_time?, end_time?, next_token?)` use the
X API directly, not web indexing. Requires the **X (Twitter)** connection in
BawtHub provider accounts (also available as optional first-run setup). The
app-only bearer token uses the existing encrypted CredentialStore; it is not
an xAI/Grok key. Connection checks `/2/usage/tweets`, not a paid post search.

- Paid, explicit only: not part of automatic `web_search` fan-out. One page per
  call; max_results is 10–100 (X's minimum is 10). Never automatically paginate
  or retry. `next_token` requests another paid page.
- Searches the last seven days. `sort_order`: `recency` (newest first, default)
  or `relevancy` (X-ranked). Newest-first pages cover minutes on busy topics;
  use `relevancy` plus an engagement floor for "what happened" summaries.
  ISO-8601 time bounds must have a timezone, be within seven days and at least
  10 seconds ago.
- API operators: `from:`, quoted phrases, `OR`, `-is:retweet`, `-is:reply`,
  `lang:en`, `has:links`, `is:verified`, `min_likes:N`, `min_reposts:N`,
  `min_replies:N`. The x.com web names `min_faves:`/`min_retweets:` are rejected
  by the API (400 → `invalid_query`).
- Returns post text, IDs, author IDs, timestamps, source URLs and
  likes/reposts/replies/quotes (post fields, no extra resource).
  `include_authors=true` adds `author {username, name, verified, followers}` and
  handle URLs, but X bills returned user objects separately — opt in only when
  attribution matters (news credibility, "who said it").
- `x_counts` returns volume buckets (`minute|hour|day`), `total`, and the top-5
  `peaks`, no posts. Efficient larger-period pattern: counts over the window →
  pick peak buckets → `x_search` each peak's `start_time`/`end_time` with
  `sort_order=relevancy`, `-is:retweet` and a `min_reposts:` floor. Counts can
  differ slightly from search results (search applies extra compliance filtering).
- Missing credentials, invalid token, exhausted credits, forbidden access,
  rate limits and transport failures return explicit error codes. An empty
  successful result is not an error; no fallback to another provider occurs.
- No posting, DMs, background monitoring, bulk archive or model-training path.
  Set credit/spending limits in console.x.com. Health polling checks stored
  credentials only; it does not prove live search entitlement or credit balance.

## Browser-check sessions (`browser_check_session`, TASK-1659)

When a BawtHub deployment has sign-in sessions on, a bot's Playwright
browser or `curl` gets the sign-in page or a 401. `browser_check_session`
(no arguments) mints a session for the reserved, non-admin
`browser-check` account on your behalf:
- The bot is taken from your trusted turn and recorded on the session.
- The session lasts at most 1h and never renews. Call the tool again for a fresh one.
- It is non-admin. Admin-only pages need the user.

It returns `{origin, entityId, expiresAt, cookies}`. The cookies are
already in Playwright's `addCookies` shape:
- **Playwright:** `browser_run_code_unsafe` with `await page.context().addCookies(<cookies>)`, then navigate to `origin`.
- **curl:** send them as one header, `-H "Cookie: name1=value1; name2=value2"`.

The cookie values are a bearer. Never echo them in chat, task fields or
files. If the tool says it isn't configured, the deployment hasn't set
`BAWTHUB_BROWSER_CHECK_INTERNAL_URL` / `BAWTHUB_BROWSER_CHECK_ORIGIN`.
