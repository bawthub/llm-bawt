# Message retrieval: `messages_find` and `messages_context`

Two MCP tools let an agent locate and read past conversation across every bot
partition it may read. `messages_find` returns compact cards, not transcripts.
`messages_context` opens one message and its same-session neighbours. Both are
read-only and budgeted, and they answer the same way for "denied" as for
"does not exist".

Code: `src/llm_bawt/retrieval/{find,find_query,context,handles}.py`, tool
adapters in `src/llm_bawt/mcp_server/find_tools.py`. Tests:
`tests/test_messages_find_pg.py`, `tests/test_messages_context_pg.py`.

## Access contract (applies to both tools)

- **Trusted caller only.** The caller comes from the verified MCP identity,
  never from an argument. No verified caller means `unavailable(caller_unavailable)`.
- **Scope is resolved fresh on every call** from the current bot-access policy.
  It only narrows: a bot partition is readable only if the policy grants it,
  rows must belong to a session pinned to the caller's user and in a live
  status (`active` or `archived`), and the row's effective author must match
  its role. A caller reading its own partition gets a stricter author rule,
  not a looser one.
- **The predicate runs in the `WHERE` clause**, before matching, ranking or
  counting. Every total, facet, card, snippet and neighbour comes from rows
  that passed it, so counts never reveal hidden rows.
- **No existence oracle.** An excluded or unknown `bot` filter narrows the
  scope to nothing. `messages_find` answers `no_match`, the same as an empty
  in-scope bot. `messages_context` answers `not_found`, the same as an id that
  never existed. These tools sit outside the generic bot-access guard
  (`SCOPED_RETRIEVAL_TOOLS`) because that guard's `access_denied` would
  confirm the bot exists.
- **The caller's own trigger message is excluded**, so a turn never finds the
  question it is currently asking.
- **Snapshot and timeouts.** Each call reads in one `REPEATABLE READ, READ ONLY`
  transaction with a statement timeout. The whole tool call is capped at
  8 seconds and answers `unavailable(timeout)` past that.

Status values: `ok`, `no_match` (find), `not_found` (context), or
`unavailable` with a `reason`: `invalid_request`, `handle_expired`,
`timeout`, `source_unavailable`, `response_overflow`, `caller_unavailable`.
Internal errors are never surfaced verbatim.

## Handles and cursors

Continuation keys come from an in-process store (`retrieval/handles.py`):

- **Owner-bound.** A key belongs to the `(bot_id, user_id)` that received it.
  Another owner's key reads as `handle_expired`, the same as an unknown key.
- **Expiring.** 15 minutes TTL, a global capacity, and a per-owner cap (8 find
  handles, 16 context cursors); the oldest key is evicted past the cap.
- **Intent, not content.** A payload holds the request and the identities
  already served, never message text. Every use reauthorizes against the
  current policy, so a revoked grant takes effect on the next page.
- **Not durable.** After a restart every key is expired.

## `messages_find`

Arguments: `query`, `bot`, `speaker` (`user` | `bot` | `delivery`), `since`,
`until`, `handle`.

- `since` / `until` accept `Nh` / `Nd` back from now (1h–8760h, 1d–365d), an
  ISO date or datetime, or Unix seconds. Anything else, including blank or
  non-finite values, is `invalid_request` and the reply lists the accepted
  formats.
- Omitting `query` browses recent messages. `handle` alone fetches the next
  page (up to 4 pages).

**Query ladder.** The query is parsed into identifier tokens (task ids, SHAs,
paths, snake_case, versions, camelCase) and words. Identifiers match as
case-insensitive substrings, words by stem. Rungs are tried in order
(`exact_all`, `exact_any`, `parts`, `parts_loose`, `words`, `words_loose`);
the first rung with matches is the primary, and it alone defines `total` and
the facets. Later rungs only fill spare space with labelled cards.

**Response.** Up to 6 best and 4 latest cards, collapsed to one per session,
each with a `ref` (`c<page>.<n>`), `bot`, `id`, `session`, a snippet and up to
3 `also` ids from the same session. Facets by bot, age and speaker; per-rung
totals; `omitted` counts; `outside_filters` when a filter hid matches; and the
`handle`. The serialized reply must fit 3,500 characters: snippets shrink
through fixed widths, and fill-dependent fields are reserved at maximum width
before measuring, so filling can never push a fitted reply over budget.

## `messages_context`

Arguments: an anchor, either `bot` + `id` or `handle` + `card_ref`; plus
`before`, `after`, `focus`, `max_chars`, `offset`, `cursor`.

- **Anchor.** A `card_ref` resolves from the handle's served record and never
  re-runs the find, so a message inserted between the two calls cannot change
  what a ref means. A ref the page did not serve is `not_found`. Either way the
  anchor is reauthorized before anything is read. An anchor with no session is
  `not_found`.
- **Neighbours** come only from the anchor's own bot partition and session, in
  the stable order `(timestamp, id)`. Each passes the scope predicate, so a
  skipped row never appears and never changes `more_before` / `more_after`.
- **Bounds.** `before` / `after`: 0–10, default 3. `max_chars`: 1,000–8,000,
  default 4,000. Out of range is `invalid_request`.
- **Budget.** The anchor gets up to half of `max_chars`. A longer anchor
  returns `truncated` and `next_offset`; call again with `offset` to continue.
  An offset past the end is `invalid_request`. Neighbours are 200-character
  snippets, centred on `focus` terms when given (up to 6). If the reply is
  still too long, neighbours shrink to 80 characters, then the farthest are
  dropped (ties drop the earlier side first). The anchor is never dropped; if
  it alone cannot fit, the answer is `response_overflow`.
- **Paging.** `more_before` / `more_after` are cursors when more neighbours
  exist on that side. A cursor call returns the next neighbours on that side
  only, after reauthorizing the anchor. Open any neighbour in full by passing
  its `bot` + `id` as a new anchor.

## Changing this contract

Both tools count against the MCP catalog token budget
(`tests/test_mcp_catalog.py`). Any change to scope, ordering or denial
behaviour needs a matching PostgreSQL acceptance test in the files above.
