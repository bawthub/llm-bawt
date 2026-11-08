# Recorded tool-call lookup

`tool_calls` is read-only. Omit `id` to list calls, grouped into recent turn
segments; pass a positive numeric `id` to inspect one call. Arguments appear in
both modes. Result bodies never appear unless `id` and `include_result=true`
are supplied together.

## Scope and authorization

The trusted caller and current bot policy are resolved on every invocation.
Reads reuse `messages_find`'s positive partition map, user/session status and
effective-author rules. A call must match its turn's bot and user, and the turn
must have a readable trigger message. Missing, excluded or malformed provenance
is invisible. The current request's trigger is excluded. An unknown or unreadable
id yields the same `not_found`; excluded, unknown and empty bot filters all yield
the same `no_match`.

Authorization is selected before reading argument/result text. Retrieval uses a
repeatable-read, read-only transaction with the existing history-reader statement
and lock timeouts. It does not construct the mutating/bootstrapping turn store.

## Filters and continuation

List filters: `bot`, exact `turn_id`, exact `tool_name`, `since`, `until`,
`errors_only`, and `limit` (default 10, maximum 25). Time values use the same
relative (`24h`, `7d`), ISO or Unix-second parser as `messages_find`. The default
window is the past 24 hours; an explicit turn id has no default lower bound.
`errors_only` selects explicit true; unknown/NULL is not an error.

Calls sort by descending `(created_at,id)`. Copy the fields in `next` into the
next invocation and repeat other filters. `next` pins the initial time window;
`null` means this window has no more matching records. Each page reauthorizes
under current policy. A page may split a turn (`partial_turns=true`); segments
are not guaranteed to contain every call from that turn.

Detail permits the optional `bot` narrowing filter, but cannot be mixed with
list filters, a cursor or a time window.

## Size and hygiene

List replies are at most 4,000 serialized UTF-8 bytes; detail replies at most
16,000 bytes, including metadata and JSON escaping. `max_chars` is 1,000–8,000
(default 4,000), not permission to return an unbounded argument or result.
Argument reads are capped at 65,536 source characters. Default list arguments
have 48-character fields and a 160-character combined view; detail has
400-character fields and at most 2,400 combined characters. Nested structures
also have depth/node/container limits. Truncation is explicit.

Secret-ish nested keys, common token forms, bearer strings and secret
assignments in text (including shell commands) are scrubbed before clipping.
Known `ops_run` operations additionally reuse their schema's `x-sensitive`
masking. Oversized, unparseable ops arguments are omitted rather than skipping
schema masks. Results receive the same key/string heuristics.

**This is best-effort hygiene, not a guarantee that arbitrary tool arguments or
results are secret-free. Authorization is the actual data boundary.**

## Result sources and integrity

Small results may exist only in the canonical preview even when
`result_payload_available=false`. Overflow results use a bounded prefix of the
legacy payload or an object-store range read, never the whole-object paging
helper. Returned metadata preserves the source size, completeness and SHA-256,
not the hash of the scrubbed excerpt.

`integrity=verified` means the whole stored payload fit the bounded read and its
hash matched. Partial prefixes cannot verify the full source hash and say
`unverified`. Full-payload hash mismatches withhold content. Missing payloads or
unavailable object sources return an unavailable result while retaining the
authorized call's metadata. No result paging or tail/full-payload export is
provided by this compact tool.
