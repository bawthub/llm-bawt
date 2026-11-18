# Turn-duration statistics

`GET /v1/turn-logs/duration-stats` is a read-only endpoint on the existing
authenticated service API. It returns `by_bot` and `by_tag` arrays with `count`,
`p50_ms`, `p95_ms`, and `max_ms`. No prompt, response, or individual turn is returned.

- The default window is seven days, ending at request time in UTC. `days=1..7`
  may shorten it. A turn's `created_at` must be in `[window_start, window_end)`,
  and it must have finished by `window_end`.
- Samples are retained `turn_logs.latency_ms` values that are finite and
  nonnegative. Missing durations and running turns are excluded. Finished
  failures and cancellations count too; this is not a successful-turn-only metric.
- Optional `tag`, `bot_id`, and `user_id` filters intersect. `tag` is a catalog
  slug, not a hard-coded role. Unknown tags return 422.
- Bot visibility and tag membership come from one validated **current** tag
  snapshot in the same repeatable-read transaction as the aggregate. Historical
  turns are grouped by today's membership, not membership at turn time.
- Private bots, profiles excluded from global search, bots tagged to prohibit
  global search, and unknown bots never contribute, even with an explicit bot
  filter. Inactive but searchable bots may contribute historical samples.
- A tag percentile pools all eligible turns from its member bots; it is not an
  average of bot percentiles. Multi-tag bots contribute once to each tag.
  Percentiles use PostgreSQL continuous interpolation.
- Empty cohorts produce empty arrays, not fabricated zero-duration samples.
  Retention gaps are not filled. Window and sample semantics accompany the data.
- Unavailable database or tag policy returns 503, never an unfiltered fallback.
  The transaction is read-only with a 5 s statement timeout and 500 ms lock timeout.

Example: `/v1/turn-logs/duration-stats?tag=your-role&days=7`.
This backend view needs no frontend activation or live configuration change.
