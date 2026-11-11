# Commit-to-turn review attribution

`POST /v1/git/commits` records a commit immediately after a signed
`commit-slice` commit. The existing task-turn capability supplies the bot and
turn; neither can be supplied in the request body. The request contains a
repository key, full Git SHA and bounded base64 commit object. The app checks
the Git object hash and its author against the signed bot identity before
inserting an immutable `(repo_key, commit_sha, turn_id, bot_id)` receipt.
Replays of the same binding are idempotent; attempts to reassign it return
409. The schema is additive and bootstraps once per engine.

This extends TASK-1280's existing six-hour bearer attribution, not its trust
model: possession of a valid turn capability does not prove that the turn is
still executing. It is advisory provenance, not an independent Git signature.

The helper resolves identity and reports the commit on the same execution
side, including SSH remote commits. It never transmits remote credentials or
filesystem paths: repository keys hash the normalized origin host/path
(shared by SSH and HTTPS), or the Git common directory for a local-only repo.
It posts only its newly created, identity-verified commit after HEAD moves.
There is no manual arbitrary-SHA reporting command or historical backfill.
If reporting fails, Git still succeeded: the helper warns that attribution
is unresolved and explicitly says not to repeat the commit.

## Review resolution

The review's `targetRevision` selects recorded commits, not its body,
requester, Git author text alone, or a bot's current model. The resolver reads
each receipt's exact `(turn_id, bot_id)` in `turn_logs`, then resolves its
model endpoint's provider from the existing model catalog.

- Single SHAs and annotated pairs such as `llm-bawt SHA + BawtHub SHA` work.
- A trailing ` on SHA` denotes the base and is excluded, as in `SHA on BASE`.
  Other hexadecimal tokens are conservatively treated as targets.
- Tokens need at least 10 characters. Short tokens invalidate the target
  set; they are never silently dropped from a pair. Full SHAs are preferred.
  Prefix uniqueness is only known among receipts, not the whole Git repo.
- Every target must have exactly one recorded match and agree on author bot
  and exact model. Missing, ambiguous, mixed-author/model, unsupported or
  unavailable evidence stays null with a specific provenance note.
- `crossProvider` is set only when author and reviewer providers are both
  known. False is a known same-provider result, rendered by the existing UI
  and noted as `same_provider:advisory`; it does not change review policy.
- Raw/manual commits and unrecorded cherry-picks stay unresolved. Never
  transfer another commit's attribution merely because its patch is equal.

Existing keyed-review immutability is unchanged. If a previously unresolved
record becomes available between attempts, provenance can differ and the
existing retry returns a provenance-mismatch conflict. Use a new review key
or a supported recheck; do not overwrite the historical review.

## Activation

Publish the backend with its exact-tip gate, restart the app (its MCP server
is hosted in the same app service), then sync the helper skills revision.
No bridge rebuild or restart is needed for the helper change. Verify a new
signed commit's receipt and author-model resolution after activation; old
commits without receipts intentionally remain unresolved.
