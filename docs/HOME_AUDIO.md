# Bot speech → Google Home / Nest speakers and displays

## Observed reference

A read-only reference inspection found a Home Assistant garage-door alert using
`media_player.play_media`, a `media-source://tts/google_translate?...` URL and
type `provider`. It also switches a device and sends a phone notification.
This feature reuses only the media-player transport; it does not alter or trigger
the reference automation. Entity IDs and device names are deployment configuration.

## Architecture

1. MCP `speech_generate` renders without playback, or `home_audio_enqueue` queues
   text for synthesis. An explicit voice or bot-profile `default_voice` is required.
2. BawtHub `POST /v1/tts/render` uses existing `stream_utterance(strict=True)` to
   return 16-bit PCM WAV. No silent voice/provider fallback. Existing preview
   behavior is unchanged.
3. `MediaStore.upload(..., replace_key="announcements/current.wav")` stores
   playback in one Garage object (`blobs/named/announcements/current.wav`) and
   reuses one `media_assets` row. The worker replaces it only while owning the
   global queue lane, after the preceding playback completes (or its uncertainty
   hold expires). Saved `speech_generate` clips remain immutable, separate assets.
4. App-owned `HomeAudioWorker` claims PostgreSQL FIFO jobs. A global leased lane
   serializes announcements across bots and overlapping Cast groups.
5. HA `media_player.play_media` receives `/v1/uploads/<stable-id>?v=<job-id>`.
   Mutable responses use `Cache-Control: no-store`, never immutable caching/304.
6. The worker observes the exact media URL playing, then idle/off. Status is not
   completed just because HA accepted the service call.

MCP tools: `home_audio_devices`, `speech_generate`, `home_audio_enqueue`,
`home_audio_status`, `home_audio_cancel`. See the installed bawthub-mcp ops/media
reference for calling examples and hazards.

## Configuration

Global DB runtime setting `home_audio` (object; omitted fields use implementation
defaults). This is an illustrative configuration, not deployed values: replace
the example origins and entity IDs with the installation's reachable services
and explicitly selected targets.

```json
{
  "tts_url": "http://tts.example/api",
  "media_base_url": "http://media.example:8642",
  "targets": [
    "media_player.example_display",
    "media_player.example_speaker"
  ]
}
```

HA base/token reuse `HA_NATIVE_MCP_URL` / `HA_NATIVE_MCP_TOKEN`; the native-MCP
feature flag does not gate this REST integration. No credentials in runtime JSON.
Discover the installation's entity IDs rather than assuming a device's display
name matches its entity ID. Target selection is explicit; All is never a default.

The reference TTS base was verified via `/api/v1/tts/providers`. The media base must
be accessible directly by Cast clients with no OAuth/cookies and valid LAN routing.
The original immutable-asset flow passed live tests on two displays with Azure
and Moshi. Named replacement needs a fresh live verification after app activation.
If a bot has no default voice, supply a voice or configure the bot profile.

## Queue semantics

- PostgreSQL tables `home_audio_jobs` and `home_audio_lane` are bootstrapped through
  the existing schema guard. No binary media in the queue.
- Idempotency is scoped by bot ID; duplicate input returns the original job;
  changed input with the same key errors. FIFO uses DB sequence, not random IDs.
- A queued job expires five minutes after acceptance. Busy/buffering/paused targets
  are not interrupted. Unavailable targets fail. Expiration is rechecked after TTS.
- Lease: 60 seconds, heartbeat every 10. No automatic replay after any ambiguous
  dispatch, failed playback, or worker crash. Active jobs become interrupted.
- After uncertain playback, hold the global lane 330 seconds (maximum clip length
  plus startup buffer) to avoid overlapping a still-playing Cast group.
- Cancellation only changes queued jobs; it never stops an active speaker.
- No volume changes, device power automations, saved-media resume, or priority
  bypass. Existing HA automations remain independent and may interrupt playback.
- Clips that start/end between polling intervals may report failed rather than
  falsely completed. A failed job may have been audible; never blindly retry.
- Playback storage is one replaceable object/asset row; old job `asset_id` values
  identify that slot, NOT an archive of the old audio. Do not enqueue that mutable
  ID as a saved clip. Use text or an immutable `speech_generate` result.
- Named slots are excluded from orphan GC (bounded reusable infrastructure).
  Earlier immutable test clips retain normal seven-day orphan cleanup. Saved
  speech follows the existing media rules. Job history pruning is unchanged.

## Named upload implementation

The existing upload API gains an internal `replace_key` keyword for file assets.
Normal uploads and the public multipart route do not opt in. `storage_key` is a
nullable column; immutable SHA uniqueness becomes a partial unique index so a
saved clip and the mutable slot can contain identical bytes without aliasing.
Named upserts lock per key and preserve ID; failed blob writes roll back metadata.
Reads checksum mutable bytes and fail closed on an incomplete cross-store update.
FS uses atomic rename; S3 uses object replacement through the same configured client.
No delete/recreate gap or extra serving system. This source change needs app reload
for the idempotent schema migration and worker activation.

## Activation / verification

Source edits are not proof of active tools. Follow ops-first authorization for
service activation, never restart the bridge/Redis for this feature. Required
services: BawtHub voice backend (render route), llm-bawt app/MCP (tools + worker).
There are no added Python dependencies or frontend changes.

1. Run the isolated home-audio, MCP-catalog and speech-render tests.
2. After authorized activation, verify `/v1/tts/render` and fresh MCP discovery.
3. Generate a short WAV silently using a selected available voice; inspect asset
   MIME, byte length, duration, and unauthenticated LAN URL access.
4. With the operator's selected target/test timing, enqueue one audible test. Verify the
   actual device, exact URL playback state, and terminal job status.
5. Queue two distinct short messages and check ordering; replay the same key and
   confirm no second announcement. Test cancellation while another job owns lane.

Do not use the garage-door automation itself as a smoke test: it has unrelated
physical and phone-notification side effects.
