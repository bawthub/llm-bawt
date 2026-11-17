# Direct voice credential migration

The shared `VoiceBackendAuth` authenticates HomeAudioClient rendering, GPU
preflight and handoff sessions/park/unpark, and the local-model bridge's video
lease check. Its credential is only sent to origins explicitly declared in
`LLM_BAWT_VOICE_BACKEND_ORIGINS`; changing a destination setting never expands
that allowlist.

The global runtime setting `voice_backend_auth_mode` controls the wire format:

| Value | Outgoing voice header |
| --- | --- |
| No global row, or `legacy` | Existing raw service seed |
| `derived-v1` | Lowercase hex HMAC-SHA256 of the trimmed UTF-8 seed with UTF-8 message `bawthub/voice-proxy/v1` |

This setting is an operator-controlled protocol migration, not a bot setting.
Generic settings PUT, batch and DELETE refuse it. Use the installation's
trusted operator database/configuration path to change the global `*` row.
Only a successfully read absent row means legacy; an unavailable DB, invalid
JSON, JSON null or unsupported value refuses authentication rather than
silently selecting the old credential.

The app and local-model bridge read the same DB row for every outgoing
trusted-origin request. There is no process-local mode cache. Seed rotation
is also seen per request. Async clients move the blocking reads off their
event loop. Service API Bearer authentication is unchanged.

## Paired activation

1. Publish reviewed client code and load it in **both the app and local-model
   bridge** before the window. Keep the setting absent or `legacy`; an
   unrelated restart must not change the credential.
2. Keep the previous matching frontend, backend and HA configuration available.
   Stage the derived-only frontend/backend changes. For an existing source-
   mounted installation, release frontend, align the canonical checkout and
   restart the voice backend in the coordinated window.
3. Immediately set the global mode to `derived-v1`. No app or video bridge
   restart is needed for this flip.
4. Replace every HA direct client's `proxy_token` with the derived credential
   privately, and apply its supported reload/restart. Never print or persist
   the service seed in a receipt. Do not relax backend enforcement.
5. Check home-audio rendering and voice/GPU read-only probes; do not park an
   active session as an acceptance shortcut. Verify frontend HTTP/WebSocket
   positive controls, direct missing/raw-seed refusals, and HA synth-only audio.
   A generated/downloaded TTS file needs no media-player service or speaker.

## Rollback

Restore the previous matching frontend/backend within the same window, flip
the global mode to `legacy`, and restore/reload HA's prior raw-compatible
configuration. The mode rollback is live, not a source rollback or restart.
Do not leave mixed protocols, add raw-token backend compatibility, or disable
enforcement as a shortcut.

The nonsecret shared test vector is seed `"v" * 40`, credential
`dd8e93cebe290185616abf0205545265abc5fdbdc317e35b628c799b518aa242`.
It is also pinned by BawtHub's TypeScript and Python proxy tests.
