"""Streaming-latency probe for llm-bawt's /v1/chat/completions.

Measures stage-by-stage timings so we can tell where voice-mode lag is
coming from:

  t_first_byte         — first SSE byte (TCP+TLS+route handoff)
  t_first_warning      — first service.warning event (usually model fallback)
  t_first_content      — first real text-content delta  ← this is what
                          the voice pipeline waits on before starting TTS
  t_last_content       — last text-content delta
  t_service_animation  — service.animation event (TASK-215)
  t_done               — [DONE] sentinel arrives
  t_total              — connection closed

Usage:
  python scripts/probe_voice_latency.py --bot <bot_id>              # 3 trials per mode
  python scripts/probe_voice_latency.py --bot <bot_id> --trials 5   # more samples
  python scripts/probe_voice_latency.py --bot <bot_id> --url http://localhost:8642
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class TurnTimings:
    """All timings stored as ms-from-t0."""
    first_byte: float | None = None
    first_warning: float | None = None
    first_content: float | None = None
    last_content: float | None = None
    service_animation: float | None = None
    done: float | None = None
    total: float | None = None
    content_chars: int = 0
    warnings: list[str] = field(default_factory=list)
    animation_picked: str | None = None


def probe_once(url: str, payload: dict, timeout: float = 60.0) -> TurnTimings:
    """Run one streaming request, capture per-stage timings."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    timings = TurnTimings()
    t0 = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                now = (time.perf_counter() - t0) * 1000
                if timings.first_byte is None:
                    timings.first_byte = now
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    timings.done = now
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                obj = data.get("object")
                if obj == "service.warning":
                    if timings.first_warning is None:
                        timings.first_warning = now
                    warnings = data.get("warnings") or []
                    for w in warnings:
                        # Trim long ones
                        timings.warnings.append(w[:100] + ("…" if len(w) > 100 else ""))
                    continue
                if obj == "service.animation":
                    timings.service_animation = now
                    timings.animation_picked = data.get("animation")
                    continue

                choices = data.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    if timings.first_content is None:
                        timings.first_content = now
                    timings.last_content = now
                    timings.content_chars += len(content)
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}", file=sys.stderr)
        return timings
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return timings

    timings.total = (time.perf_counter() - t0) * 1000
    return timings


def fmt(v: float | None) -> str:
    return f"{v:6.0f}ms" if v is not None else "    n/a"


def print_table(name: str, runs: list[TurnTimings]) -> None:
    print(f"\n┌─ {name} " + "─" * max(0, 78 - len(name) - 4))
    print(f"│ trial │ first_byte │ first_content │ last_content │ animation │ done  │ chars")
    for i, t in enumerate(runs):
        print(
            f"│   {i+1}   │ {fmt(t.first_byte)} │  {fmt(t.first_content)}   │"
            f" {fmt(t.last_content)} │ {fmt(t.service_animation)} │"
            f" {fmt(t.done)} │ {t.content_chars}"
        )
    # Aggregates over runs that actually produced content
    fcs = [t.first_content for t in runs if t.first_content is not None]
    totals = [t.done for t in runs if t.done is not None]
    if fcs:
        print(f"│ first_content  →  median {statistics.median(fcs):.0f}ms"
              f"  min {min(fcs):.0f}ms  max {max(fcs):.0f}ms")
    if totals:
        print(f"│ done           →  median {statistics.median(totals):.0f}ms"
              f"  min {min(totals):.0f}ms  max {max(totals):.0f}ms")
    # Surface any warnings we collected (deduped)
    all_warnings = {w for t in runs for w in t.warnings}
    if all_warnings:
        for w in sorted(all_warnings):
            print(f"│ warning: {w}")
    picks = [t.animation_picked for t in runs if t.animation_picked]
    if picks:
        print(f"│ animations picked: {', '.join(picks)}")
    print("└" + "─" * 79)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8642",
                   help="llm-bawt service base URL (default: http://localhost:8642)")
    p.add_argument("--bot", required=True, help="bot_id to probe")
    p.add_argument("--trials", type=int, default=3, help="trials per mode (default: 3)")
    p.add_argument("--user", default="probe_voice_latency", help="user id")
    p.add_argument("--prompt", default="Say a single short sentence about the weather.",
                   help="prompt text")
    args = p.parse_args()

    base_payload = {
        "model": args.bot,
        "bot_id": args.bot,
        "user": args.user,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
    }

    sample_animations = [
        {"name": "Head Nod Yes", "description": "Agreement, confirmation, saying yes"},
        {"name": "Happy Hand Gesture", "description": "Enthusiasm, excitement, good news"},
        {"name": "Acknowledging", "description": "Understanding, listening"},
        {"name": "Weight Shift", "description": "Neutral idle stance"},
    ]

    modes = [
        ("baseline (no tts, no animations)", {
            **base_payload,
        }),
        ("tts_mode=true, avatar_visible=false (no classifier)", {
            **base_payload,
            "tts_mode": True,
            "avatar_visible": False,
            "animations": sample_animations,
        }),
        ("tts_mode=true, avatar_visible=true (classifier runs)", {
            **base_payload,
            "tts_mode": True,
            "avatar_visible": True,
            "animations": sample_animations,
        }),
    ]

    print(f"Probing {args.url} bot={args.bot} trials={args.trials} prompt={args.prompt!r}")

    for name, payload in modes:
        runs: list[TurnTimings] = []
        for i in range(args.trials):
            t = probe_once(args.url, payload)
            runs.append(t)
            # Brief pause so we don't slam the rate limits
            time.sleep(0.5)
        print_table(name, runs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
