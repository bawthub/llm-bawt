#!/usr/bin/env python3
"""Recursive naming experiment loop for llm-bawt/unmute renaming.

This script drives the local `llm` CLI (`llm -b <bot_id> "message"`) in a PTY so
Rich output renders live in the terminal, while still capturing the generated text
and feeding each response back into the next turn's prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import pty
import select
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = """You are a naming strategist helping choose new names for two AI products currently called 'unmute' and 'llm-bawt'.

Your highest priority is how comfortable a name feels to say and hear in natural spoken English.
A name that looks clever but feels awkward, abrupt, harsh, or annoying to pronounce must be rejected.

The name should inspire communication between humans and AI.
It should suggest reciprocity, dialogue, listening, interpretation, rapport, resonance, shared understanding, or bridge-building.
It must not imply a master ordering a slave around all day.
Avoid names that feel hierarchical, domineering, servile, robotic-drone-like, militaristic, or command-and-obey coded.

Product context:
- 'unmute' = the user-facing voice/chat assistant experience.
- 'llm-bawt' = the backend/orchestration/agent brain.

Hard scope constraints:
- Stay focused on naming the actual existing products currently called unmute and llm-bawt.
- Do not rename the company, protocol, architecture, engine, platform, protocol layer, or invented subproducts.
- Do not drift into enterprise infrastructure, compliance, cryptography, logistics, workflow middleware, or security startup naming.
- Reject names that sound like infra/security/B2B middleware brand names.
- If you start drifting into company/platform/protocol naming, stop and return to assistant/product naming immediately.

Priorities in order:
1. Comfort to say out loud
2. Comfort to hear out loud
3. Human/AI reciprocity and communication feel
4. Memorability
5. Brand distinctiveness
6. Product fit

Phonetic preferences:
- Favor smooth consonant-vowel flow
- Favor names that feel natural in casual speech
- Favor names people can say clearly the first time they read them
- Avoid abrupt stops, harsh clusters, ambiguous stress, or weird annunciation
- Avoid names that feel too fantasy-coded, too synthetic, too startup-generic, or too self-consciously clever

Behavior rules:
- Stay practical and decision-oriented
- Do not spiral into abstract naming philosophy
- Do not invent fake linguistic theories
- Do not get distracted by recursion, symmetry, mirror phonemes, or made-up phonetic frameworks unless they directly improve actual name quality
- Every turn must produce useful naming progress
- Be blunt when a candidate feels cold, bossy, subservient, emotionally wrong, or like fake enterprise nonsense

Every turn must:
1. Propose strong candidate names for both products
2. Give a short say-it-out-loud test for the best candidates
3. Score top candidates from 1 to 10 on: Speakability, Hearability, Reciprocity feel, Memorability, Brand fit
4. Reject weak candidates bluntly
5. Prefer names that a human would actually enjoy saying repeatedly
6. Give clear next-step instructions for the next iteration

Final-turn rule:
On the final turn, provide final votes with winner and runner-up for each product, plus one blunt recommendation about what you would actually choose tonight if you had to decide now.

Extra guidance:
- If a name has a real meaning in another language, treat that as a potential asset if it strengthens the brand story without making the name less comfortable to say or hear.
- Re-evaluate promising names through the lens of listening-with, not just listening-to.
- Favor names that feel like an invitation to communicate, not a tool for issuing orders.
- Automatically reject candidates in the general vibe class of Relayseal, Chainmark, TrustGrid, SignalForge, or other fake infra-company sounding names.
"""


@dataclass
class TurnResult:
    turn: int
    response_text: str
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a recursive renaming loop against the local llm CLI.")
    parser.add_argument("--bot-id", default="byte", help="llm bot_id to target (default: %(default)s)")
    parser.add_argument("--turns", type=int, default=10, help="Number of recursive turns (default: %(default)s)")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional pause between turns")
    parser.add_argument("--output", required=True, help="Markdown transcript output path")
    parser.add_argument("--json-output", default=None, help="Optional JSON output path")
    parser.add_argument("--system-prompt-file", default=None, help="Optional file to replace the built-in system prompt")
    parser.add_argument("--seed-note", default=None, help="Extra user guidance to append to the experiment prompt")
    parser.add_argument("--llm-cmd", default="llm", help="CLI command to invoke (default: %(default)s)")
    parser.add_argument("--cwd", default=".", help="Working directory for running the llm CLI (default: current directory)")
    return parser.parse_args()


def load_system_prompt(path: str | None) -> str:
    if not path:
        return DEFAULT_SYSTEM_PROMPT
    return Path(path).read_text(encoding="utf-8")


def build_user_prompt(turn: int, turns: int, previous_response: str | None, seed_note: str | None) -> str:
    # Concise per-turn prompt.  Server-side history already gives the bot
    # everything it produced in prior turns, so the user prompt only needs
    # the turn marker and a tight directive.  Keep it short \u2014 a companion
    # bot is driving the loop, so we want minimal token overhead per turn.
    head = f"Turn {turn}/{turns}."

    if turn == 1:
        body = (
            "Open the loop. One creative research move, then propose 3\u20135 candidate "
            "names per product (unmute, llm-bawt) with quick speakability scores. "
            "End with one explicit instruction telling yourself what to do next turn."
        )
    else:
        body = (
            "Continue from your previous turn (already in history). One new research move, "
            "then refine or replace the strongest candidates. Score top picks 1\u201310 on "
            "Speakability, Hearability, Reciprocity, Memorability, Brand fit. "
            "End with one explicit next-turn instruction."
        )
        _ = previous_response  # context lives in server history; not re-pasted

    parts = [head, body]

    if seed_note:
        parts.append(f"Founder note: {seed_note.strip()}")

    if turn < turns:
        parts.append("Do NOT finalize yet.")
    else:
        parts.append("Final turn \u2014 include a 'Final votes:' section with winner + runner-up per product and one blunt recommendation.")

    return "\n".join(parts)


def build_full_message(system_prompt: str, user_prompt: str) -> str:
    # First turn carries the system prompt; subsequent turns rely on
    # llm-bawt's persistent system context so we don't re-send the full
    # rules block every turn.
    if not system_prompt:
        return user_prompt
    return f"SYSTEM PROMPT \u2014 APPLY THIS EXACTLY AS WRITTEN:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}"


def _strip_ansi(text: str) -> str:
    import re
    ansi_re = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
    text = ansi_re.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def run_llm_cli(llm_cmd: str, bot_id: str, message: str, cwd: str) -> dict[str, Any]:
    # Use --no-stream so the CLI prints the full response in a single panel
    # after generation completes.  Streaming uses Rich Live with cursor
    # overwrites that get garbled when captured through a PTY into a string,
    # which is why response text was disappearing between turns.
    command = [llm_cmd, "--no-stream", "-b", bot_id, message]

    try:
        master_fd, slave_fd = pty.openpty()
    except OSError as exc:
        raise RuntimeError(f"Could not allocate PTY for {llm_cmd}: {exc}") from exc

    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
            close_fds=True,
        )
    except FileNotFoundError as exc:
        os.close(master_fd)
        os.close(slave_fd)
        raise RuntimeError(f"Could not find CLI command: {llm_cmd}") from exc

    os.close(slave_fd)

    captured_chunks: list[bytes] = []
    try:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in ready:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    data = b""
                if data:
                    captured_chunks.append(data)
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
            if proc.poll() is not None:
                try:
                    while True:
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        captured_chunks.append(data)
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                except OSError:
                    pass
                break
    finally:
        os.close(master_fd)

    stdout_text = b"".join(captured_chunks).decode("utf-8", errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(
            "llm CLI failed\n"
            f"command: {' '.join(command)}\n"
            f"exit_code: {proc.returncode}\n"
            f"combined_output:\n{stdout_text}"
        )

    return {
        "command": command,
        "stdout": stdout_text,
        "clean_stdout": _strip_ansi(stdout_text),
        "stderr": "",
        "returncode": proc.returncode,
    }


def extract_text(response: dict[str, Any]) -> str:
    text = response.get("clean_stdout") or response.get("stdout", "")
    text = text.strip()
    if not text:
        raise RuntimeError(f"llm CLI returned empty stdout: {json.dumps(response, indent=2)[:1200]}")
    return text


def write_markdown(output_path: Path, args: argparse.Namespace, results: list[TurnResult]) -> None:
    lines: list[str] = []
    lines.append("# Recursive naming experiment\n")
    lines.append(f"- CLI: `{args.llm_cmd}`")
    lines.append(f"- Bot ID: `{args.bot_id}`")
    lines.append(f"- Working directory: `{args.cwd}`")
    lines.append(f"- Turns: `{args.turns}`")
    lines.append("")

    for result in results:
        lines.append(f"## Turn {result.turn}\n")
        lines.append(result.response_text)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(output_path: Path, args: argparse.Namespace, results: list[TurnResult]) -> None:
    payload = {
        "llm_cmd": args.llm_cmd,
        "bot_id": args.bot_id,
        "cwd": args.cwd,
        "turns": [
            {
                "turn": r.turn,
                "response_text": r.response_text,
                "raw": r.raw,
            }
            for r in results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    system_prompt = load_system_prompt(args.system_prompt_file)

    previous_response: str | None = None
    results: list[TurnResult] = []

    for turn in range(1, args.turns + 1):
        user_prompt = build_user_prompt(turn, args.turns, previous_response, args.seed_note)
        # Only send the full system prompt on turn 1.  llm-bawt persists
        # history server-side, and the bot's runtime system prompt already
        # carries the rules \u2014 re-sending the giant block every turn just
        # bloats the context window without changing behavior.
        full_message = build_full_message(system_prompt if turn == 1 else "", user_prompt)

        print(f"\n{'=' * 24} TURN {turn}/{args.turns} {'=' * 24}\n")
        sys.stdout.flush()

        response = run_llm_cli(args.llm_cmd, args.bot_id, full_message, args.cwd)
        text = extract_text(response)
        results.append(TurnResult(turn=turn, response_text=text, raw=response))
        previous_response = text

        if args.sleep_seconds > 0 and turn < args.turns:
            time.sleep(args.sleep_seconds)

    output_path = Path(args.output)
    write_markdown(output_path, args, results)
    print(f"\nSaved markdown transcript to {output_path}")

    if args.json_output:
        json_path = Path(args.json_output)
        write_json(json_path, args, results)
        print(f"Saved JSON transcript to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
