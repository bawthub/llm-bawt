"""TASK-1976: ``apply_patch`` for GPT routes, translated to native Claude Code tools.

GPT models are trained to edit files with the Codex patch format. Presenting
Claude Code's Edit/Write instead makes them stream whole files and long
old/new strings as JSON-escaped arguments, which is where the whitespace stalls
and repetition loops happen (TASK-1966).

When the catalog flag is on, the Responses request advertises one freeform
``apply_patch`` tool in place of Edit/Write/MultiEdit. The patch the model
writes is held until it is complete and then emitted to the Claude Code SDK as
ordinary Edit / Write / Bash(rm) tool_use blocks, so the CLI still runs its own
tools: same permission checks, read-before-edit tracking, transcript and chat
cards. On the next request those blocks are folded back into one apply_patch
call and one output, regenerated deterministically from the block inputs so
the replayed history is stateless and byte-stable.

The patch grammar follows OpenAI Codex's apply_patch tool (Apache-2.0).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

APPLY_PATCH_HEADER = "x-llm-bawt-apply-patch"
APPLY_PATCH_TOOL = "apply_patch"
REPLACED_TOOLS = frozenset({"Edit", "Write", "MultiEdit"})

_ID_RE = re.compile(r"^(?P<call>.+)__ap(?:(?P<i>\d+)of(?P<n>\d+)|err)$")

APPLY_PATCH_GRAMMAR = r"""start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
"""

APPLY_PATCH_DESCRIPTION = """Edit files with a patch. This is the only way to create, change or delete files.

*** Begin Patch
*** Update File: /abs/path/to/file.py
@@ def existing_function():
 context line (starts with a space)
-line to remove
+line to add
 context line
*** Add File: /abs/path/new_file.txt
+every line of the new file starts with +
*** Delete File: /abs/path/old_file.txt
*** End Patch

Rules:
- Use absolute paths. Read a file before updating it.
- Each @@ section must match the current file exactly and uniquely: include 2-3 unchanged context lines around each change.
- Use one @@ section per separate region; several files may go in one patch.
- To rename a file, Add the new path and Delete the old one.
- The result lists each file; if a section failed, re-read that file and send a corrected patch for it only."""


class PatchError(ValueError):
    """A patch the translator cannot turn into native tools; shown to the model."""


@dataclass
class Chunk:
    lines: list[tuple[str, str]] = field(default_factory=list)  # (" "|"-"|"+", text)


@dataclass
class FileOp:
    kind: str  # "add" | "delete" | "update"
    path: str
    add_lines: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


# ── tool definitions ──────────────────────────────────────────────────────────

def responses_tool(mode: str = "freeform") -> dict:
    """The Responses tool definition for ``apply_patch``."""
    if mode == "function":
        return {
            "type": "function",
            "name": APPLY_PATCH_TOOL,
            "description": APPLY_PATCH_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string", "description": "The full patch text."}},
                "required": ["input"],
                "additionalProperties": False,
            },
        }
    return {
        "type": "custom",
        "name": APPLY_PATCH_TOOL,
        "description": APPLY_PATCH_DESCRIPTION,
        "format": {"type": "grammar", "syntax": "lark", "definition": APPLY_PATCH_GRAMMAR},
    }


def parse_mode(value: object) -> str | None:
    """Header/config value → ``freeform`` | ``function`` | None (off)."""
    if value is True:
        return "freeform"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "on", "freeform"}:
            return "freeform"
        if v == "function":
            return "function"
    return None


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_patch(text: str) -> list[FileOp]:
    """Parse Codex patch text, tolerating fences, CRLF and missing markers."""
    if not isinstance(text, str):
        raise PatchError("apply_patch input must be patch text.")
    lines = text.replace("\r\n", "\n").split("\n")
    # Strip a surrounding markdown fence and blank edges.
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[0].lstrip().startswith("```"):
        lines.pop(0)
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
    if lines and lines[0].strip() == "*** Begin Patch":
        lines.pop(0)
    if lines and lines[-1].strip() == "*** End Patch":
        lines.pop()
    if any(line.strip() in {"*** Begin Patch", "*** End Patch"} for line in lines):
        raise PatchError("Send exactly one patch between *** Begin Patch and *** End Patch.")

    ops: list[FileOp] = []
    cur: FileOp | None = None
    for raw in lines:
        if raw.startswith("*** Add File: "):
            cur = FileOp("add", _path(raw, "*** Add File: "))
            ops.append(cur)
            continue
        if raw.startswith("*** Delete File: "):
            ops.append(FileOp("delete", _path(raw, "*** Delete File: ")))
            cur = None
            continue
        if raw.startswith("*** Update File: "):
            cur = FileOp("update", _path(raw, "*** Update File: "))
            ops.append(cur)
            continue
        if raw.startswith("*** Move to: "):
            raise PatchError(
                "Move to is not supported. Add the file at the new path and Delete the old one."
            )
        if raw.strip() == "*** End of File":
            continue
        if cur is None:
            if not raw.strip():
                continue
            raise PatchError(f"Unexpected line outside a file section: {raw[:80]!r}")
        if cur.kind == "add":
            if raw.startswith("+"):
                cur.add_lines.append(raw[1:])
            elif not raw:
                cur.add_lines.append("")  # tolerate a bare blank line
            else:
                raise PatchError(f"Add File lines must start with '+' ({cur.path}): {raw[:80]!r}")
            continue
        # update
        if raw.startswith("@@"):
            cur.chunks.append(Chunk())
            continue
        if not cur.chunks:
            cur.chunks.append(Chunk())
        if raw[:1] in (" ", "-", "+"):
            cur.chunks[-1].lines.append((raw[0], raw[1:]))
        elif not raw:
            cur.chunks[-1].lines.append((" ", ""))  # blank context line without its space
        else:
            raise PatchError(
                f"Update lines must start with ' ', '-' or '+' ({cur.path}): {raw[:80]!r}"
            )
    if not ops:
        raise PatchError("The patch contains no file sections.")
    for op in ops:
        if op.kind == "update":
            op.chunks = [c for c in op.chunks if any(k != " " for k, _ in c.lines)]
            if not op.chunks:
                raise PatchError(f"Update File {op.path} has no + or - lines.")
    return ops


def _path(raw: str, prefix: str) -> str:
    path = raw[len(prefix):].strip()
    if not path:
        raise PatchError(f"Missing path after {prefix.strip()!r}.")
    return path


# ── patch → native tool_use blocks ───────────────────────────────────────────

def resolve_path(path: str, cwd: str | None) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    if not cwd:
        raise PatchError(f"Use an absolute path (got {path!r}).")
    return os.path.normpath(os.path.join(cwd, path))


def _ops_to_inputs(ops: list[FileOp], cwd: str | None) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for op in ops:
        path = resolve_path(op.path, cwd)
        if op.kind == "add":
            out.append(("Write", {"file_path": path, "content": "\n".join(op.add_lines) + "\n"}))
        elif op.kind == "delete":
            out.append(("Bash", {
                "command": f"rm -- {shlex.quote(path)}",
                "description": f"Delete {os.path.basename(path)}",
            }))
        else:
            for chunk in op.chunks:
                old = "\n".join(t for k, t in chunk.lines if k != "+")
                new = "\n".join(t for k, t in chunk.lines if k != "-")
                if not old.strip():
                    raise PatchError(
                        f"A section of {op.path} has no context or removed lines; "
                        "include unchanged lines around the insertion so it can be located."
                    )
                out.append(("Edit", {"file_path": path, "old_string": old, "new_string": new}))
    return out


def error_tool_use(message: str) -> tuple[str, dict]:
    """A Bash call whose output is the error, so the model sees it as the tool result."""
    text = f"apply_patch error: {message}"
    return ("Bash", {
        "command": f"printf '%s\\n' {shlex.quote(text)} >&2; exit 1",
        "description": "apply_patch: invalid patch",
    })


def patch_to_tool_uses(call_id: str, text: str, cwd: str | None) -> list[tuple[str, str, dict]]:
    """Patch text → [(tool_use_id, name, input)], or one error block."""
    try:
        inputs = _ops_to_inputs(parse_patch(text), cwd)
    except PatchError as exc:
        name, inp = error_tool_use(str(exc))
        return [(f"{call_id}__aperr", name, inp)]
    n = len(inputs)
    return [(f"{call_id}__ap{i}of{n}", name, inp) for i, (name, inp) in enumerate(inputs)]


def patch_from_function_arguments(raw: str) -> str:
    """Function-mode fallback: arguments are ``{"input": "<patch>"}``."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
        return parsed["input"]
    return raw


# ── history: native blocks → one apply_patch call/output ─────────────────────

def group_call_id(tool_use_id: str) -> str | None:
    m = _ID_RE.match(tool_use_id or "")
    return m.group("call") if m else None


def _lines(s: str) -> list[str]:
    s = s or ""
    if s.endswith("\n"):
        s = s[:-1]
    return s.split("\n") if s else []


def _edit_chunk(old: str, new: str) -> list[str]:
    a, b = (old or "").split("\n"), (new or "").split("\n")
    pre = 0
    while pre < min(len(a), len(b)) and a[pre] == b[pre]:
        pre += 1
    suf = 0
    while suf < min(len(a), len(b)) - pre and a[-1 - suf] == b[-1 - suf]:
        suf += 1
    out = ["@@"]
    out += [" " + t for t in a[:pre]]
    out += ["-" + t for t in a[pre:len(a) - suf]]
    out += ["+" + t for t in b[pre:len(b) - suf]]
    out += [" " + t for t in a[len(a) - suf:]]
    return out


def regenerate_patch(blocks: list[dict]) -> str:
    """Native tool_use blocks (Edit/Write/MultiEdit/Bash rm) → patch text."""
    body: list[str] = []
    last_update: str | None = None
    for block in blocks:
        name = block.get("name")
        inp = block.get("input") if isinstance(block.get("input"), dict) else {}
        path = str(inp.get("file_path") or "")
        if name == "Edit":
            if last_update != path:
                body.append(f"*** Update File: {path}")
                last_update = path
            body += _edit_chunk(str(inp.get("old_string") or ""), str(inp.get("new_string") or ""))
            continue
        if name == "MultiEdit":
            if last_update != path:
                body.append(f"*** Update File: {path}")
                last_update = path
            for e in inp.get("edits") or []:
                if isinstance(e, dict):
                    body += _edit_chunk(str(e.get("old_string") or ""), str(e.get("new_string") or ""))
            continue
        last_update = None
        if name == "Write":
            body.append(f"*** Add File: {path}")
            body += ["+" + t for t in _lines(str(inp.get("content") or ""))]
        elif name == "Bash":
            cmd = str(inp.get("command") or "")
            if cmd.startswith("rm -- "):
                try:
                    target = shlex.split(cmd[len("rm -- "):])[0]
                except (ValueError, IndexError):
                    target = cmd[len("rm -- "):]
                body.append(f"*** Delete File: {target}")
            # error blocks contribute no file section
    return "\n".join(["*** Begin Patch", *body, "*** End Patch"])


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text") or "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _label(block: dict) -> str:
    name = block.get("name")
    inp = block.get("input") if isinstance(block.get("input"), dict) else {}
    if name == "Write":
        return f"A {inp.get('file_path')}"
    if name == "Bash":
        cmd = str(inp.get("command") or "")
        if cmd.startswith("rm -- "):
            return f"D {regenerate_patch([block]).splitlines()[1][len('*** Delete File: '):]}"
        return "patch"
    return f"M {inp.get('file_path')}"


def fold_results(blocks: list[dict], results: dict[str, dict]) -> str:
    """Per-block tool_results → one apply_patch output string."""
    entries: list[tuple[str, bool, str]] = []
    for block in blocks:
        res = results.get(block.get("id") or "") or {}
        text = _result_text(res.get("content")).strip()
        failed = bool(res.get("is_error")) or "<tool_use_error>" in text or not res
        if not res:
            text = "no result returned"
        entries.append((_label(block), failed, text))
    if len(entries) == 1 and entries[0][0] == "patch":
        return entries[0][2]
    if not any(failed for _, failed, _ in entries):
        seen: list[str] = []
        for label, _, _ in entries:
            if label not in seen:
                seen.append(label)
        return "Success. Updated the following files:\n" + "\n".join(seen)
    out = ["Patch partly applied. Sections run in order; a failed section changed nothing."]
    for label, failed, text in entries:
        if failed:
            out.append(f"FAILED {label}: {text[:2000]}")
        else:
            out.append(f"ok {label}")
    out.append("Re-read any failed file and send a corrected patch for those sections only.")
    return "\n".join(out)


# ── cwd discovery ─────────────────────────────────────────────────────────────

_CWD_RE = re.compile(r"Primary working directory:\s*(/[^\n(]*?)\s*(?:\(was [^)]*\))?\s*$", re.M)


def find_cwd(body: dict) -> str | None:
    """Latest ``Primary working directory`` the CLI reported in system/messages."""
    texts: list[str] = []

    def collect(content: Any) -> None:
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    if isinstance(b.get("text"), str):
                        texts.append(b["text"])
                    elif b.get("type") == "tool_result":
                        continue

    collect(body.get("system"))
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            collect(msg.get("content"))
    cwd: str | None = None
    for text in texts:
        for m in _CWD_RE.finditer(text):
            cwd = m.group(1).strip()
    return cwd


# ── history rewrite (Anthropic messages, before Responses translation) ───────

PATCH_INPUT_KEY = "__apply_patch__"


def rewrite_history(messages: list) -> tuple[list, set[str]]:
    """Fold translated (and legacy) edit blocks into apply_patch call/result pairs.

    Returns new messages (input not mutated) and the set of apply_patch call ids.
    Assistant blocks become one ``tool_use`` named ``apply_patch`` whose input
    carries the regenerated patch under :data:`PATCH_INPUT_KEY`; the matching
    user ``tool_result`` blocks become one result with the folded output.
    """
    groups: dict[str, list[dict]] = {}
    member: dict[str, str] = {}
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            bid = block.get("id") or ""
            gid = group_call_id(bid)
            if gid is None and block.get("name") in REPLACED_TOOLS:
                gid = bid  # legacy native edit: its own one-block patch
            if gid:
                groups.setdefault(gid, []).append(block)
                member[bid] = gid
    if not groups:
        return messages, set()

    out: list = []
    for msg in messages:
        if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
            out.append(msg)
            continue
        role = msg.get("role")
        new_content: list = []
        if role == "assistant":
            emitted: set[str] = set()
            for block in msg["content"]:
                bid = block.get("id") if isinstance(block, dict) else None
                gid = member.get(bid or "") if isinstance(block, dict) and block.get("type") == "tool_use" else None
                if gid is None:
                    new_content.append(block)
                    continue
                if gid in emitted:
                    continue
                emitted.add(gid)
                new_content.append({
                    "type": "tool_use", "id": gid, "name": APPLY_PATCH_TOOL,
                    "input": {PATCH_INPUT_KEY: regenerate_patch(groups[gid])},
                })
        elif role == "user":
            pending: dict[str, dict[str, dict]] = {}
            order: list[tuple[str, Any]] = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    gid = member.get(block.get("tool_use_id") or "")
                    if gid is not None:
                        if gid not in pending:
                            pending[gid] = {}
                            order.append(("group", gid))
                        pending[gid][block.get("tool_use_id")] = block
                        continue
                order.append(("block", block))
            for kind, value in order:
                if kind == "block":
                    new_content.append(value)
                else:
                    new_content.append({
                        "type": "tool_result", "tool_use_id": value,
                        "content": fold_results(groups[value], pending[value]),
                    })
        else:
            new_content = msg["content"]
        out.append({**msg, "content": new_content})
    return out, set(groups)
