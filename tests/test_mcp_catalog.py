"""Catalog size, semantic compatibility, skill discovery and dispatch contracts."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import tiktoken
from mcp.server.fastmcp import FastMCP

from llm_bawt.mcp_server.approval_interceptor import ApprovalAwareFastMCP
from llm_bawt.mcp_server.catalog import CatalogFastMCP, compact_schema
from llm_bawt.mcp_server.catalog_contracts import TOOL_SUMMARIES, tool_reference
from llm_bawt.mcp_server.server import mcp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_catalog_covers_every_registered_tool_without_changing_validation():
    original = await FastMCP.list_tools(mcp)
    compact = await mcp.list_tools()
    assert [tool.name for tool in original] == [tool.name for tool in compact]
    assert set(TOOL_SUMMARIES) == {tool.name for tool in original}
    for before, after in zip(original, compact, strict=True):
        assert after.inputSchema == compact_schema(before.inputSchema)
        # Everything outside the two presentation fields stays identical.
        excluded = {"description", "inputSchema"}
        assert before.model_dump(exclude=excluded) == after.model_dump(exclude=excluded)
        assert after.description.endswith(f"Details: {tool_reference(after.name)}.")
    # Reading discovery must not mutate FastMCP's actual validator schemas.
    assert original == await FastMCP.list_tools(mcp)
    # TASK-1222 intentionally wraps both entry points. Preserve the approval
    # superclass, and prove delegation below rather than requiring method identity.
    assert CatalogFastMCP.__bases__ == (ApprovalAwareFastMCP,)


@pytest.mark.anyio
async def test_authorization_adapters_still_delegate_both_approval_entry_points(monkeypatch):
    calls = []

    async def ordinary(self, name, arguments):
        calls.append(("ordinary", name, arguments))
        return "ordinary-result"

    async def approved(self, name, arguments, **kwargs):
        calls.append(("approved", name, arguments, kwargs))
        return "approved-result"

    monkeypatch.setattr(ApprovalAwareFastMCP, "call_tool", ordinary)
    monkeypatch.setattr(ApprovalAwareFastMCP, "call_approved_tool", approved)
    server = CatalogFastMCP("contract")
    assert await server.call_tool("unregistered", {"x": 1}) == "ordinary-result"
    assert await server.call_approved_tool("unregistered", {"x": 2}, expected_invocation_hash="h") == "approved-result"
    assert calls == [("ordinary", "unregistered", {"x": 1}),
                     ("approved", "unregistered", {"x": 2}, {"expected_invocation_hash": "h"})]


def test_schema_compaction_preserves_title_fields_and_arbitrary_data():
    schema = {
        "title": "Root generated title", "type": "object",
        "properties": {
            "title": {"title": "Title annotation", "type": "string", "minLength": 1},
            "payload": {"type": "object", "default": {"title": "KEEP"},
                        "const": {"title": "KEEP"}, "examples": [{"title": "KEEP"}]},
            "optional": {"anyOf": [{"title": "String", "type": "string"}, {"type": "null"}], "default": None},
            "ref": {"$ref": "#/$defs/Row"},
        },
        "required": ["title"], "additionalProperties": False,
        "$defs": {"Row": {"title": "Row", "type": "object", "properties": {"title": {"type": "string"}}}},
        "description": "KEEP semantic description",
    }
    before = deepcopy(schema)
    result = compact_schema(schema)
    assert schema == before
    assert "title" not in result
    assert "title" in result["properties"]
    assert result["properties"]["title"] == {"type": "string", "minLength": 1}
    assert result["properties"]["payload"] == before["properties"]["payload"]
    assert result["properties"]["optional"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert result["properties"]["optional"]["default"] is None
    assert result["properties"]["ref"] == {"$ref": "#/$defs/Row"}
    assert result["required"] == ["title"]
    assert result["additionalProperties"] is False
    assert "title" in result["$defs"]["Row"]["properties"]
    assert "title" not in result["$defs"]["Row"]
    assert result["description"] == schema["description"]


def test_nested_schema_keywords_and_data_are_distinct():
    schema = {
        "dependentSchemas": {"title": {"title": "annotation", "properties": {"title": {"type": "string"}}}},
        "patternProperties": {"title": {"title": "annotation", "type": "string"}},
        "if": {"title": "annotation", "const": {"title": "data"}},
        "then": {"not": {"title": "annotation", "enum": [{"title": "data"}]}},
        "items": [{"title": "annotation", "type": "string"}],
        "prefixItems": [{"title": "annotation", "type": "integer"}],
    }
    result = compact_schema(schema)
    assert "title" in result["dependentSchemas"]
    assert "title" not in result["dependentSchemas"]["title"]
    assert result["patternProperties"]["title"] == {"type": "string"}
    assert result["if"] == {"const": {"title": "data"}}
    assert result["then"]["not"] == {"enum": [{"title": "data"}]}
    assert result["items"] == [{"type": "string"}]
    assert result["prefixItems"] == [{"type": "integer"}]


@pytest.mark.anyio
async def test_catalog_token_budgets():
    encoder = tiktoken.get_encoding("o200k_base")
    tools = await mcp.list_tools()
    descriptions = [len(encoder.encode(tool.description or "")) for tool in tools]
    assert max(descriptions) <= 100
    # x_counts + x_search relevancy/operator hints: measured 3,251 (was <=3,200).
    # TASK-1147 adds two review tools: measured 3,380 description tokens.
    # TASK-1210: isolated recall + missing TASK-1202 routing contract reconciled.
    # Measured 100 tools, 3,482 description tokens (warnings retained).
    # TASK-1239: compact bots_deliveries_list filter contract: 3,514 (+32).
    # TASK-1233: ops_quiesce_ack (held turns answer a quiesce): 3,545 (+31).
    # TASK-1281: working_memory_list/create/update (CAS, idempotency and cap
    # contract stated in-band so agents don't need the reference): 3,680 (+135).
    # TASK-1323: messages_find (cards, facets, handle paging, status set in-band
    # so agents don't need the reference to page or narrow): 3,734 (+54).
    # TASK-1368: Nick's BLOCKED rule in-band on tasks_create, pointer on
    # tasks_update (misused BLOCKED pages Nick): 3,797 (+63).
    # TASK-1374: visually confirmable REVIEW vs coordinator-verified closure:
    # 3,819 (+22).
    # TASK-1380: tasks_list names the user's focus list and where to change it:
    # 3,834 (+15). tasks_update stays at 97; its focus args are typed.
    # TASK-1380: tasks_list names archived=false (tasks_list now exactly 100):
    # 3,842 (+8).
    # TASK-1400: bots_list_available names its live state and tag filter:
    # 3,860 (+18). One call answers "who is busy/queued/held" for every bot.
    # TASK-1431: bots_send_message names steer_only and keyed targets fan-out:
    # 3,874 (+14). Both change what one call may do, so they stay in-band.
    # TASK-1460: tasks_board (the board's lanes as /tasks/board draws them):
    # 3,911 (+37).
    # TASK-1318: self_system_prompt says pinned sessions (Claude Code, Codex)
    # get edits at the next boundary, unpinned bots next turn: 3,914 (+3).
    # TASK-1323: messages_context (both anchor forms, offset and cursors
    # in-band; replaces hand-written session SQL and get_by_id windows):
    # 3,974 (+60).
    # TASK-1521: bots_profile_patch (CAS, private refusal and rollback stated
    # in-band; replaces hand-written curl PATCHes of other bots): 4,024 (+50).
    # TASK-1521: bots_profile_patch names the own-tags refusal: 4,027 (+3).
    # TASK-1526: the image attach policy and opt-out on tasks_create, the
    # linked turn's images on tasks_update: 4,058 (+31). Bots attach what
    # they were shown instead of each task bot re-capturing it.
    # TASK-1555: the bulky lookups (tasks_list, tasks_search_semantic,
    # messages_search_all, memory_search_all) name the cheaper tool first and
    # tasks_board says it is where active-work lookups start: 4,125 (+67).
    # One avoided default tasks_list call (~13k chars) pays for it.
    # TASK-1557: context_recall names its new default (tasks) and the two
    # opt-in depths, task_details and tool_args: 4,136 (+11). Without them a
    # caller can't know reviews, steps and clone facts are searchable.
    assert sum(descriptions) <= 4136, f"Descriptions grew to {sum(descriptions)} tokens"
    definitions = [{"name": f"mcp__bawthub__{t.name}", "description": t.description or "", "input_schema": t.inputSchema} for t in tools]
    total = len(encoder.encode(json.dumps(definitions, ensure_ascii=False, separators=(",", ":"))))
    # TASK-900: explicit X search adds ~160 tokens; measured catalog 11,239.
    # Keep its paid-call warning and typed time/pagination fields intact.
    # x_counts tool + x_search sort_order/include_authors: measured 11,425.
    # TASK-1047 self_model (+146 tokens incl. typed action enum): measured 11,618.
    # TASK-1082: optional one-word session rotation key adds eight schema tokens.
    # Measured compact catalog: 11,626 (+8 over the previous 11,618).
    # TASK-1126: task/step/project id args say "ID (cuid)" instead of the wrong
    # "UUID" and accept TASK-N refs. Measured compact catalog: 11,645 (+19).
    # TASK-1147: review list/add typed schemas and attribution guidance: 11,945.
    # TASK-1188: accepted rotation pointer flag adds 12 typed-schema tokens.
    # TASK-1210: measured typed recall schema + existing routing registration.
    # TASK-1239: bots_deliveries_list bot_id/active_only/task_id/since_minutes/
    # include_chat filters: 12,520 (+116). One filtered call replaces paired
    # ~10k-token full-receipt calls, so the schema growth pays for itself.
    # TASK-1233: ops_quiesce_ack (status + optional note): 12,611 (+91).
    # TASK-1313: step status enum on steps_add/steps_set/tasks_create step
    # objects and steps_update.status: 12,683 (+72, ~18 per enum). Invalid
    # statuses used to reach BawtHub as HTTP 500s.
    # TASK-1281: working_memory_list/create/update with the design §6.1 typed
    # fields (status/clear enums, typed Ref objects, CAS revision): 13,451
    # (+768). Typed enums and Ref reject bad writes at the schema boundary.
    # TASK-1281: sessions_rotate include_working_memory flag (WORKING_MEMORY.md
    # 10.1): 13,474 (+23).
    # TASK-1323: messages_find (six optional typed args, speaker enum): 13,691
    # (+217). Replaces chained messages_get/search_all calls when locating.
    # TASK-1368: BLOCKED guidance + blocked_reason on tasks_create/update:
    # 13,796 (+105).
    # TASK-1374: visually confirmable REVIEW vs coordinator-verified closure:
    # 13,818 (+22).
    # TASK-1380: tasks_update focus/focus_position/focus_anchor (typed on/off
    # and placement enums, short field descriptions) + tasks_list focused:
    # 13,982 (+164). One chat call answers or changes the user's focus.
    # TASK-1380: typed archived flag on tasks_update and tasks_list: 14,046
    # (+64). Archiving the Done lane from chat is one call per task.
    # TASK-1400: bots_list_available optional tag + include_state args and the
    # live-state summary: 14,125 (+79). Replaces per-bot health/delivery probes.
    # TASK-1422: typed check_links on tasks_update (REVIEW's "Where to look"):
    # 14,234 (+109). Replaces free-text dev URLs and look steps in response.
    # TASK-1431: bots_send_message targets (selector object or bot list),
    # optional target_bot_id and the summary above: 14,281 (+47). One keyed
    # call replaces N single sends.
    # TASK-1460: tasks_board with the board URL's six filters and a row limit:
    # 14,468 (+187). One call reads the whole board, lanes and counts, in
    # place of a tasks_list call per status.
    # TASK-1249: bots_list_available optional capability filter: 14,499 (+31).
    # Finds a bot by what its role tags can do without naming a custom tag.
    # TASK-1318: self_system_prompt pinned-vs-unpinned edit timing: 14,502 (+3).
    # TASK-1323: messages_context (anchored neighbour window with offset
    # and owner-bound cursors): 14,756 (+254).
    # TASK-1521: bots_profile_patch (bot_id, changes, reason, expected):
    # 14,894 (+138). One audited CAS call replaces a GET + curl PATCH pair.
    # TASK-1526: image_urls / add_image_urls / remove_image_urls /
    # attach_turn_images on tasks_create and tasks_update, plus the attach
    # policy: 15,001 (+107). Task images reach every bot on the task instead of
    # each one re-capturing them with Playwright.
    # TASK-1521: bots_profile_patch summary names the own-tags refusal: 15,004 (+3).
    # TASK-1526: the attach policy moved into the tasks_create/tasks_update
    # summaries the model actually sees: 15,035 (+31).
    # TASK-1555: the description hints above, tasks_list detail flag, and
    # tasks_board's typed view/updated enums and rows scope: 15,170 (+135).
    # An unknown window used to come back as the default board, unannounced.
    # TASK-1557: context_recall's sources enum gains task_details and
    # tool_args, plus the summary above: 15,189 (+19).
    assert total <= 15189, f"Catalog grew to {total} tokens; see docs/MCP_TOOL_DESIGN.md"
    assert all("Args:" not in (tool.description or "") for tool in tools)


@pytest.mark.parametrize("name,words", [
    ("tasks_update", ("REVIEW", "COMPLETED", "BUG")),
    ("tasks_list", ("focused=true", "archived=false", "tasks_update edits both")),
    ("tasks_associate_current", ("actually", "mention", "fails closed")),
    ("steps_set", ("ENTIRE", "deleting", "[]")),
    ("tasks_delete", ("Permanently", "CANCELLED")),
    ("projects_delete", ("Permanently", "unassigned")),
    ("memory_clear", ("ALL", "Destructive")),
    ("messages_clear", ("ALL", "Destructive")),
    ("messages_restore_ignored", ("ALL",)),
    ("messages_remove_last_partial", ("does NOT verify partial",)),
    ("sessions_rotate", ("active thread",)),
    ("self_system_prompt", ("fully replace", "own")),
    ("self_model", ("persistently", "next turn", "when_idle")),
    ("self_tail", ("do not reprint",)),
    ("bots_send_message", ("async", "when_idle", "force never")),
    ("bots_delivery_cancel", ("QUEUED", "cannot abort")),
    ("ops_run", ("Approval/queued is not success", "Never bypass")),
    ("media_add", ("downloading", "first match")),
])
def test_call_critical_warnings_stay_inline(name, words):
    for word in words:
        assert word in TOOL_SUMMARIES[name]


# TASK-1555: a bulky lookup tool names the cheaper one to try first; one
# question once cost a bot ~48k tokens through these.
@pytest.mark.parametrize("name,cheaper", [
    ("tasks_list", ("tasks_board(q=)",)),
    ("tasks_search_semantic", ("tasks_board(q=)",)),
    ("messages_search_all", ("context_recall", "messages_find")),
    ("memory_search_all", ("memory_search_source",)),
    ("tasks_board", ("Start here for active work",)),
])
def test_bulky_lookups_name_the_cheaper_tool(name, cheaper):
    for hint in cheaper:
        assert hint in TOOL_SUMMARIES[name]


def test_tool_results_are_compact_json():
    from mcp.types import ImageContent, TextContent

    from llm_bawt.mcp_server.catalog import compact_call_result

    pretty = TextContent(type="text", text=json.dumps({"a": [1, {"b": "é"}]}, indent=2, ensure_ascii=False))
    image = ImageContent(type="image", data="AA==", mimeType="image/png")
    plain = TextContent(type="text", text="{not json")

    [compact, untouched_image, untouched_text] = compact_call_result([pretty, image, plain])
    assert compact.text == '{"a":[1,{"b":"é"}]}'
    assert json.loads(compact.text) == json.loads(pretty.text)
    assert (untouched_image, untouched_text) == (image, plain)

    blocks, structured = compact_call_result(([pretty], {"result": 1}))
    assert blocks[0].text == compact.text and structured == {"result": 1}
    assert compact_call_result({"error": "denied"}) == {"error": "denied"}


@pytest.mark.anyio
async def test_call_tool_returns_compact_json(monkeypatch):
    # FastMCP renders a dict result with indent=2; the catalog server re-emits it.
    from mcp.types import TextContent

    async def fastmcp_result(self, name, arguments):
        value = {"rows": [{"id": 1}, {"id": 2}]}
        return [TextContent(type="text", text=json.dumps(value, indent=2))], value

    async def allow(self, name, arguments):
        return None

    monkeypatch.setattr(ApprovalAwareFastMCP, "call_tool", fastmcp_result)
    server = CatalogFastMCP("compact-test")
    monkeypatch.setattr(type(server.bot_access), "preflight", allow)

    result = await server.call_tool("rows", {})
    blocks = result[0] if isinstance(result, tuple) else result
    assert blocks[0].text == '{"rows":[{"id":1},{"id":2}]}'


def test_task_update_closure_guidance_matches_visual_review_policy():
    from llm_bawt.mcp_server.task_tools import update_task

    for text in (TOOL_SUMMARIES["tasks_update"], update_task.__doc__):
        assert "REVIEW" in text and "COMPLETED" in text
        assert "UI/UX" in text and ("Snark or Rook" in text or "Snark/Rook" in text)
        assert "BUG" in text
        assert "Nick-requested" not in text
        assert "REVIEW by default" not in text


@pytest.mark.anyio
async def test_task_and_step_status_guidance_uses_canonical_values():
    from llm_bawt.mcp_server.task_step_types import STEP_STATUSES, TASK_STATUSES
    from llm_bawt.mcp_server.task_tools import list_tasks, update_task

    assert TASK_STATUSES == (
        "QUEUED", "PLANNING", "REFINED", "IN_PROGRESS", "REVIEW",
        "COMPLETED", "FAILED", "BUG", "CANCELLED", "DELETED", "BLOCKED",
    )
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    for status in TASK_STATUSES:
        assert status in tools["tasks_list"].description
        assert status in list_tasks.__doc__
        assert status in update_task.__doc__
    assert "excludes DELETED" in tools["tasks_list"].description
    for name in ("steps_add", "steps_update", "steps_set"):
        for status in STEP_STATUSES:
            assert status in tools[name].description
        assert "not IN_PROGRESS" in tools[name].description


@pytest.mark.anyio
async def test_blocked_definition_reaches_task_write_tools():
    # TASK-1368: Nick's BLOCKED definition rides every task-writing surface.
    from llm_bawt.mcp_server.task_step_types import BLOCKED_DEFINITION
    from llm_bawt.mcp_server.task_tools import create_task, update_task

    tools = {tool.name: tool for tool in await mcp.list_tools()}
    for name, function in (("tasks_create", create_task), ("tasks_update", update_task)):
        description = tools[name].description
        assert "blocked_reason" in description and "what Nick must do/decide" in description
        assert ("Never nits" if name == "tasks_create" else "rules in tasks_create") in description
        assert BLOCKED_DEFINITION in function.__doc__
        assert "{blocked_definition}" not in function.__doc__
        assert "blocked_reason" in tools[name].inputSchema["properties"]
    for clause in ("only Nick can make", "restart, deploy/ship, credentials", "against Nick's stated intent",
                   "bot-vs-bot disagreements (Gavel mediates)", "misusing BLOCKED is a process failure"):
        assert clause in BLOCKED_DEFINITION


@pytest.mark.anyio
async def test_unknown_tools_remain_discoverable_and_callables_unchanged():
    server = CatalogFastMCP("test-catalog")

    @server.tool(name="new_unmigrated_tool")
    async def sample(title: str = "hello") -> str:
        """An unconverted tool's description must not disappear."""
        return title

    tool = (await server.list_tools())[0]
    assert tool.description == "An unconverted tool's description must not disappear."
    assert tool.inputSchema["properties"]["title"]["default"] == "hello"
    # Isolated pure callable only; direct base dispatch tests unchanged binding.
    result = await FastMCP.call_tool(server, "new_unmigrated_tool", {"title": "preserved"})
    assert "preserved" in str(result)
    assert await sample() == "hello"


@pytest.mark.anyio
async def test_real_mutation_still_passes_through_policy_gate(monkeypatch):
    import llm_bawt.mcp_server.approval_interceptor as interceptor
    from agent_bridge.approval import PolicyAction

    store = SimpleNamespace(compile_bundle=lambda: SimpleNamespace(policies=[]), record_decision=lambda **kw: None)
    server = CatalogFastMCP("test-denial", approval_store_provider=lambda: store)
    called = []

    @server.tool(name="tasks_delete")
    async def delete(task_id: str):
        called.append(task_id)
        return {"ok": True}

    monkeypatch.setattr(interceptor, "evaluate", lambda *args: SimpleNamespace(action=PolicyAction.DENY, subject="test"))
    await server.list_tools()
    result = await server.call_tool("tasks_delete", {"task_id": "test-task"})
    assert result["status"] == "denied"
    assert called == []


@pytest.mark.anyio
async def test_mcp_protocol_advertises_compact_catalog():
    from mcp.shared.memory import create_connected_server_and_client_session

    # Exercises the MCP handler bound during FastMCP initialization, not just
    # direct Python calls. No DB, model execution or external service involved.
    async with create_connected_server_and_client_session(mcp) as session:
        result = await session.list_tools()
    assert len(result.tools) == len(TOOL_SUMMARIES)
    task = next(t for t in result.tools if t.name == "tasks_create")
    assert "Details: bawthub-mcp/" in task.description
    assert "title" in task.inputSchema["properties"]
    assert "title" not in task.inputSchema


# The bawthub-mcp skill publishes exactly these pages. Its own check
# (agent-skills scripts/test_bawthub_mcp_references.py) pins the same set, so a
# rename on either side fails one of the two suites.
PUBLISHED_REFERENCES = frozenset({
    "bawthub-mcp/references/tasks.md",
    "bawthub-mcp/references/memory-history.md",
    "bawthub-mcp/references/sessions-delivery.md",
    "bawthub-mcp/references/ops-media.md",
})


def test_every_tool_points_at_a_published_skill_reference():
    used = {tool_reference(name) for name in TOOL_SUMMARIES}
    # No pointer to an unpublished page, and no published page left unused.
    assert used == PUBLISHED_REFERENCES
