"""Which tool results the bridges persist as chat images (TASK-2642)."""

from agent_bridge.tool_images import ToolImageMixin


def test_image_result_tools_include_tasks_inspection():
    for name in (
        "mcp__bawthub__inspect_bawthub_tasks",
        "inspect_bawthub_tasks",
        "mcp__playwright__browser_take_screenshot",
        "mcp__bawthub__generate_image",
    ):
        assert ToolImageMixin._is_image_result_tool(name), name


def test_other_tools_are_not_persisted_as_images():
    for name in (None, "", "mcp__bawthub__tasks_get", "mcp__playwright__browser_snapshot"):
        assert not ToolImageMixin._is_image_result_tool(name), name
