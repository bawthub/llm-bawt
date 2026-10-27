"""Tests for HA native MCP integration."""

import pytest
from dataclasses import dataclass

from llm_bawt.integrations import ha_credentials
from llm_bawt.integrations.ha_mcp.client import (
    HomeAssistantNativeClient,
    HAToolDefinition,
)
from llm_bawt.tools.definitions import (
    Tool,
    ToolParameter,
    ha_tools_to_tool_definitions,
    get_tools_list,
)


@pytest.fixture(autouse=True)
def no_ha_record(monkeypatch):
    """No Integrations record: these tests cover the legacy settings path."""
    monkeypatch.setattr(ha_credentials, "read_connection", lambda *args, **kwargs: None)


@dataclass
class MockConfig:
    HA_NATIVE_MCP_URL: str = "http://hass.home:8123/api/mcp"
    HA_NATIVE_MCP_TOKEN: str = "test-token"
    HA_MCP_TIMEOUT: int = 10
    HA_MCP_TOOL_EXCLUDE: str = "GetDateTime,HassCancelAllTimers"


class TestHomeAssistantNativeClient:
    """Test the native HA MCP client."""

    def test_available_with_url_and_token(self):
        config = MockConfig()
        client = HomeAssistantNativeClient(config)
        assert client.available is True

    def test_not_available_without_url(self):
        config = MockConfig(HA_NATIVE_MCP_URL="")
        client = HomeAssistantNativeClient(config)
        assert client.available is False

    def test_not_available_without_token(self):
        config = MockConfig(HA_NATIVE_MCP_TOKEN="")
        client = HomeAssistantNativeClient(config)
        assert client.available is False

    def test_exclude_list_parsing(self):
        config = MockConfig(HA_MCP_TOOL_EXCLUDE="GetDateTime, HassBroadcast , ")
        client = HomeAssistantNativeClient(config)
        assert client._exclude_names == {"GetDateTime", "HassBroadcast"}

    def test_exclude_list_empty(self):
        config = MockConfig(HA_MCP_TOOL_EXCLUDE="")
        client = HomeAssistantNativeClient(config)
        assert client._exclude_names == set()

    def test_is_ha_tool_before_discovery(self):
        config = MockConfig()
        client = HomeAssistantNativeClient(config)
        assert client.is_ha_tool("HassTurnOn") is False
        assert client.initialized is False

    def test_is_ha_tool_after_manual_setup(self):
        config = MockConfig()
        client = HomeAssistantNativeClient(config)
        # Manually set tools as if discovered
        client._tools = [HAToolDefinition(name="HassTurnOn", description="Turn on")]
        client._tool_names = {"HassTurnOn"}
        client._initialized = True
        assert client.is_ha_tool("HassTurnOn") is True
        assert client.is_ha_tool("FakeTool") is False

    def test_tools_property_before_discovery(self):
        config = MockConfig()
        client = HomeAssistantNativeClient(config)
        assert client.tools == []
        assert client.tool_names == set()

    def test_call_tool_when_not_available(self):
        config = MockConfig(HA_NATIVE_MCP_URL="")
        client = HomeAssistantNativeClient(config)
        result = client.call_tool("HassTurnOn", {"name": "kitchen lights"})
        assert "Error" in result
        assert "not configured" in result

    def test_call_tool_when_not_initialized(self):
        config = MockConfig()
        client = HomeAssistantNativeClient(config)
        result = client.call_tool("HassTurnOn", {"name": "kitchen lights"})
        assert "Error" in result
        assert "not discovered" in result

    def test_call_tool_unknown_tool(self):
        config = MockConfig()
        client = HomeAssistantNativeClient(config)
        client._initialized = True
        client._tool_names = {"HassTurnOn"}
        result = client.call_tool("UnknownTool", {})
        assert "Error" in result
        assert "Unknown HA tool" in result


class TestHAToolConversion:
    """Test converting HA tool definitions to Tool dataclass."""

    def test_basic_conversion(self):
        ha_tools = [
            HAToolDefinition(
                name="HassTurnOn",
                description="Turns on a device",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "area": {"type": "string"},
                    },
                },
            ),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        assert len(tools) == 1
        assert tools[0].name == "HassTurnOn"
        assert tools[0].description == "Turns on a device"
        assert len(tools[0].parameters) == 2
        assert tools[0].parameters[0].name in ("name", "area")

    def test_enum_in_description(self):
        ha_tools = [
            HAToolDefinition(
                name="HassSetVolume",
                description="Sets volume",
                input_schema={
                    "type": "object",
                    "properties": {
                        "volume_level": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Volume percentage",
                        },
                    },
                },
            ),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        param = tools[0].parameters[0]
        assert "min=0" in param.description
        assert "max=100" in param.description

    def test_array_type_conversion(self):
        ha_tools = [
            HAToolDefinition(
                name="HassTurnOn",
                description="Turn on",
                input_schema={
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            ),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        assert tools[0].parameters[0].type == "array[string]"

    def test_empty_schema(self):
        ha_tools = [
            HAToolDefinition(name="GetDateTime", description="Get time", input_schema={}),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        assert len(tools) == 1
        assert len(tools[0].parameters) == 0

    def test_required_parameters(self):
        ha_tools = [
            HAToolDefinition(
                name="HassLightSet",
                description="Set light",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Light name"},
                        "brightness": {"type": "integer", "description": "Brightness 0-100"},
                    },
                    "required": ["name"],
                },
            ),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        params = {p.name: p for p in tools[0].parameters}
        assert params["name"].required is True
        assert params["brightness"].required is False

    def test_enum_values_in_description(self):
        ha_tools = [
            HAToolDefinition(
                name="HassMediaControl",
                description="Control media",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["play", "pause", "stop"],
                            "description": "Command to execute",
                        },
                    },
                },
            ),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        param = tools[0].parameters[0]
        assert "one of:" in param.description
        assert "play" in param.description

    def test_fallback_description(self):
        ha_tools = [
            HAToolDefinition(
                name="HassTurnOn",
                description="",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
            ),
        ]
        tools = ha_tools_to_tool_definitions(ha_tools)
        # Description defaults to "Home Assistant tool: <name>"
        assert "HassTurnOn" in tools[0].description


class TestToolListIntegration:
    """Test that HA native tools integrate into the tool list correctly."""

    def test_ha_native_tools_replace_home_tool(self):
        ha_tools = [
            Tool(name="HassTurnOn", description="Turn on", parameters=[]),
            Tool(name="HassTurnOff", description="Turn off", parameters=[]),
        ]
        tools = get_tools_list(
            include_home_tools=True,  # would normally include HOME_TOOL
            ha_native_tools=ha_tools,  # but native takes priority
        )
        tool_names = [t.name for t in tools]
        assert "HassTurnOn" in tool_names
        assert "HassTurnOff" in tool_names
        assert "home" not in tool_names  # legacy should NOT be present

    def test_legacy_home_when_no_native(self):
        tools = get_tools_list(include_home_tools=True, ha_native_tools=None)
        tool_names = [t.name for t in tools]
        assert "home" in tool_names

    def test_no_home_tools_at_all(self):
        tools = get_tools_list(include_home_tools=False, ha_native_tools=None)
        tool_names = [t.name for t in tools]
        assert "home" not in tool_names

    def test_ha_native_tools_without_include_home_flag(self):
        """Native tools should be added even if include_home_tools=False."""
        ha_tools = [
            Tool(name="HassTurnOn", description="Turn on", parameters=[]),
        ]
        tools = get_tools_list(include_home_tools=False, ha_native_tools=ha_tools)
        tool_names = [t.name for t in tools]
        assert "HassTurnOn" in tool_names
        assert "home" not in tool_names
