"""Tool registry, built-in assistant tools, and safe tool-call execution."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from config import Settings, sanitize_error_message

MAX_SEARCH_QUERY_CHARS = 400


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata and Python callable for an LLM-accessible tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]


class ToolRegistry:
    """Registry that exposes OpenAI-compatible schemas and executes validated tool calls."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        function: Callable[..., Any],
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"The tool '{name}' is already registered.")
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=function,
        )

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        raw_arguments: str | dict[str, Any] | None,
        turn_log: dict[str, Any],
        *,
        secrets: tuple[str, ...] = (),
    ) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        tool = self._tools[name]
        if raw_arguments is None or raw_arguments == "":
            arguments: dict[str, Any] = {}
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        elif isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON arguments for tool '{name}': {error}") from error
            if not isinstance(parsed, dict):
                raise ValueError("Tool arguments must be a JSON object.")
            arguments = parsed
        else:
            raise ValueError("Tool arguments must be a JSON object or string.")

        allowed_props = set(tool.parameters.get("properties", {}).keys())
        required_props = set(tool.parameters.get("required", []))
        missing = required_props - set(arguments.keys())
        if missing:
            raise ValueError(
                f"Missing required argument(s) for '{name}': {', '.join(sorted(missing))}"
            )
        unexpected = set(arguments.keys()) - allowed_props
        if unexpected and tool.parameters.get("additionalProperties", True) is False:
            raise ValueError(
                f"Unexpected argument(s) for '{name}': {', '.join(sorted(unexpected))}"
            )
        filtered_args = {key: value for key, value in arguments.items() if key in allowed_props}

        turn_log["events"].append(
            {
                "type": "tool_call",
                "name": name,
                "arguments": copy.deepcopy(filtered_args),
            }
        )

        try:
            result = tool.handler(**filtered_args)
        except Exception as error:
            cleaned = sanitize_error_message(str(error) or type(error).__name__, secrets)
            raise RuntimeError(cleaned) from error

        turn_log["events"].append(
            {
                "type": "tool_result",
                "name": name,
                "result": copy.deepcopy(result),
            }
        )
        return result


def get_current_datetime(timezone_name: str = "Asia/Karachi") -> dict[str, str]:
    """Return the current date, weekday, 12-hour time, and timezone label."""
    now = datetime.now(ZoneInfo(timezone_name))
    offset = now.strftime("%z")
    formatted_offset = (
        f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    )
    abbr = now.tzname() or timezone_name
    return {
        "date": now.strftime("%Y-%m-%d"),
        "day": now.strftime("%A"),
        "time": now.strftime("%I:%M:%S %p"),
        "timezone": f"{timezone_name} ({abbr}, UTC{formatted_offset})",
    }


def search_web(
    query: str,
    *,
    api_key: str | None,
    client: Any = None,
) -> dict[str, Any]:
    """Search the web via Tavily using a lazily initialized client."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")
    cleaned_query = query.strip()[:MAX_SEARCH_QUERY_CHARS]

    if client is None:
        if not api_key:
            raise RuntimeError("Web search is unavailable because TAVILY_API_KEY is not configured.")
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)

    response = client.search(
        query=cleaned_query,
        search_depth="basic",
        max_results=5,
    )

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score", 0),
        }
        for item in response.get("results", [])
    ]
    return {"query": cleaned_query, "results": results}


def create_tool_registry(settings: Settings) -> ToolRegistry:
    """Build the runtime ToolRegistry based on configured API keys and timezone."""
    registry = ToolRegistry()

    registry.register(
        name="get_current_datetime",
        function=lambda: get_current_datetime(settings.default_timezone),
        description=f"Get the current date, weekday, and time in {settings.default_timezone}.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    if settings.tavily_api_key:
        registry.register(
            name="search_web",
            function=lambda query: search_web(query, api_key=settings.tavily_api_key),
            description="Search the internet for current information and return relevant web results.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A clear search query.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    return registry
