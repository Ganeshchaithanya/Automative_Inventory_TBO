"""Read-only MCP server for public event discovery."""

import anyio
from mcp import ClientSession, types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.memory import create_client_server_memory_streams

from ascis.research import discover


SEARCH = types.Tool(
    name="search_public_events",
    description="Find public event evidence and return source links.",
    input_schema={
        "type": "object", "properties": {"query": {"type": "string", "minLength": 3, "maxLength": 160}},
        "required": ["query"], "additionalProperties": False,
    },
)


async def list_tools(context, params):
    return types.ListToolsResult(tools=[SEARCH])


async def call_tool(context, params):
    if params.name != SEARCH.name:
        return types.CallToolResult(is_error=True, content=[types.TextContent(type="text", text="Unknown tool.")])
    query = (params.arguments or {}).get("query")
    if not isinstance(query, str) or not 3 <= len(query.strip()) <= 160:
        return types.CallToolResult(is_error=True, content=[types.TextContent(type="text", text="Invalid query.")])
    result = await anyio.to_thread.run_sync(discover, query)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=f"Found {len(result['sources'])} source links.")],
        structured_content=result,
    )


server = Server("ASCIS public research", on_list_tools=list_tools, on_call_tool=call_tool)


async def request_sources(query: str) -> dict:
    """Use MCP messages over a local transport without opening a public endpoint."""
    async with create_client_server_memory_streams() as (client, provider):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(server.run, *provider, server.create_initialization_options())
            async with ClientSession(*client) as session:
                await session.initialize()
                result = await session.call_tool(SEARCH.name, {"query": query})
            task_group.cancel_scope.cancel()
    if result.is_error or result.structured_content is None:
        raise RuntimeError("The read-only MCP search returned an error.")
    return result.structured_content


async def serve():
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(serve)
