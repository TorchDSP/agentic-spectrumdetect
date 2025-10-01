import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spectrum_server.api import radios_work
from fastapi import FastAPI
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount
from fastmcp.server.openapi import RouteMap, MCPType

fastapi_app = FastAPI()
fastapi_app.include_router(radios_work.router, prefix="/radio", tags=["radio"])


@fastapi_app.get("/health")
async def health():
    return "ok"

mcp = FastMCP.from_fastapi(app=fastapi_app, name="Spectrum MCP", route_maps=[RouteMap(pattern=r"^/health", mcp_type=MCPType.EXCLUDE)])
mcp_app = mcp.http_app()

app = Starlette(
    routes=[
        Mount("/llm", app=mcp_app),
        Mount("/", app=fastapi_app),
    ],
    lifespan=mcp_app.lifespan,
)
