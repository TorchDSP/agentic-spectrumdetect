# __main__.py
import asyncio
import argparse
import sys
from agent.agent import AgentRunner
from agent.settings import config

async def agent_cli():
    mcp_server = f"{config.SPECTRUM_SERVER_MCP_SERVER}{config.SPECTRUM_SERVER_MCP_ROUTE}"
    agent_runner = AgentRunner(
        spectrum_server_mcp=mcp_server,
        llm_api_key=config.LLM_API_KEY,
        llm_api=config.LLM_API,
        llm_model=config.LLM_MODEL,
        llm_reasoning=config.REASONING_LEVEL,
        system_prompt=config.SYSTEM_PROMPT,
        otel_exporter_otlp_endpoint=config.OTEL_EXPORTER_OTLP_ENDPOINT,
        ca_chain=config.CA_CHAIN
    )
    await agent_runner.run()

def agent_server():
    import uvicorn
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8001, reload=False)

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run either agent as a cli or as a webserver.",
        prog=argv[0] if argv else None,
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run the agent in server mode.",
    )
    parser.add_argument(
        "-m",
        help="module, do nothing"
        )
    return parser.parse_args(argv)

def main_entry(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.server:
        agent_server()
    else:
        asyncio.run(agent_cli())

if __name__ == "__main__":
    print(sys.argv) 
    main_entry(sys.argv[1:])
