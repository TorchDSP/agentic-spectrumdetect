# agent.py
import os
import ssl
import httpx
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from rich.live import Live
from rich.markdown import Markdown
from rich.console import Console
from rich.text import Text
from rich.panel import Panel


class AgentRunner:
    def __init__(
        self,
        spectrum_server_mcp: str,
        llm_api: str,
        llm_api_key: str,
        llm_model: str,
        llm_reasoning: str,
        system_prompt: str,
        otel_exporter_otlp_endpoint: str,
        ca_chain: str | None = None,
    ):
        self.llm_key = llm_api_key
        self.mcp_server = spectrum_server_mcp
        self.llm_server_hostname = llm_api
        self.llm_model = llm_model
        self.llm_reasoning_level = llm_reasoning
        self.system_prompt = system_prompt
        self.http_client: httpx.AsyncClient | None = None
        if ca_chain:
            self.ssl_ctx = ssl.create_default_context(cafile=ca_chain)
            self.http_client = httpx.AsyncClient(
                verify=self.ssl_ctx,
                timeout=httpx.Timeout(600.0),
            )
        if otel_exporter_otlp_endpoint:
            try:
                import logfire
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_exporter_otlp_endpoint
                logfire.configure(send_to_logfire=False)
                logfire.instrument_pydantic_ai()
                logfire.instrument_httpx(capture_all=True)
            except:
                pass

        self.console = Console()

    async def run(self):
        server = MCPServerStreamableHTTP(url=self.mcp_server)
        self.console.print("[white]Connecting to MCP server...[/white]")
        async with server:
            self.console.print(
                "[bold white]Connection successful. Initializing agent...[/bold white]"
            )
            message_history = list()
            local_provider = OpenAIProvider(
                base_url=self.llm_server_hostname,
                api_key=self.llm_key,
                http_client=self.http_client,
            )
            local_model = OpenAIModel(
                model_name=self.llm_model,
                provider=local_provider,
            )
            model_settings = OpenAIResponsesModelSettings(openai_reasoning_effort=self.llm_reasoning_level,openai_reasoning_summary='concise')
            local_model2 = OpenAIResponsesModel(
                model_name=self.llm_model,
                provider=local_provider,
            )
            agent = Agent(
                model=local_model, model_settings=model_settings, toolsets=[server], system_prompt=self.system_prompt
            )

            self.console.print(
                "[bold green]Agent is ready. Type 'exit' or 'quit' to end.[/bold green]"
            )
            while True:
                try:
                    prompt = self.console.input("[bold blue]You:[/bold blue] ")
                    if prompt.lower() in ["exit", "quit"]:
                        break
                    if not prompt:
                        continue
                    full_response_text = Text("")
                    result= await agent.run(user_prompt=prompt, message_history=message_history)
                    self.console.log(f"Usage: {result.usage()}")
                    self.console.log(Markdown(result.output))
                    message_history = result.all_messages()
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
            self.console.print("\n[bold red]Exiting chat. Goodbye![/bold red]")
