# Agentic SpectrumDetect Application

A **FastAPI** based chat application that uses a configurable LLM (via OpenAI‑compatible API) and a custom MCP (Message‑Control‑Protocol) toolset to provide an interactive, AI‑driven spectrum analysis assistant.

The repository contains two ways to run the application:

* **CLI mode** – a terminal based chat interface.
* **Server mode** – a FastAPI web server exposing a simple chat UI and a JSON streaming API.

Both modes share the same core logic implemented in `agent/src/agent/agent.py` and `agent/src/agent/server.py`.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Docker (recommended)](#docker-recommended)

---

## Features

- **LLM integration** – works with any OpenAI‑compatible endpoint (e.g., vLLM, OpenAI, Azure).
- **MCP toolset** – plug‑in toolsets via `pydantic_ai.mcp` for custom server‑side actions.
- **Rich console UI** – colourful, markdown‑aware terminal chat.
- **Web UI** – minimal HTML/JS front‑end served by FastAPI.
- **Streaming responses** – newline‑delimited JSON for incremental updates.
- **Configurable via environment variables** – all settings are defined in `agent/src/agent/settings.py`.

---

## Prerequisites

- **Docker** (>= 20.10) – for the containerised workflow.
- **Python 3.12** – required for local development.
- **uv** (optional) – a fast Python package manager used in the Dockerfile.
- An **LLM endpoint** reachable from the container/host (e.g., `http://vllm:8888/v1`).

---

## Installation

### Docker (recommended)

The repository already ships a multi‑stage Dockerfile.

---

## License

Agentic Spectrumdetect is released under the MIT License. The MIT license is a popular open-source software license enabling free use, redistribution, and modifications, even for commercial purposes, provided the license is included in all copies or substantial portions of the software. Agentic Spectrumdetect has no connection to MIT, other than through the use of this license.

---

