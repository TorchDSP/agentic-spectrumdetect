# Spectrum Server

**Spectrum Server** is a lightweight FastAPI application that provides a set of HTTP endpoints for interacting with a radio receiver.  
It offers functionality to:

* Retrieve the current UTC time.
* Wait for a configurable number of seconds.
* Tune the radio to a specific center frequency.
* Query recent measurement results stored in a MongoDB database.

The server is wrapped with **FastMCP**, exposing a Management Control Plane (MCP) for health‑checking and future extensions.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  

---

## Prerequisites

* Python 3.9+  
* `pip` (or `poetry`/`pipenv` if you prefer)  
* Access to a MongoDB instance (default port `27018`)  
* Network access to the radio receiver on UDP port `63333`  

---


## License

Agentic Spectrumdetect is released under the MIT License. The MIT license is a popular open-source software license enabling free use, redistribution, and modifications, even for commercial purposes, provided the license is included in all copies or substantial portions of the software. Agentic Spectrumdetect has no connection to MIT, other than through the use of this license.

---

