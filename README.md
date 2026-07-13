# 🤖 Zen Agent

An AI agent with access to **1,000+ tools** via Composio, powered by the OpenCode API.

## Architecture

```
zen-agent/
├── core/                  # Core engine
│   ├── composio_client.py  # Composio REST API wrapper (v3/v3.1)
│   ├── llm_client.py       # OpenCode API client (OpenAI-compatible)
│   └── agent.py            # Agent orchestration (LLM + tools)
├── cli/                   # CLI interface
│   └── main.py             # Typer-based CLI
├── server/                # Web server
│   ├── main.py             # FastAPI app with REST + WebSocket APIs
│   └── static/             # Frontend dashboard
│       ├── index.html       # Chat UI
│       ├── style.css        # Dark theme
│       └── app.js           # WebSocket + REST client
├── tests/                 # Test suite
│   ├── test_composio_client.py
│   ├── test_llm_client.py
│   ├── test_agent.py
│   └── test_server.py
├── config.py              # Configuration
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API keys (or they default to the ones in config.py)
export OPENGATE_API_KEY="sk-..."
export COMPOSIO_API_KEY="ak-..."
```

### CLI Mode

```bash
# Interactive chat
python -m cli.main

# One-shot question
python -m cli.main --oneshot "What can you do?"

# Search for tools
python -m cli.main tools "manage github issues"

# Create a new session
python -m cli.main session --create
```

### Web Dashboard

```bash
python -m server.main
# Open http://localhost:8000
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/chat` | POST | Send a message to the agent |
| `/api/session/{user_id}` | GET | Get session info |
| `/api/session/{user_id}/reset` | POST | Reset conversation |
| `/api/tools/list` | GET | List Composio catalog tools |
| `/api/tools/search` | GET | Search tools in a session |
| `/ws/chat/{user_id}` | WebSocket | Streaming chat |
| `/` | GET | Dashboard UI |

## Composio Integration

The agent uses the **Composio v3/v3.1 REST API** directly (no SDK dependency):

- **Sessions**: Create, retrieve, reuse per user
- **Meta tools**: SEARCH, EXECUTE, MANAGE_CONNECTIONS, GET_TOOL_SCHEMAS, WORKBENCH, BASH
- **Tool execution**: Execute any of 23,000+ tools by slug
- **Sandbox**: Run Python code in a remote sandbox
- **Multi-execute**: Run several tools in parallel
- **Proxy**: Make HTTP requests through connected accounts

## Model

Uses `deepseek-v4-flash-free` via `https://opencode.ai/zen/v1` (OpenAI-compatible).
