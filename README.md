# 🤖 Zen Agent ARM32

An **ARMv7/ARM32** compatible AI agent with access to **23,790+ tools** via Composio, powered by OpenCode AI API.

## ✨ Features

- **🧠 AI Agent** — Streaming chat with reasoning, tool calling, and code execution
- **🔧 23,790+ Tools** — GitHub, Gmail, Slack, Notion, Google Sheets, Linear, Jira, and more
- **💻 Web Dashboard** — Beautiful chat UI with dark/light theme, mobile responsive
- **⌨️ CLI Mode** — Interactive terminal chat with rich commands
- **🔌 REST API + WebSocket** — Full API for programmatic access
- **🐍 Code Sandbox** — Execute Python code remotely via Composio
- **🔄 Multi-Model Fallback** — Automatic fallback to secondary model on failure
- **💾 Conversation Persistence** — Save, load, and export conversations (JSON/Markdown)
- **📊 Token Usage Tracking** — Track input/output tokens and tool call count per session
- **🚦 Rate Limiting** — Configurable per-IP rate limiting for API endpoints
- **📝 Structured Logging** — JSON log format option for production deployments
- **🐳 Docker Support** — Dockerfile & docker-compose for containerized deployment
- **⚙️ Config Validation** — Startup warnings for missing or misconfigured settings
- **🛡️ Graceful Shutdown** — Clean session cleanup on server stop



## 🏗 ARMv7/ARM32 Compatibility

This fork is specially configured for **ARMv7 (32-bit ARM)** devices like:
- Raspberry Pi 2/3/4 (32-bit OS)
- Android phones with Termux
- ARM32 embedded Linux devices

### Key Changes from Main Branch
- **pydantic v1** (pure Python, no Rust `pydantic-core` needed)
- **FastAPI <1.0.0** (compatible with pydantic v1)
- **uvicorn without `[standard]`** (avoids `watchfiles` Rust dependency)
- **websockets v12+** (pure Python)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- An OpenCode API key
- A Composio API key

### 1. Clone & Install
```bash
git clone https://github.com/tundefund0-gif/composio-ai-agent.git
cd composio-ai-agent
pip install -r requirements-armv7.txt  # ARMv7 optimized
# OR: pip install -r requirements.txt
```

### 2. Set API Keys
```bash
# Either export them:
export OPENGATE_API_KEY="your-key"
export COMPOSIO_API_KEY="your-composio-key"

# Or copy .env.example to .env and fill in
cp .env.example .env
```

### 3. Run Web Dashboard
```bash
./start.sh
# OR: python3 -m server.main
# OR: python3 run.py web
# Open http://localhost:9090
```

### 4. Run CLI Mode
```bash
# Interactive chat
python3 -m cli.main

# One-shot question
python3 -m cli.main oneshot "What can you do?"

# Search for tools
python3 -m cli.main tools "manage github issues"

# Export conversation
python3 -m cli.main export

# View token usage
python3 -m cli.main tokens

# View config
python3 -m cli.main config
```

### 5. Docker Deployment
```bash
# ARMv7 builds are available as linux/arm/v7
# Build locally:
docker buildx build --platform linux/arm/v7 -t zen-agent-arm32 .
```

## 🖥️ CLI Commands

| Command | Description |
|---|---|
| `interactive` | Start interactive chat session |
| `oneshot "question"` | Ask a single question |
| `tools [query]` | Search or list Composio tools |
| `export` | Export conversation as Markdown |
| `tokens` | Show token usage statistics |
| `config [key]` | View configuration |
| `history` | Show recent conversation history |
| `session [info\|reset]` | Manage sessions |

### Interactive Chat Commands

| Command | Description |
|---|---|
| `/clear` | Clear conversation history |
| `/info` | Show session info and stats |
| `/save` | Save conversation to file |
| `/load` | Load most recent saved conversation |
| `/export` | Export conversation as Markdown |
| `/tokens` | Show token usage |
| `/quit` or `/exit` | Exit interactive mode |

## 🖥️ Web Dashboard

The dashboard features:
- **Dark/Light theme** — Toggle with the 🌓 button
- **Chat history** — Stored in browser (localStorage)
- **Streaming responses** — Real-time token streaming via WebSocket
- **Tool call cards** — Collapsible tool execution details
- **Thinking indicator** — See the AI's reasoning process
- **Mobile responsive** — Works on phones and tablets

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check with uptime & rate limit info |
| `/api/chat` | POST | Send a message |
| `/api/config` | GET | Get server configuration |
| `/api/session/{user_id}` | GET | Get session info |
| `/api/session/{user_id}/reset` | POST | Reset conversation |
| `/api/session/{user_id}` | DELETE | Delete session |
| `/api/tools/list` | GET | List Composio tools |
| `/api/tools/search` | GET | Search tools |
| `/ws/chat/{user_id}` | WS | Streaming chat |
| `/` | GET | Dashboard UI |

All API responses include `X-Request-ID` header for tracing.

## 🧪 Running Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_agent.py -v

# Stress test (41 tests)
python3 tests/stress_test.py
```

## 📁 Project Structure

```
zen-agent/
├── core/                    # Core engine
│   ├── agent.py             # AI agent orchestration + persistence
│   ├── llm_client.py        # LLM API client with fallback support
│   └── composio_client.py   # Composio REST API wrapper
├── cli/                     # CLI interface
│   └── main.py              # Typer-based CLI (interactive + commands)
├── server/                  # Web server
│   ├── main.py              # FastAPI app (REST + WebSocket + rate limit)
│   └── static/index.html    # Dashboard (single-file SPA)
├── tests/                   # Test suite
│   ├── test_agent.py
│   ├── test_llm_client.py
│   ├── test_composio_client.py
│   ├── test_server.py
│   └── stress_test.py
├── config.py                # Config with env vars & validation
├── Dockerfile               # Production Docker image
├── docker-compose.yml       # Docker Compose setup
├── zen-agent.service        # Systemd service file
├── .github/workflows/ci.yml # GitHub Actions CI
├── start.sh                 # One-command launcher
└── run.py                   # Unified launcher
```

## 🔌 Composio Integration

The agent uses the Composio v3/v3.1 REST API directly (no SDK required):

- **Sessions** — Create, retrieve, reuse per user
- **Meta tools** — SEARCH, EXECUTE, MANAGE_CONNECTIONS, WORKBENCH, BASH
- **Tool execution** — Execute any of 23,790+ tools by slug
- **Sandbox** — Run Python code in a remote sandbox
- **Multi-execute** — Run several tools in parallel
- **Proxy** — Make HTTP requests through connected accounts

## 🔧 Configuration

All configuration is via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OPENGATE_API_KEY` | — | LLM API key (required) |
| `OPENGATE_BASE_URL` | `https://opencode.ai/zen/v1` | LLM API base URL |
| `OPENGATE_MODEL` | `deepseek-v4-flash-free` | Model name |
| `OPENGATE_MAX_TOKENS` | `131072` | Max tokens per request |
| `OPENGATE_FALLBACK_MODEL` | — | Fallback model on primary failure |
| `OPENGATE_FALLBACK_BASE_URL` | — | Fallback API base URL |
| `OPENGATE_FALLBACK_API_KEY` | — | Fallback API key |
| `COMPOSIO_API_KEY` | — | Composio API key (required) |
| `COMPOSIO_BASE_URL` | `https://backend.composio.dev` | Composio API base URL |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `9090` | Server port |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Max API requests per IP per minute |
| `DATA_DIR` | `./data` | Data directory for persistence |
| `MAX_HISTORY_MESSAGES` | `60` | Max messages in context window |
| `MAX_TOOL_RESULTS_LENGTH` | `15000` | Max chars per tool result |
| `MAX_MESSAGE_LENGTH` | `100000` | Max chars per message |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `default` | Log format (`default` or `json`) |

## 🚢 Production Deployment

### Docker
```bash
# Build and run
# ARMv7 builds are available as linux/arm/v7
# Build locally:
docker buildx build --platform linux/arm/v7 -t zen-agent-arm32 .

# Check logs
docker compose logs -f
```

### Systemd (Linux)
```bash
# Install service
sudo cp zen-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable zen-agent
sudo systemctl start zen-agent

# Check status
sudo systemctl status zen-agent
```

## 🤝 Need Help?

- Open an issue on GitHub
- Check the Composio docs for tool-specific questions
- The dashboard health page shows system status
