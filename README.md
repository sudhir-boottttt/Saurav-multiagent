# Multi-Agent Platform

A hierarchical multi-agent AI platform built on Microsoft AutoGen that manages the full software development lifecycle. Agents are organized in a three-tier hierarchy — PM, Tech Leads, and Developers — communicating through an event-driven message bus with semantic knowledge retrieval.

## Architecture

```
PM Agent (cross-domain coordination)
├── Tech Lead - Backend (SocietyOfMindAgent)
│   ├── Dev Agent: user-service (backend)
│   ├── Dev Agent: order-service (backend)
│   ├── Dev Agent: payment-service (backend)
│   └── DB Admin Agent
├── Tech Lead - Frontend (SocietyOfMindAgent)
│   ├── Dev Agent: web-app (frontend)
│   └── Dev Agent: mobile-app (frontend)
├── Staff Engineer (cross-domain)
└── ...additional agents as configured
```

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (for message bus)
- Qdrant (for vector search)

### Installation

```bash
# Clone and install
pip install -e ".[dev]"

# Or install dependencies directly
pip install "autogen-agentchat>=0.4.0" "autogen-ext[openai]>=0.4.0" qdrant-client redis fastembed openai pyyaml pydantic
pip install hypothesis pytest pytest-asyncio  # dev dependencies
```

### Local Development Setup

For a smoother local run, create a virtual environment, install the project, and start the supporting services:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Redis and Qdrant are required by the platform architecture. If a `docker-compose.yml` file is added, these services can be started with:

```bash
docker compose up -d redis qdrant
```

Create a local environment file for API keys and service URLs:

```bash
cp .env.example .env
```

Expected environment values:

| Variable | Purpose |
|----------|---------|
| `AGENT_LLM_API_KEY` | API key used by the configured LLM provider |
| `OPENAI_API_KEY` | OpenAI-compatible model and embedding access, when used |
| `GEMINI_API_KEY` | Gemini embedding access, when used |
| `QDRANT_URL` | Qdrant URL, usually `http://localhost:6333` |
| `REDIS_URL` | Redis URL, usually `redis://localhost:6379` |

### Run the Platform

```bash
# Start with default manifest
python -m src.main --manifest agent-manifest.yaml

# With debug logging
python -m src.main --manifest agent-manifest.yaml --log-level DEBUG

# Interactive CLI mode
python -m src.cli --manifest agent-manifest.yaml
```

The platform will:
1. Load role definitions and specializations from the manifest
2. Provision all agents (PM, Tech Leads, Developers, custom roles)
3. Register agents, subscribe to message bus topics
4. Start health monitoring (15s heartbeat interval)
5. Run a main loop with hot-reload and approval timeout checks

### Suggested Upgrade Notes

These are small, practical upgrades that make the project easier to install, run, and contribute to:

- Add `autogen-ext[openai]` and `openai` to project dependencies so AutoGen's OpenAI-compatible model client can be created without manual package discovery.
- Add optional provider dependencies such as `google-generativeai` for Gemini embedding support.
- Add a deterministic offline keyword embedding fallback so local development can continue even when external embedding providers are unavailable.
- Wire the `embedding` section in `agent-manifest.yaml` into platform startup, instead of always relying on a hard-coded local fallback.
- Add `.env.example` with the expected LLM, Redis, and Qdrant environment variables.
- Add `docker-compose.yml` for Redis and Qdrant so contributors can start infrastructure consistently.
- Add a `Makefile` with common commands such as `make install`, `make test`, `make run`, and `make infra-up`.
- Replace placeholder repositories in `agent-manifest.yaml`, such as `git@github.com:your-org/...`, before using the platform against real projects.

### Run Tests

```bash
# Full test suite
python3 -m pytest tests/ -v

# Specific test file
python3 -m pytest tests/test_models.py -v

# Property-based tests only
python3 -m pytest tests/ -v -k "property"
```

## Project Structure

```
src/
├── main.py                          # Entry point — Platform bootstrap
├── cli.py                           # Interactive REPL CLI
├── core/
│   ├── enums.py                     # AgentStatus, BugSeverity, ApprovalDecision
│   └── models.py                    # Pydantic data models (RoleDefinition, AgentRecord, etc.)
├── agents/
│   ├── factory.py                   # AgentFactory, AgentWrapper, TeamWrapper
│   ├── orchestrator.py              # PM → TL → Dev conversation orchestration
│   ├── context_assembler.py         # RAG context injection for agents
│   ├── output_validator.py          # Validates agent responses against workspace
│   ├── scoping_gate.py              # Human-in-the-loop scoping confirmation
│   ├── bug_scanner.py               # Static analysis bug detection
│   ├── feature_scoper.py            # Feature scope estimation
│   ├── impact_analyzer.py           # Cross-domain impact analysis
│   ├── context_manager.py           # Conversation context management
│   └── tool_registry.py             # Tool registration and schema generation
├── infrastructure/
│   ├── agent_registry.py            # Agent lifecycle management
│   ├── role_registry.py             # Role definitions + specializations
│   ├── provisioner.py               # Dynamic agent provisioning from YAML
│   ├── message_bus.py               # Pub/sub event-driven messaging
│   ├── approval_gate.py             # Human approval workflows
│   ├── discussion_thread.py         # Multi-agent threaded discussions
│   ├── health_monitor.py            # Agent health + heartbeat tracking
│   ├── task_queue.py                # Priority task queue with state machine
│   ├── channels.py                  # Communication channel protocols
│   ├── human_interaction.py         # Human-in-the-loop interaction points
│   ├── notifications.py             # Notification provider protocols
│   ├── retry_policy.py              # Exponential backoff retry logic
│   ├── storage.py                   # SQLite storage backend
│   └── dry_run.py                   # Dry-run mode for testing
├── integrations/
│   ├── github_connector.py          # GitHub API integration
│   ├── devops_connector.py          # Work item tracking integration
│   ├── pr_review.py                 # Automated PR review
│   └── work_items.py                # Work item management
├── knowledge/
│   ├── code_indexing.py             # AST-based code chunking + indexing
│   ├── embedding.py                 # Embedding provider abstraction
│   ├── embedding_cache.py           # LRU cache for embeddings
│   ├── knowledge_ingestion.py       # Document ingestion pipeline
│   └── qdrant_service.py            # Vector search service
├── code_ops/
│   ├── code_analyzer.py             # File reading, grep, AST analysis
│   ├── local_file_manager.py        # Workspace file management with access control
│   ├── sandbox.py                   # Sandboxed command execution
│   └── task_rollback.py             # Git-based task rollback
└── observability/
    ├── audit.py                     # Audit trail + metrics
    ├── cost_tracker.py              # Token usage tracking + limits
    ├── decision_log.py              # Agent decision logging
    └── performance_metrics.py       # Agent performance tracking
```

## Key Features

- **Hierarchical agent teams** — PM → Tech Lead → Developer with clear responsibility boundaries
- **YAML-driven configuration** — Add agents, roles, and repos without code changes
- **RAG-powered context** — Agents ground their responses in actual code via vector search
- **Tool-enabled agents** — File reading, code search, command execution, PR creation
- **Human-in-the-loop** — Approval gates and scoping confirmation for critical decisions
- **Observability** — Cost tracking, decision logs, performance metrics, health monitoring
- **Hot-reload** — Add agents at runtime by editing the manifest
- **Pluggable integrations** — Protocol-based interfaces for GitHub, work items, notifications

## Configuration

Edit `agent-manifest.yaml` to configure:
- LLM provider and model (any OpenAI-compatible API)
- Agent instances, roles, and team structure
- Repository assignments and domain groupings
- Specializations (backend, frontend, fullstack, devops, etc.)
- Cost controls and health monitoring thresholds

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AGENT_LLM_API_KEY` | API key for the LLM provider |
| `QDRANT_URL` | Qdrant vector DB URL (default: localhost:6333) |
| `REDIS_URL` | Redis URL for message bus (default: localhost:6379) |
