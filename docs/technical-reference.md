# Multi-Agent Platform — Technical Reference Guide

## Techniques, Patterns, and Architecture Decisions

This document covers the key techniques used in the platform, why they matter, and how they improve the system. Use it as a reference when evaluating trade-offs or extending the architecture.

---

## 1. RAG (Retrieval-Augmented Generation) with Dual Retrieval

**What it is:** Agents don't rely on the LLM's training data alone. Before every response, the ContextAssembler retrieves relevant code snippets and knowledge from Qdrant and injects them into the prompt.

**Why it matters:**
- Eliminates hallucinated code references — agents see real code, not imagined code
- Keeps agents current — when code changes, the index updates, and agents see the latest version
- Reduces token waste — only the top-5 relevant chunks are injected

**The dual approach:**
- Vector search (Qdrant) for discovery: "find me files related to authentication"
- Direct file access (CodeAnalyzer) for precision: "read the exact contents of `src/auth/handler.py` line 45-80"

---

## 2. AST-Based Code Chunking

**What it is:** Instead of splitting code into fixed-size text blocks, the CodeIndexingPipeline parses code into its Abstract Syntax Tree and chunks by logical units — functions, classes, module-level blocks.

**Why it matters:**
- A function stays together as one chunk, not split across two
- Search results return complete, meaningful code units
- Metadata includes function name, class name, start/end lines

**Supported parsers:**
- Python: built-in `ast` module
- Java, JavaScript, TypeScript, Go: `tree-sitter` with language-specific grammars

---

## 3. Pluggable Protocol Interfaces (Structural Subtyping)

**What it is:** Key extension points use Python `Protocol` classes instead of abstract base classes. Any class with the right methods is automatically compatible — no inheritance required.

**Protocols in the platform:**
- `StorageBackend` — swap SQLite for PostgreSQL without touching any component code
- `WorkItemProvider` — swap one tracker for another by implementing 7 methods
- `CommunicationChannel` — add Slack/Teams by implementing `send_question` and `poll_response`
- `NotificationProvider` — add PagerDuty by implementing `notify`
- `EmbeddingProvider` — add any embedding service by implementing 3 methods

---

## 4. Circuit Breaker Pattern

**What it is:** When an external service (LLM API, GitHub, Qdrant) fails repeatedly, the circuit breaker "opens" and stops sending requests for a cooldown period.

**States:**
```
CLOSED (normal) → 5 consecutive failures → OPEN (reject all)
OPEN → 300s cooldown → HALF-OPEN (allow 1 probe)
HALF-OPEN → probe succeeds → CLOSED
HALF-OPEN → probe fails → OPEN (restart cooldown)
```

---

## 5. Exponential Backoff with Jitter

**What it is:** When retrying failed requests, each retry waits longer than the last (1s → 2s → 4s), with random jitter (0-25%) to prevent thundering herd.

**Formula:** `delay = min(base * multiplier^attempt + random(0, 0.25 * delay), max_delay)`

---

## 6. Workspace Isolation with Path Traversal Protection

**What it is:** Each Developer Agent operates in a sandboxed directory (`/tmp/agent-repos/{repo_name}/`). All file operations resolve paths relative to this directory and reject any attempt to escape it.

**Protection layers:**
1. Reject `../` sequences in paths
2. Resolve symlinks and verify the real path is within the workspace
3. Reject absolute paths that don't start with the workspace prefix
4. File-level locking prevents concurrent writes to the same file

---

## 7. Output Validation (Anti-Hallucination)

**What it is:** After every agent response that references code, the OutputValidator re-reads the referenced files from disk and confirms the references are real.

**What it catches:**
- References to files that don't exist
- Incorrect function names or line numbers
- Imports from nonexistent modules

---

## 8. Git-Based Task Rollback

**What it is:** Before a multi-file task begins, the platform creates a checkpoint branch (`checkpoint/{task_id}`). If the task fails, all changes are rolled back atomically by resetting to the checkpoint.

---

## 9. TTL-Based LRU Caching for Embeddings

**What it is:** The EmbeddingCache stores embedding vectors and search results with time-to-live (TTL) expiration and least-recently-used (LRU) eviction.

**Cache layers:**
- Embedding vectors: keyed by text hash, TTL 1 hour
- Search results: keyed by (query, collection, top_k, filters), TTL 5 minutes

---

## 10. Priority Queue with Workspace Locking

**What it is:** Tasks are queued with priorities (high/normal/low) and processed in order. Before a task starts, the queue checks if any running task holds a write lock on the same repository.

---

## 11. Human-in-the-Loop with Multi-Channel Delivery

**What it is:** When an agent needs human input, it raises a HumanInteractionPoint (HIP) that delivers the question to all configured channels simultaneously. The first response from any channel is accepted.

---

## 12. Scoping Confirmation Gate

**What it is:** After PM and Tech Lead agents complete task breakdown, the platform pauses and presents the full scope to a human before Developer Agents start working.

---

## 13. Structured Decision Logging

**What it is:** Every significant agent decision is logged with the context that was available at decision time, alternatives considered, and the chosen action.

---

## 14. Cost Controls with Budget Enforcement

**What it is:** Every LLM call's token usage is tracked per agent, per task, and per day. Configurable limits pause tasks or throttle agents when budgets are exceeded.

**Limits (configurable via manifest):**
- Per task: 500K tokens
- Per agent per hour: 200K tokens
- Per day: 5M tokens

---

## 15. Pluggable Persistence (StorageBackend)

**What it is:** All platform state is stored through a `StorageBackend` protocol. The default uses SQLite; a filesystem (JSON) backend is available for development.

**Migration path:**
1. SQLite backend (single file, zero config) — default
2. PostgreSQL for multi-instance deployment — swap via protocol

---

## 16. Dry-Run Mode

**What it is:** A `--dry-run` flag that lets agents think and reason normally (LLM calls still happen) but intercepts all side effects — file writes, PR creation, work items, notifications — and logs them instead.

---

## Architecture Decision Summary

| Decision | Alternative Considered | Why We Chose This |
|---|---|---|
| AutoGen framework | LangGraph | AutoGen's GroupChat and SocietyOfMindAgent map directly to PM→TL→Dev hierarchy |
| Qdrant for vectors | Pinecone, ChromaDB | Local mode (no server needed), Python-native API, scales to remote deployment |
| SQLite for structured state | PostgreSQL, Redis | Zero-config, single-file, ACID-compliant. Easy upgrade path |
| Protocol classes | Abstract base classes | Structural subtyping — no inheritance required |
| AST chunking | Fixed-size text splitting | Preserves logical code units |
| Git branches for rollback | File-level backup copies | Atomic multi-file rollback leveraging git |
| Embedding cache with TTL | No caching | 60-70% fewer API calls per task flow |
| Circuit breaker | Simple retry only | Prevents cascading failures during outages |
| Per-task cost limits | Global limit only | Isolates runaway tasks |
