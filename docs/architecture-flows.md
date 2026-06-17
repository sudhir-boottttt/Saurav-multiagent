# Multi-Agent Platform — Architecture Flows

## 1. Boot Sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI as src/cli.py
    participant Platform as src/main.py (Platform)
    participant Provisioner as provisioner.py
    participant RoleReg as RoleRegistry
    participant Factory as AgentFactory
    participant AgentReg as AgentRegistry
    participant Qdrant as QdrantService
    participant CodeIdx as CodeIndexingPipeline
    participant Embed as KeywordEmbedding

    User->>CLI: python3 -m src.cli
    CLI->>Platform: Platform("agent-manifest.yaml")
    
    Note over Platform: __init__ creates all components
    Platform->>Platform: _load_llm_config() from YAML
    Platform->>Platform: _init_new_components()
    Platform-->>CLI: Platform ready (components initialized)

    Note over Platform: Components created in __init__:
    Note over Platform: StorageBackend (SQLite)
    Note over Platform: ToolRegistry (8 tools registered)
    Note over Platform: EmbeddingCache
    Note over Platform: CostTracker
    Note over Platform: DecisionLog
    Note over Platform: HealthMonitor
    Note over Platform: TaskQueue
    Note over Platform: CLIChannel + HIP
    Note over Platform: ScopingGate
    Note over Platform: RetryPolicy
    Note over Platform: ContextAssembler
    Note over Platform: OutputValidator

    CLI->>CLI: boot()
    CLI->>Provisioner: provision_from_manifest("agent-manifest.yaml")
    Provisioner->>RoleReg: load_role_definitions(manifest)
    RoleReg-->>Provisioner: 5 roles loaded

    loop For each agent in manifest
        Provisioner->>Factory: create_agent(config)
        Factory->>RoleReg: get_role(), get_tools(), get_system_prompt()
        Factory-->>Provisioner: AgentWrapper
        Provisioner->>AgentReg: register_agent(record)
    end
    Provisioner-->>CLI: agents provisioned

    CLI->>CLI: _index_code_sync()
    CLI->>CodeIdx: index_repository("src/", "platform-source")
    
    loop For each .py file in src/
        CodeIdx->>CodeIdx: parse_file() via Python AST
        CodeIdx->>Embed: embed_batch(chunk_texts)
        Embed-->>CodeIdx: [float] vectors
        CodeIdx->>Qdrant: index_code("platform-source", chunks)
    end
    CodeIdx-->>CLI: chunks indexed

    CLI->>User: "Provisioned N agents" + "Indexed N chunks"
    CLI->>User: REPL prompt (agent>)
```

## 2. Task Execution Flow (/task command)

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI REPL
    participant Orch as AgentOrchestrator
    participant CtxAsm as ContextAssembler
    participant Cache as EmbeddingCache
    participant Embed as Embedding
    participant Qdrant as QdrantService
    participant LLM as LLM API
    participant CostTrk as CostTracker

    User->>CLI: /task implement user authentication
    CLI->>Orch: run_task("implement user authentication")

    Note over Orch: Step 1: PM Agent analyzes

    Orch->>Orch: _smart_call(pm_agent, prompt)
    Orch->>Orch: _build_system_prompt(pm_agent)
    Orch->>CtxAsm: assemble("implement user authentication", pm_agent)
    
    CtxAsm->>Cache: get_embedding("implement user authentication")
    Cache-->>CtxAsm: miss
    CtxAsm->>Embed: embed_text("implement user authentication")
    Embed-->>CtxAsm: [0.12, -0.34, ...]
    CtxAsm->>Cache: put_embedding(text, vector)
    
    CtxAsm->>Qdrant: search_code(vector, top_k=5)
    Qdrant-->>CtxAsm: [relevant code chunks...]
    CtxAsm->>Qdrant: search_knowledge(vector, top_k=5)
    Qdrant-->>CtxAsm: [relevant knowledge...]
    
    Note over CtxAsm: Builds enriched prompt:
    Note over CtxAsm: 1. Grounding instruction
    Note over CtxAsm: 2. Original system prompt
    Note over CtxAsm: 3. ## Relevant Code (real snippets)
    Note over CtxAsm: 4. ## Current Task
    
    CtxAsm-->>Orch: enriched_system_prompt

    Orch->>LLM: chat.completions.create(system=enriched_prompt, user=task_prompt)
    LLM-->>Orch: PM analysis with DOMAINS: backend, frontend
    Orch->>CostTrk: log_usage(pm, task, input_tokens, output_tokens)
    Orch->>User: PM Agent response (grounded in real code)

    Note over Orch: Step 2: Tech Leads break down (per domain)

    loop For each affected domain
        Orch->>Orch: _smart_call(tech_lead, breakdown_prompt)
        Orch->>CtxAsm: assemble(task, tech_lead)
        CtxAsm->>Qdrant: search_code(vector, repo_filter=domain_repos)
        Orch->>LLM: TL call with code context
        LLM-->>Orch: sub-task breakdown
    end

    Note over Orch: Step 3: Developers analyze (per sub-task)

    loop For each developer agent
        Orch->>Orch: _smart_call(dev_agent, analysis_prompt)
        Orch->>CtxAsm: assemble(subtask, dev_agent, assigned_repos=[...])
        CtxAsm->>Qdrant: search_code(vector, repo_filter=assigned_repos)
        Note over CtxAsm: Dev agents only see code from their repos
        Orch->>LLM: Dev call with repo-specific code context
        LLM-->>Orch: technical analysis referencing real files
    end

    Note over Orch: Step 4: PM consolidates

    Orch->>LLM: PM consolidation with all responses
    LLM-->>Orch: final summary
    Orch->>User: "Task complete. N agent messages exchanged."
```

## 3. Context Assembly Detail

```mermaid
flowchart TD
    A[Agent needs to respond] --> B[ContextAssembler.assemble]
    
    B --> C{EmbeddingCache hit?}
    C -->|yes| D[Use cached embedding]
    C -->|no| E[Embed query text]
    E --> F[Store in cache with TTL]
    
    D --> G[Search Qdrant Code_Index]
    F --> G
    
    G --> H{Agent has assigned_repos?}
    H -->|yes, Dev agent| I[Filter results by assigned_repos]
    H -->|no, PM/TL| J[Return all code results]
    
    I --> K[Search Qdrant Knowledge_Base]
    J --> K
    
    K --> L[Get conversation memory]
    
    L --> M[Build enriched prompt]
    
    M --> N["1. GROUNDING INSTRUCTION<br/>'Base your response ONLY on...'"]
    N --> O["2. Original system prompt<br/>(from manifest role config)"]
    O --> P["3. ## Relevant Code<br/>(real snippets from Qdrant)"]
    P --> Q["4. ## Relevant Knowledge<br/>(docs, decisions)"]
    Q --> R["5. ## Recent Context<br/>(conversation memory)"]
    R --> S["6. ## Current Task<br/>(user's request)"]
    
    S --> T[Return enriched prompt to orchestrator]
    
    style N fill:#f96,stroke:#333
    style P fill:#6f9,stroke:#333
```

## 4. Code Indexing Pipeline

```mermaid
flowchart TD
    A[CodeIndexingPipeline.index_repository] --> B[Find all source files in repo]
    B --> C{For each file}
    
    C --> D[parse_file via AST]
    D --> E{Node type?}
    
    E -->|FunctionDef| F[Extract function chunk]
    E -->|ClassDef| G[Extract class chunk]
    E -->|Module level| H[Extract module chunk]
    
    F --> I{Chunk > 512 tokens?}
    G --> I
    H --> I
    
    I -->|yes| J[Truncate to 2048 chars]
    I -->|no| K[Keep as-is]
    
    J --> L[EmbeddingService.embed_batch]
    K --> L
    
    L --> M[QdrantService.index_code]
    M --> N["Stored with metadata:<br/>repo, file_path, language,<br/>chunk_type, start_line, end_line,<br/>commit_hash, content"]
    
    C -->|next file| D
    C -->|done| O[Return total chunk count]
```

## 5. Component Dependency Map

```mermaid
graph TB
    subgraph Entry["Entry Points"]
        CLI["src/cli.py<br/>InteractiveCLI"]
        Main["src/main.py<br/>Platform"]
    end

    subgraph Orchestration["Orchestration"]
        Orch["orchestrator.py<br/>AgentOrchestrator"]
        Factory["factory.py<br/>AgentFactory"]
        CtxAsm["context_assembler.py<br/>ContextAssembler"]
        OutVal["output_validator.py<br/>OutputValidator"]
        ScopeGate["scoping_gate.py<br/>ScopingConfirmationGate"]
        ToolReg["tool_registry.py<br/>ToolRegistry"]
    end

    subgraph CodeOps["Code Operations"]
        LFM["local_file_manager.py<br/>LocalFileManager"]
        Sandbox["sandbox.py<br/>SandboxExecutor"]
        Analyzer["code_analyzer.py<br/>CodeAnalyzer"]
        Rollback["task_rollback.py<br/>TaskRollback"]
    end

    subgraph Knowledge["Knowledge Layer"]
        CodeIdx["code_indexing.py<br/>CodeIndexingPipeline"]
        KnowIdx["knowledge_ingestion.py<br/>KnowledgeIngestionPipeline"]
        EmbCache["embedding_cache.py<br/>EmbeddingCache"]
        Embed["embedding.py<br/>EmbeddingProvider"]
        Qdrant["qdrant_service.py<br/>QdrantService"]
    end

    subgraph Infra["Infrastructure"]
        Storage["storage.py<br/>StorageBackend"]
        AgentReg["agent_registry.py<br/>AgentRegistry"]
        MsgBus["message_bus.py<br/>MessageBus"]
        TaskQ["task_queue.py<br/>TaskQueue"]
        HealthMon["health_monitor.py<br/>AgentHealthMonitor"]
        Channels["channels.py<br/>CLIChannel"]
        HIP["human_interaction.py<br/>HumanInteractionPoint"]
        Notif["notifications.py<br/>ConsoleLogProvider"]
        Retry["retry_policy.py<br/>RetryPolicy"]
        DryRun["dry_run.py<br/>DryRunManager"]
        Approval["approval_gate.py<br/>ApprovalGateService"]
    end

    subgraph Observability["Observability"]
        CostTrk["cost_tracker.py<br/>CostTracker"]
        DecLog["decision_log.py<br/>DecisionLog"]
        PerfMet["performance_metrics.py<br/>AgentPerformanceMetrics"]
        Audit["audit.py<br/>AuditTrail"]
    end

    subgraph Integrations["Integrations"]
        GitHub["github_connector.py<br/>GitHubConnector"]
        DevOps["devops_connector.py<br/>DevOpsConnector"]
        WorkItems["work_items.py<br/>LocalWorkItemProvider"]
        PRReview["pr_review.py<br/>PRReviewService"]
    end

    CLI --> Main
    CLI --> Orch
    Main --> Factory & AgentReg & MsgBus & Qdrant & Audit

    Orch --> CtxAsm & OutVal & CostTrk & Retry & HealthMon & ToolReg
    Orch --> TaskQ & HIP & ScopeGate
    CtxAsm --> Qdrant & Embed & EmbCache
    ToolReg --> Analyzer & LFM & Sandbox & Qdrant

    CodeIdx --> Embed & Qdrant
    KnowIdx --> Embed & Qdrant

    AgentReg & MsgBus & Approval & TaskQ --> Storage
    CostTrk & DecLog & PerfMet --> Storage
    DecLog --> Audit
    HIP --> Channels
    ScopeGate --> HIP
    PRReview --> GitHub

    style CLI fill:#4af,stroke:#333
    style Orch fill:#fa4,stroke:#333
    style CtxAsm fill:#4f4,stroke:#333
    style Qdrant fill:#f4f,stroke:#333
    style Storage fill:#ff4,stroke:#333
```

## 6. Data Flow: Where Knowledge Lives

```
┌─────────────────────────────────────────────────────────────┐
│                    QdrantService (in-memory)                 │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐ │
│  │   Code_Index     │  │  Knowledge_Base  │  │ Conv_Mem  │ │
│  │                  │  │                  │  │           │ │
│  │ Chunks from      │  │ Ingested docs    │  │ Agent     │ │
│  │ src/*.py         │  │ (.md, .txt, etc) │  │ convo     │ │
│  │                  │  │                  │  │ history   │ │
│  │ Each chunk has:  │  │ Each chunk has:  │  │           │ │
│  │ - repo name      │  │ - source_file    │  │ - agent_id│ │
│  │ - file_path      │  │ - content        │  │ - content │ │
│  │ - chunk_type     │  │ - format         │  │ - tags    │ │
│  │ - content        │  │                  │  │           │ │
│  │ - embedding vec  │  │                  │  │           │ │
│  │ - start/end line │  │                  │  │           │ │
│  └──────────────────┘  └──────────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              StorageBackend (SQLite / Filesystem)            │
│                                                              │
│  Collections:                                                │
│  - agents/        → AgentRegistry persistence                │
│  - messages/      → MessageBus durable history               │
│  - approvals/     → ApprovalGate requests                    │
│  - threads/       → DiscussionThread persistence             │
│  - tasks/         → TaskQueue state                          │
│  - cost_records/  → CostTracker token usage                  │
│  - decisions/     → DecisionLog entries                      │
│  - heartbeats/    → HealthMonitor state                      │
│  - hip_records/   → HumanInteractionPoint records            │
│  - task_outcomes/ → PerformanceMetrics                       │
│  - task_conversations/ → Orchestrator conversation history   │
└─────────────────────────────────────────────────────────────┘
```

## 7. Agent Hierarchy (Example)

```
PM Agent (pm-001)
├── Coordinates all domains
├── Analyzes tasks, routes to Tech Leads
├── Consolidates final summaries
│
├── Tech Lead Backend (tl-backend-001)
│   ├── Breaks down backend domain tasks
│   ├── Dev: user-service (dev-user-001)
│   ├── Dev: order-service (dev-order-001)
│   └── Dev: payment-service (dev-payment-001)
│
├── Tech Lead Frontend (tl-frontend-001)
│   ├── Breaks down frontend domain tasks
│   ├── Dev: web-app (dev-webapp-001)
│   └── Dev: mobile-app (dev-mobile-001)
│
├── DB Admin (dba-001)
│   └── Schema analysis, migrations, query optimization
│
└── Staff Engineer (staff-eng-001)
    └── Cross-domain architecture decisions
```
