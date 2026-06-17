"""Multi-Agent Platform — Main Entry Point.

Bootstraps the entire platform from a YAML manifest:
1. Initializes infrastructure (Role Registry, Agent Registry, Message Bus)
2. Initializes knowledge layer (Embedding Service, Qdrant Service)
3. Initializes observability (Audit Trail, Metrics Collector)
4. Provisions all agents from the manifest via Dynamic Provisioner
5. Starts the health monitoring loop
6. Runs until interrupted (Ctrl+C)

Usage:
    python -m src.main --manifest agent-manifest.yaml
    python -m src.main --manifest agent-manifest.yaml --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from src.agents.factory import AgentFactory, LLMConfig
from src.infrastructure.agent_registry import AgentRegistry
from src.infrastructure.approval_gate import ApprovalGateService
from src.infrastructure.discussion_thread import DiscussionThreadManager
from src.infrastructure.message_bus import MessageBus
from src.infrastructure.provisioner import DynamicProvisioner
from src.infrastructure.role_registry import RoleRegistry
from src.knowledge.qdrant_service import QdrantService
from src.observability.audit import AuditTrail, MetricsCollector

logger = logging.getLogger("agent_platform")


class Platform:
    """Top-level orchestrator that wires all components together."""

    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = manifest_path
        self._shutdown_event = asyncio.Event()

        # Infrastructure
        self.role_registry = RoleRegistry()
        self.agent_registry = AgentRegistry()
        self.message_bus = MessageBus()

        # Knowledge layer — vector size matches embedding provider
        self.qdrant_service = QdrantService(vector_size=8)

        # Observability
        self.audit_trail = AuditTrail()
        self.metrics = MetricsCollector()

        # Services
        self.approval_gate = ApprovalGateService()
        self.discussion_manager = DiscussionThreadManager(
            qdrant_service=self.qdrant_service,
        )

        # Agent factory + provisioner (LLM config loaded from manifest)
        self._llm_config = self._load_llm_config()
        self.agent_factory = AgentFactory(self.role_registry, self._llm_config)
        self.provisioner = DynamicProvisioner(
            role_registry=self.role_registry,
            agent_registry=self.agent_registry,
            agent_factory=self.agent_factory,
            message_bus=self.message_bus,
            qdrant_service=self.qdrant_service,
        )

        # --- New components (real-data-sources) ---
        self._init_new_components()

    def _load_llm_config(self) -> LLMConfig:
        """Read llm section from manifest YAML."""
        import os
        import yaml

        try:
            with open(self.manifest_path) as f:
                manifest = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return LLMConfig()

        llm = manifest.get("llm", {})
        return LLMConfig(
            model=llm.get("model", "gpt-4o-mini"),
            base_url=llm.get("base_url"),
            api_key=llm.get("api_key") or os.environ.get("AGENT_LLM_API_KEY"),
            provider=llm.get("provider", "openai"),
        )

    def _init_new_components(self) -> None:
        """Initialize all new real-data-sources components."""
        import yaml

        try:
            with open(self.manifest_path) as f:
                manifest = yaml.safe_load(f) or {}
        except FileNotFoundError:
            manifest = {}

        # Storage
        from src.infrastructure.storage import SQLiteBackend
        workspace_cfg = manifest.get("workspace", {})
        base_path = workspace_cfg.get("base_path", "/tmp/agent-repos/")
        self.storage = SQLiteBackend(base_path=base_path)

        # Tool Registry
        from src.agents.tool_registry import ToolRegistry
        self.tool_registry = ToolRegistry()

        # Embedding Cache
        from src.knowledge.embedding_cache import EmbeddingCache
        cache_cfg = manifest.get("cache", {})
        self.embedding_cache = EmbeddingCache(
            embedding_ttl=cache_cfg.get("embedding_ttl_seconds", 3600),
            search_ttl=cache_cfg.get("search_ttl_seconds", 300),
            max_size_mb=cache_cfg.get("max_cache_size_mb", 100),
        )

        # Cost Tracker
        from src.observability.cost_tracker import CostTracker
        cost_cfg = manifest.get("cost_controls", {})
        self.cost_tracker = CostTracker(
            storage=self.storage,
            max_tokens_per_task=cost_cfg.get("max_tokens_per_task", 500_000),
            max_tokens_per_agent_per_hour=cost_cfg.get("max_tokens_per_agent_per_hour", 200_000),
            max_tokens_per_day=cost_cfg.get("max_tokens_per_day", 5_000_000),
        )

        # Decision Log
        from src.observability.decision_log import DecisionLog
        self.decision_log = DecisionLog(storage=self.storage, audit_trail=self.audit_trail)

        # Health Monitor
        from src.infrastructure.health_monitor import AgentHealthMonitor
        health_cfg = manifest.get("health", {})
        self.health_monitor = AgentHealthMonitor(
            agent_registry=self.agent_registry,
            storage=self.storage,
            heartbeat_timeout=health_cfg.get("heartbeat_timeout_seconds", 120),
            max_tool_calls_per_turn=health_cfg.get("max_tool_calls_per_turn", 10),
        )

        # Task Queue
        from src.infrastructure.task_queue import TaskQueue
        self.task_queue = TaskQueue(storage=self.storage)

        # Communication Channels
        from src.infrastructure.channels import CLIChannel
        self.cli_channel = CLIChannel()

        # Notifications
        from src.infrastructure.notifications import ConsoleLogProvider
        self.notification_provider = ConsoleLogProvider()

        # HIP
        from src.infrastructure.human_interaction import HumanInteractionPoint
        self.hip = HumanInteractionPoint(
            channels={"cli": self.cli_channel},
            storage=self.storage,
        )

        # Scoping Gate
        from src.agents.scoping_gate import ScopingConfirmationGate
        self.scoping_gate = ScopingConfirmationGate(
            hip=self.hip,
            enabled=manifest.get("scoping_confirmation", True),
        )

        # Retry Policy
        from src.infrastructure.retry_policy import RetryConfig, RetryPolicy
        retry_cfg = manifest.get("retry", {})
        self.retry_policy = RetryPolicy(RetryConfig(
            max_retries=retry_cfg.get("max_retries", 3),
            base_delay_seconds=retry_cfg.get("base_delay_seconds", 1.0),
        ))

        # Performance Metrics
        from src.observability.performance_metrics import AgentPerformanceMetrics
        self.performance_metrics = AgentPerformanceMetrics(storage=self.storage)

        # Dry Run Manager
        from src.infrastructure.dry_run import DryRunManager
        self.dry_run_manager = DryRunManager(enabled=manifest.get("dry_run", False))

        # Context Assembler
        from src.agents.context_assembler import ContextAssembler

        # Create a simple hash-based embedding for local use (no API needed)
        embedding_provider = self._create_embedding_provider(manifest)
        self._context_assembler = ContextAssembler(
            qdrant=self.qdrant_service,
            embedding=embedding_provider,
            cache=self.embedding_cache,
        )

        # Output Validator
        from src.agents.output_validator import OutputValidator
        self._output_validator = OutputValidator(base_path=base_path)

        # Register built-in tools
        self._register_builtin_tools(base_path)

    def _create_embedding_provider(self, manifest: dict) -> Any:
        """Create an embedding provider. Uses keyword-overlap vectors if no API configured."""
        import hashlib
        import math
        import re

        class KeywordEmbedding:
            """Keyword-based embedding that produces meaningful similarity scores.

            Instead of random hashes, this creates vectors where similar text
            produces similar embeddings via a bag-of-words approach projected
            into a fixed-size vector using feature hashing.
            """

            VECTOR_SIZE = 8

            def _tokenize(self, text: str) -> list[str]:
                """Extract lowercase word tokens and bigrams."""
                words = re.findall(r'[a-z_][a-z0-9_]*', text.lower())
                # Include bigrams for better semantic matching
                tokens = list(words)
                for i in range(len(words) - 1):
                    tokens.append(f"{words[i]}_{words[i+1]}")
                return tokens

            def embed_text(self, text: str) -> list[float]:
                """Create a feature-hashed vector from text tokens."""
                vec = [0.0] * self.VECTOR_SIZE
                tokens = self._tokenize(text)
                if not tokens:
                    return vec

                for token in tokens:
                    h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                    bucket = h % self.VECTOR_SIZE
                    # Use sign hashing to reduce collisions
                    sign = 1.0 if (h // self.VECTOR_SIZE) % 2 == 0 else -1.0
                    vec[bucket] += sign

                # L2 normalize
                norm = math.sqrt(sum(v * v for v in vec))
                if norm > 0:
                    vec = [v / norm for v in vec]
                return vec

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_text(t) for t in texts]

            def get_dimensions(self) -> int:
                return self.VECTOR_SIZE

        return KeywordEmbedding()

    def _register_builtin_tools(self, base_path: str) -> None:
        """Register built-in tools for agents to use."""
        from src.code_ops.code_analyzer import CodeAnalyzer
        from src.code_ops.local_file_manager import LocalFileManager
        from src.code_ops.sandbox import SandboxExecutor

        analyzer = CodeAnalyzer(base_path=base_path)
        lfm = LocalFileManager(base_path=base_path)
        sandbox = SandboxExecutor(base_path=base_path)

        # Register individual tools with clear names
        self.tool_registry.register("read_file", analyzer.read_file)
        self.tool_registry.register("grep", analyzer.grep)
        self.tool_registry.register("find_references", analyzer.find_references)
        self.tool_registry.register("get_ast", analyzer.get_ast)
        self.tool_registry.register("write_file", lfm.write_file)
        self.tool_registry.register("execute_command", sandbox.execute)
        self.tool_registry.register("search_code", self.qdrant_service.search_code)
        self.tool_registry.register("search_knowledge", self.qdrant_service.search_knowledge)

    def _resolve_agent_tools(self) -> None:
        """Replace manifest tool names with actual registered tool names on all agents.

        The manifest uses names like 'GitHubConnector.clone_repo' and 'CodeIndex.search',
        but the ToolRegistry registers them as 'read_file', 'grep', 'search_code', etc.
        This method gives each agent the correct set of real tool names based on their role.
        """
        registered = set(self.tool_registry.list_tools())
        code_tools = ["read_file", "grep", "find_references", "get_ast", "search_code"]
        write_tools = ["write_file", "execute_command"]
        knowledge_tools = ["search_knowledge"]

        for agent in self.provisioner.provisioned_agents.values():
            if agent.role == "developer_agent":
                agent.tools = code_tools + write_tools + knowledge_tools
            elif agent.role == "tech_lead_agent":
                agent.tools = code_tools + knowledge_tools
            elif agent.role == "pm_agent":
                agent.tools = ["search_code", "search_knowledge"]
            elif agent.role in ("staff_engineer", "db_admin"):
                agent.tools = code_tools + knowledge_tools
            else:
                # Give at least search tools to any other role
                agent.tools = ["search_code", "search_knowledge"]

    async def start(self) -> None:
        """Bootstrap and run the platform."""
        manifest = Path(self.manifest_path)
        if not manifest.exists():
            logger.error("Manifest not found: %s", self.manifest_path)
            sys.exit(1)

        logger.info("Starting Multi-Agent Platform...")
        logger.info("Manifest: %s", self.manifest_path)
        logger.info("LLM provider: %s (model=%s)", self._llm_config.provider, self._llm_config.model)

        # 1. Provision agents from manifest
        agents = self.provisioner.provision_from_manifest(self.manifest_path)
        logger.info("Provisioned %d agents", len(agents))
        for agent in agents:
            logger.info("  → %s (role=%s, spec=%s)", agent.agent_id, agent.role, agent.specialization)

        # 2. Mark all agents healthy
        from src.core.enums import AgentStatus
        for agent in agents:
            self.agent_registry.update_status(agent.agent_id, AgentStatus.HEALTHY)

        # 2.5. Index local source code for grounded context
        await self._index_local_code()

        # 2.6. Resolve manifest tool names to actual registered tool names
        self._resolve_agent_tools()

        # 3. Start health monitoring
        await self.agent_registry.start_health_monitor(interval_seconds=15.0)
        logger.info("Health monitor started (15s interval)")

        # 4. Log platform summary
        roles = self.role_registry.list_roles()
        logger.info("Loaded %d roles: %s", len(roles), [r.role_name for r in roles])
        self.metrics.set_gauge("agents_total", len(agents))
        self.metrics.set_gauge("roles_total", len(roles))

        logger.info("Platform ready. Press Ctrl+C to shut down.")

        # 5. Main loop — hot-reload check + approval timeout check
        await self._run_loop()

    async def _index_local_code(self) -> None:
        """Index the project's own source code into Qdrant for grounded context."""
        import os
        try:
            from src.knowledge.code_indexing import CodeIndexingPipeline

            # Use the same embedding provider as context assembler
            embedding = self._context_assembler._embedding
            if not embedding:
                logger.warning("No embedding provider, skipping code indexing")
                return

            pipeline = CodeIndexingPipeline(
                embedding_service=embedding,
                qdrant_service=self.qdrant_service,
            )

            # Index the project's own src/ directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            src_dir = os.path.join(project_root, "src")
            if os.path.isdir(src_dir):
                count = pipeline.index_repository(src_dir, "platform-source", "python", "local")
                logger.info("Indexed %d code chunks from src/", count)
            else:
                logger.info("No src/ directory found for indexing")

        except Exception as e:
            logger.warning("Code indexing failed (non-fatal): %s", e)

    async def _run_loop(self) -> None:
        """Main event loop: periodic hot-reload and timeout checks."""
        while not self._shutdown_event.is_set():
            try:
                # Check for manifest changes (hot-reload)
                new_agents = self.provisioner.check_hot_reload(self.manifest_path)
                if new_agents:
                    logger.info("Hot-reload: provisioned %d new agents", len(new_agents))
                    self.metrics.set_gauge(
                        "agents_total",
                        len(self.provisioner.provisioned_agents),
                    )

                # Check for approval timeouts
                escalated = self.approval_gate.check_timeouts()
                if escalated:
                    logger.info("Escalated %d timed-out approval requests", len(escalated))

                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in main loop")
                await asyncio.sleep(5)

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._shutdown_event.set()
        await self.agent_registry.stop_health_monitor()
        logger.info("Platform stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Platform")
    parser.add_argument(
        "--manifest", "-m",
        default="agent-manifest.yaml",
        help="Path to the YAML agent manifest (default: agent-manifest.yaml)",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    platform = Platform(manifest_path=args.manifest)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_signal() -> None:
        loop.create_task(platform.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    try:
        loop.run_until_complete(platform.start())
    except KeyboardInterrupt:
        loop.run_until_complete(platform.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
