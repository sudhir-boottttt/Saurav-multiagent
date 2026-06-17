"""Interactive CLI for the Multi-Agent Platform.

Boots the platform from a YAML manifest and provides a REPL where
you can send messages to agents, list them, check status, and trigger
workflows like bug scanning or feature scoping.

Usage:
    python -m src.cli --manifest agent-manifest.yaml
    python -m src.cli  # defaults to agent-manifest.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional

from src.agents.factory import AgentWrapper, LLMConfig
from src.core.enums import AgentStatus
from src.main import Platform

logger = logging.getLogger("agent_cli")

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║          Multi-Agent Platform — Interactive CLI          ║
╠══════════════════════════════════════════════════════════╣
║  Commands:                                               ║
║    /task <desc>     Run PM→TL→Dev multi-agent workflow   ║
║    /agents          List all provisioned agents          ║
║    /status          Show agent health status             ║
║    /info <id>       Show agent details                   ║
║    /ask <id> <msg>  Send a message to a specific agent   ║
║    /tools <id>      List tools available to an agent     ║
║    /bus             Show message bus stats               ║
║    /approvals       Show pending approval requests       ║
║    /rollback <id> <path> Rollback task to checkpoint     ║
║    /ingest <path>   Ingest docs into knowledge base      ║
║    /help            Show this help                       ║
║    /quit            Exit the CLI                         ║
║                                                          ║
║  Or just type a message — it goes to the PM agent.       ║
╚══════════════════════════════════════════════════════════╝
"""


class InteractiveCLI:
    """REPL that wraps the Platform and routes user input to agents."""

    def __init__(self, manifest_path: str) -> None:
        self.platform = Platform(manifest_path)
        self._booted = False

    def boot(self) -> None:
        """Provision agents from manifest (synchronous, no event loop)."""
        manifest = Path(self.platform.manifest_path)
        if not manifest.exists():
            print(f"Error: manifest not found at {self.platform.manifest_path}")
            sys.exit(1)

        agents = self.platform.provisioner.provision_from_manifest(
            self.platform.manifest_path,
        )
        # Mark all healthy
        for agent in agents:
            self.platform.agent_registry.update_status(
                agent.agent_id, AgentStatus.HEALTHY,
            )

        # Index local source code for grounded context
        self._index_code_sync()

        # Resolve manifest tool names to actual registered tool names
        self.platform._resolve_agent_tools()

        self._booted = True
        print(f"\n  Provisioned {len(agents)} agents from {self.platform.manifest_path}\n")

    def _index_code_sync(self) -> None:
        """Index project source code into Qdrant at boot."""
        import os
        try:
            from src.knowledge.code_indexing import CodeIndexingPipeline

            embedding = self.platform._context_assembler._embedding
            if not embedding:
                return

            pipeline = CodeIndexingPipeline(
                embedding_service=embedding,
                qdrant_service=self.platform.qdrant_service,
            )

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            src_dir = os.path.join(project_root, "src")
            if os.path.isdir(src_dir):
                count = pipeline.index_repository(src_dir, "platform-source", "python", "local")
                print(f"  Indexed {count} code chunks from src/")
        except Exception as e:
            print(f"  Code indexing skipped: {e}")

    # ── Command dispatch ─────────────────────────────────────────

    def run(self) -> None:
        """Main REPL loop."""
        if not self._booted:
            self.boot()

        print(BANNER)

        while True:
            try:
                raw = input("\033[1;36magent>\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not raw:
                continue

            if raw.startswith("/"):
                self._handle_command(raw)
            else:
                # Default: send to PM agent
                self._send_to_agent("pm-001", raw)

    def _handle_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=2)
        cmd = parts[0].lower()

        dispatch = {
            "/agents": self._cmd_agents,
            "/status": self._cmd_status,
            "/info": self._cmd_info,
            "/ask": self._cmd_ask,
            "/task": self._cmd_task,
            "/tools": self._cmd_tools,
            "/bus": self._cmd_bus,
            "/approvals": self._cmd_approvals,
            "/help": self._cmd_help,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
            "/queue": self._cmd_queue,
            "/submit": self._cmd_submit,
            "/cancel": self._cmd_cancel,
            "/costs": self._cmd_costs,
            "/metrics": self._cmd_metrics,
            "/health": self._cmd_health,
            "/decisions": self._cmd_decisions,
            "/cache": self._cmd_cache,
            "/rollback": self._cmd_rollback,
            "/ingest": self._cmd_ingest,
        }

        handler = dispatch.get(cmd)
        if handler:
            handler(parts)
        else:
            print(f"  Unknown command: {cmd}. Type /help for available commands.")

    # ── Commands ─────────────────────────────────────────────────

    def _cmd_agents(self, _parts: list[str]) -> None:
        agents = self.platform.provisioner.provisioned_agents
        if not agents:
            print("  No agents provisioned.")
            return
        registered_tools = set(self.platform.tool_registry.list_tools())
        print(f"\n  {'ID':<25} {'Role':<20} {'Spec':<12} {'Tools':<8} Repos")
        print(f"  {'─'*25} {'─'*20} {'─'*12} {'─'*8} {'─'*20}")
        for aid, agent in sorted(agents.items()):
            spec = agent.specialization or "–"
            # Show how many tools actually resolve
            real_tools = [t for t in agent.tools if t in registered_tools]
            repos = ", ".join(agent.assigned_repos[:2]) if agent.assigned_repos else "–"
            if len(agent.assigned_repos) > 2:
                repos += f" +{len(agent.assigned_repos) - 2}"
            print(f"  {aid:<25} {agent.role:<20} {spec:<12} {len(real_tools):<8} {repos}")
        print()

    def _cmd_status(self, _parts: list[str]) -> None:
        agents = self.platform.provisioner.provisioned_agents
        print(f"\n  {'ID':<25} {'Status':<12} {'Role':<20}")
        print(f"  {'─'*25} {'─'*12} {'─'*20}")
        for aid, agent in sorted(agents.items()):
            record = self.platform.agent_registry.get_agent(aid)
            status = record.status.value if record else "unknown"
            print(f"  {aid:<25} {status:<12} {agent.role:<20}")
        print()

    def _cmd_info(self, parts: list[str]) -> None:
        if len(parts) < 2:
            print("  Usage: /info <agent-id>")
            return
        agent_id = parts[1]
        agent = self.platform.provisioner.provisioned_agents.get(agent_id)
        if not agent:
            print(f"  Agent '{agent_id}' not found. Use /agents to list.")
            return
        print(f"\n  Agent: {agent.agent_id}")
        print(f"  Name:  {agent.name}")
        print(f"  Role:  {agent.role}")
        print(f"  Spec:  {agent.specialization or '–'}")
        print(f"  Model: {agent.model}")
        print(f"  Domain: {', '.join(agent.domain) if agent.domain else '–'}")
        print(f"  Repos:  {', '.join(agent.assigned_repos) if agent.assigned_repos else '–'}")
        print(f"  Parent: {agent.parent_agent_id or '–'}")
        print(f"  AutoGen: {'Yes' if agent.is_autogen_backed else 'No (AgentWrapper fallback)'}")
        print(f"  Tools ({len(agent.tools)}):")
        for t in agent.tools:
            print(f"    • {t}")
        print(f"\n  System Prompt (first 300 chars):")
        print(textwrap.indent(agent.system_prompt[:300], "    "))
        if len(agent.system_prompt) > 300:
            print("    ...")
        print()

    def _cmd_ask(self, parts: list[str]) -> None:
        if len(parts) < 3:
            print("  Usage: /ask <agent-id> <message>")
            return
        agent_id = parts[1]
        message = parts[2]
        self._send_to_agent(agent_id, message)

    def _cmd_task(self, parts: list[str]) -> None:
        """Run a multi-agent task: PM → Tech Lead(s) → Developer(s) → Summary."""
        if len(parts) < 2:
            print("  Usage: /task <description>")
            print("  Example: /task add barcode scanning to the receiving service")
            return
        # Rejoin everything after /task
        task_desc = " ".join(parts[1:])
        self._run_multi_agent_task(task_desc)

    def _run_multi_agent_task(self, task_description: str) -> None:
        """Execute the full orchestration pipeline."""
        import asyncio
        import warnings
        from src.agents.orchestrator import AgentOrchestrator

        agents = self.platform.provisioner.provisioned_agents
        llm_cfg = self.platform._llm_config

        # Track step count for progress
        step_count = [0]

        def on_message(agent_id: str, role: str, content: str) -> None:
            # Skip status messages (Analyzing..., Breaking down..., Consolidating...)
            if content.endswith("...") and len(content) < 50:
                print(f"\n  \033[2m  ⏳ {role} ({agent_id}): {content}\033[0m")
                return

            step_count[0] += 1
            # Truncate very long responses for readability
            display = content
            if len(display) > 2000:
                display = display[:2000] + f"\n    ... [truncated, {len(content)} chars total]"

            print(f"\n  \033[1;36m{'─' * 60}\033[0m")
            print(f"  \033[1;33mStep {step_count[0]}\033[0m │ \033[1;32m{role}\033[0m ({agent_id})")
            print(f"  \033[1;36m{'─' * 60}\033[0m")
            for line in display.split("\n"):
                print(f"    {line}")

        orchestrator = AgentOrchestrator(
            agents=agents,
            llm_config=llm_cfg,
            on_message=on_message,
            tool_registry=getattr(self.platform, "tool_registry", None),
            context_assembler=getattr(self.platform, "_context_assembler", None),
            output_validator=getattr(self.platform, "_output_validator", None),
            cost_tracker=getattr(self.platform, "cost_tracker", None),
            health_monitor=getattr(self.platform, "health_monitor", None),
            task_queue=getattr(self.platform, "task_queue", None),
            hip=getattr(self.platform, "hip", None),
            scoping_gate=getattr(self.platform, "scoping_gate", None),
            qdrant_service=getattr(self.platform, "qdrant_service", None),
            storage=getattr(self.platform, "storage", None),
        )

        print(f"\n  \033[1;33m▶ Starting multi-agent task...\033[0m")
        print(f"  Task: {task_description}")
        domains = orchestrator._get_all_domains()
        tool_count = sum(len(a.tools) for a in agents.values())
        print(f"  Agents: {len(agents)} | Domains: {', '.join(domains)} | Tools: {tool_count}")
        print(f"  \033[2mFlow: PM analysis → TL breakdown → Dev analysis → PM summary\033[0m")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Resolved model mismatch")
                result = loop.run_until_complete(orchestrator.run_task(task_description))
            # Clean up pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            print(f"\n  Task failed: {e}")
            return
        finally:
            loop.close()

        print(f"\n  \033[1;33m{'═' * 60}\033[0m")
        print(f"  \033[1;33m▶ Task complete.\033[0m")
        print(f"    Messages: {len(result.messages)} | Steps: {step_count[0]}")
        if hasattr(self.platform, 'cost_tracker'):
            summary = self.platform.cost_tracker.get_usage_summary()
            if summary.get('total_calls', 0) > 0:
                print(f"    Tokens used: {summary['daily_total']:,}")
        print(f"  \033[1;33m{'═' * 60}\033[0m\n")

    def _cmd_tools(self, parts: list[str]) -> None:
        if len(parts) < 2:
            print("  Usage: /tools <agent-id>")
            return
        agent_id = parts[1]
        agent = self.platform.provisioner.provisioned_agents.get(agent_id)
        if not agent:
            print(f"  Agent '{agent_id}' not found.")
            return
        registered = set(self.platform.tool_registry.list_tools())
        print(f"\n  Tools for {agent_id} ({agent.role}):")
        for t in agent.tools:
            status = "\033[32m✓\033[0m" if t in registered else "\033[31m✗ unregistered\033[0m"
            print(f"    {status} {t}")
        print()

    def _cmd_bus(self, _parts: list[str]) -> None:
        bus = self.platform.message_bus
        print(f"\n  Message Bus Stats:")
        print(f"    Messages stored:  {len(bus._message_store)}")
        print(f"    Subscriptions:    {len(bus._subscriptions)}")
        print(f"    Dead-letter queue: {len(bus._dead_letter_queue)}")
        print()

    def _cmd_approvals(self, _parts: list[str]) -> None:
        pending = self.platform.approval_gate.get_pending()
        if not pending:
            print("  No pending approval requests.")
            return
        print(f"\n  Pending Approvals ({len(pending)}):")
        for req in pending:
            print(f"    • {req.request_id}: {req.action_type} by {req.requester_id}")
        print()

    def _cmd_help(self, _parts: list[str]) -> None:
        print(BANNER)

    def _cmd_quit(self, _parts: list[str]) -> None:
        print("  Bye!")
        sys.exit(0)

    # ── New commands (real-data-sources) ─────────────────────────

    def _cmd_queue(self, _parts: list[str]) -> None:
        if not hasattr(self.platform, "task_queue"):
            print("  Task queue not initialized.")
            return
        tasks = self.platform.task_queue.list_tasks()
        if not tasks:
            print("  Task queue is empty.")
            return
        print(f"\n  {'ID':<38} {'State':<15} {'Priority':<10} Description")
        print(f"  {'─'*38} {'─'*15} {'─'*10} {'─'*20}")
        for t in tasks:
            print(f"  {t.task_id:<38} {t.state.value:<15} {t.priority.value:<10} {t.description[:40]}")
        print()

    def _cmd_submit(self, parts: list[str]) -> None:
        if len(parts) < 2:
            print("  Usage: /submit <task description>")
            return
        if not hasattr(self.platform, "task_queue"):
            print("  Task queue not initialized.")
            return
        desc = " ".join(parts[1:])
        tid = self.platform.task_queue.enqueue(desc)
        print(f"  Task submitted: {tid}")

    def _cmd_cancel(self, parts: list[str]) -> None:
        if len(parts) < 2:
            print("  Usage: /cancel <task-id>")
            return
        if not hasattr(self.platform, "task_queue"):
            print("  Task queue not initialized.")
            return
        try:
            self.platform.task_queue.cancel(parts[1])
            print(f"  Task {parts[1]} cancelled.")
        except KeyError:
            print(f"  Task {parts[1]} not found.")

    def _cmd_costs(self, _parts: list[str]) -> None:
        if not hasattr(self.platform, "cost_tracker"):
            print("  Cost tracker not initialized.")
            return
        summary = self.platform.cost_tracker.get_usage_summary()
        print(f"\n  Token Usage Summary:")
        print(f"    Daily total: {summary['daily_total']:,} / {summary['daily_limit']:,}")
        print(f"    Total calls: {summary['total_calls']}")
        if summary["by_agent"]:
            print(f"    By agent:")
            for agent, tokens in summary["by_agent"].items():
                print(f"      {agent}: {tokens:,}")
        print()

    def _cmd_metrics(self, _parts: list[str]) -> None:
        if not hasattr(self.platform, "performance_metrics"):
            print("  Performance metrics not initialized.")
            return
        summary = self.platform.performance_metrics.get_platform_summary()
        print(f"\n  Platform Metrics:")
        print(f"    Agents tracked: {summary['agents_tracked']}")
        print(f"    Task outcomes: {summary['total_task_outcomes']}")
        alerts = self.platform.performance_metrics.check_alerts()
        if alerts:
            print(f"    Alerts:")
            for a in alerts:
                print(f"      ⚠ {a['agent_id']}: {a['alert']} ({a['rate']:.1%})")
        print()

    def _cmd_health(self, _parts: list[str]) -> None:
        if not hasattr(self.platform, "health_monitor"):
            print("  Health monitor not initialized.")
            return
        status = self.platform.health_monitor.get_status()
        print(f"\n  Health Status:")
        print(f"    Monitored agents: {status['monitored_agents']}")
        print(f"    Stalled agents: {status['stalled_agents'] or 'none'}")
        print()

    def _cmd_decisions(self, _parts: list[str]) -> None:
        if not hasattr(self.platform, "decision_log"):
            print("  Decision log not initialized.")
            return
        entries = self.platform.decision_log._entries[-10:]  # last 10
        if not entries:
            print("  No decisions logged yet.")
            return
        print(f"\n  Recent Decisions (last {len(entries)}):")
        for e in entries:
            print(f"    [{e.timestamp[:19]}] {e.agent_id}: {e.decision_type} → {e.chosen_action[:50]}")
        print()

    def _cmd_cache(self, _parts: list[str]) -> None:
        if not hasattr(self.platform, "embedding_cache"):
            print("  Embedding cache not initialized.")
            return
        stats = self.platform.embedding_cache.stats()
        print(f"\n  Embedding Cache:")
        print(f"    Hit rate: {stats['hit_rate']:.1%}")
        print(f"    Entries: {stats['embedding_entries']} embeddings, {stats['search_entries']} searches")
        print(f"    Size: {stats['size_bytes'] / 1024:.1f} KB / {stats['max_size_bytes'] / 1024 / 1024:.0f} MB")
        print()

    def _cmd_rollback(self, parts: list[str]) -> None:
        if len(parts) < 3:
            print("  Usage: /rollback <task-id> <repo-path>")
            return
        task_id = parts[1]
        repo_path = parts[2]
        try:
            from src.code_ops.task_rollback import TaskRollback
            rollback = getattr(self.platform, "task_rollback", None)
            if rollback is None:
                rollback = TaskRollback()
            rollback.rollback(task_id, repo_path)
            print(f"  Rolled back task {task_id} in {repo_path}.")
        except ValueError as e:
            print(f"  Rollback failed: {e}")
        except Exception as e:
            print(f"  Rollback error: {e}")

    def _cmd_ingest(self, parts: list[str]) -> None:
        if len(parts) < 2:
            print("  Usage: /ingest <file-or-directory-path>")
            return
        target = parts[1]
        try:
            from src.knowledge.knowledge_ingestion import KnowledgeIngestionPipeline
            embedding = self.platform._context_assembler._embedding
            if not embedding:
                print("  No embedding provider configured.")
                return
            pipeline = KnowledgeIngestionPipeline(
                embedding_service=embedding,
                qdrant_service=self.platform.qdrant_service,
            )
            from pathlib import Path
            p = Path(target)
            if p.is_dir():
                count = pipeline.ingest_directory(target)
            elif p.is_file():
                count = pipeline.ingest_file(target)
            else:
                print(f"  Path not found: {target}")
                return
            print(f"  Ingested {count} chunks from {target}.")
        except Exception as e:
            print(f"  Ingestion error: {e}")

    # ── Agent messaging ──────────────────────────────────────────

    def _send_to_agent(self, agent_id: str, message: str) -> None:
        agent = self.platform.provisioner.provisioned_agents.get(agent_id)
        if not agent:
            print(f"  Agent '{agent_id}' not found. Use /agents to list.")
            return

        print(f"\n  → Sending to {agent_id} ({agent.role})...")

        if agent.is_autogen_backed:
            self._send_autogen(agent, message)
        else:
            # Fallback: direct LLM call if API key available but AutoGen agent wasn't created
            llm_cfg = self.platform._llm_config
            if llm_cfg.api_key or __import__("os").environ.get("AGENT_LLM_API_KEY"):
                self._send_direct_llm(agent, message, llm_cfg)
            else:
                self._send_wrapper(agent, message)

    def _send_direct_llm(self, agent: AgentWrapper, message: str, llm_cfg: "LLMConfig") -> None:
        """Call the LLM directly via OpenAI-compatible API (bypasses AutoGen agent layer)."""
        import asyncio
        import os

        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError:
            # Fall back to raw httpx if openai package not installed
            self._send_direct_httpx(agent, message, llm_cfg)
            return

        api_key = llm_cfg.api_key or os.environ.get("AGENT_LLM_API_KEY", "")
        base_url = llm_cfg.base_url

        # Log connection info at debug level
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "(short/empty)"
        print(f"    [fallback] model={agent.model}  base_url={base_url}  key={masked_key}")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        async def _call() -> str:
            try:
                resp = await client.chat.completions.create(
                    model=agent.model,
                    messages=[
                        {"role": "system", "content": agent.system_prompt},
                        {"role": "user", "content": message},
                    ],
                )
            except Exception as api_err:
                return f"(API error: {api_err})"

            # Debug: print raw response if choices is missing
            if not resp.choices:
                # Try to get error info from response
                raw = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
                print(f"\n  [DEBUG] Raw response: {raw}")
                return "(API returned no choices — see debug output above)"
            return resp.choices[0].message.content or "(empty response)"

        loop = asyncio.new_event_loop()
        try:
            reply = loop.run_until_complete(_call())
        except Exception as e:
            print(f"  LLM call failed: {e}")
            self._send_wrapper(agent, message)
            return
        finally:
            loop.close()

        print(f"\n  \033[1;32m{agent.agent_id}\033[0m:")
        for line in reply.split("\n"):
            print(f"    {line}")
        print()

    def _send_direct_httpx(self, agent: AgentWrapper, message: str, llm_cfg: "LLMConfig") -> None:
        """Fallback: call LLM via raw HTTP if openai package not installed."""
        import json
        import os
        import urllib.request

        api_key = llm_cfg.api_key or os.environ.get("AGENT_LLM_API_KEY", "")
        base = (llm_cfg.base_url or "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")
        url = f"{base}/chat/completions"

        payload = json.dumps({
            "model": agent.model,
            "messages": [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": message},
            ],
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            reply = data["choices"][0]["message"]["content"]
            print(f"\n  \033[1;32m{agent.agent_id}\033[0m:")
            for line in reply.split("\n"):
                print(f"    {line}")
            print()
        except Exception as e:
            print(f"  Direct LLM call failed: {e}")
            self._send_wrapper(agent, message)

    def _send_autogen(self, agent: AgentWrapper, message: str) -> None:
        """Send message via real AutoGen agent and print the response."""
        try:
            import asyncio
            import warnings
            from autogen_agentchat.messages import TextMessage  # type: ignore[import-not-found]
            from autogen_core import CancellationToken  # type: ignore[import-not-found]

            async def _run() -> str:
                ct = CancellationToken()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Resolved model mismatch")
                    response = await agent._inner_agent.on_messages(
                        [TextMessage(content=message, source="user")],
                        cancellation_token=ct,
                    )
                # Handle different response shapes
                if hasattr(response, "chat_message") and response.chat_message:
                    content = response.chat_message.content
                elif hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)
                return content if content else "(empty response)"

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                reply = loop.run_until_complete(_run())
                # Clean up pending tasks to avoid "Task was destroyed" warnings
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                loop.close()

            print(f"\n  \033[1;32m{agent.agent_id}\033[0m:")
            for line in reply.split("\n"):
                print(f"    {line}")
            print()

        except Exception as e:
            print(f"  AutoGen call failed: {e}")
            print("  Falling back to direct LLM call...\n")
            llm_cfg = self.platform._llm_config
            self._send_direct_llm(agent, message, llm_cfg)

    def _send_wrapper(self, agent: AgentWrapper, message: str) -> None:
        """Show what the agent would do (no LLM call)."""
        print(f"\n  \033[1;33m[Simulation — no LLM backend configured]\033[0m")
        print(f"  Agent:  {agent.agent_id} ({agent.name})")
        print(f"  Role:   {agent.role}")
        if agent.specialization:
            print(f"  Spec:   {agent.specialization}")
        print(f"  Model:  {agent.model}")
        print(f"\n  Your message: \"{message}\"")
        print(f"\n  This agent would process your message using:")
        print(f"    Tools: {', '.join(agent.tools)}")
        if agent.domain:
            print(f"    Domain scope: {', '.join(agent.domain)}")
        if agent.assigned_repos:
            print(f"    Repos: {', '.join(agent.assigned_repos)}")

        # Publish to message bus so subscribers can see it
        bus = self.platform.message_bus
        topic = f"cli.user.message"
        msg_id = bus.publish(
            topic, {"from": "user", "to": agent.agent_id, "text": message},
            sender_id="cli-user", _skip_validation=True,
        )
        print(f"\n  Message published to bus (id: {msg_id[:8]}...)")
        print(f"  To get real LLM responses, set OPENAI_API_KEY and ensure autogen-agentchat is installed.")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Platform Interactive CLI")
    parser.add_argument(
        "--manifest", "-m",
        default="agent-manifest.yaml",
        help="Path to YAML agent manifest (default: agent-manifest.yaml)",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING for clean CLI output)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cli = InteractiveCLI(manifest_path=args.manifest)
    cli.run()


if __name__ == "__main__":
    main()
