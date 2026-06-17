"""Multi-agent orchestration layer.

Implements the PM → Tech Lead → Developer conversation flow.
When a task comes in, the PM analyzes it, routes to the right
Tech Lead(s), who then delegate to Developer agents. Responses
flow back up the chain and are consolidated.

Works with any OpenAI-compatible LLM backend (Gemini, OpenAI, etc.)
via the same direct-call approach used in the CLI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.agents.factory import AgentWrapper, LLMConfig

logger = logging.getLogger(__name__)

# Max sequential tool calls per agent turn
MAX_TOOL_CALLS_PER_TURN = 5


@dataclass
class AgentMessage:
    """A message in the multi-agent conversation."""
    agent_id: str
    role: str
    content: str


@dataclass
class TaskResult:
    """Result of a multi-agent task execution."""
    task: str
    messages: list[AgentMessage] = field(default_factory=list)
    final_summary: str = ""

    def add(self, agent_id: str, role: str, content: str) -> None:
        self.messages.append(AgentMessage(agent_id=agent_id, role=role, content=content))


class AgentOrchestrator:
    """Orchestrates multi-agent conversations through the hierarchy.

    Agents are called via direct LLM API (OpenAI-compatible), with each
    agent's system prompt providing its role/personality. The orchestrator
    manages the conversation flow and context passing between agents.
    """

    def __init__(
        self,
        agents: dict[str, AgentWrapper],
        llm_config: LLMConfig,
        on_message: Optional[Callable[[str, str, str], None]] = None,
        tool_registry: Optional[Any] = None,
        context_assembler: Optional[Any] = None,
        output_validator: Optional[Any] = None,
        cost_tracker: Optional[Any] = None,
        retry_policy: Optional[Any] = None,
        health_monitor: Optional[Any] = None,
        task_queue: Optional[Any] = None,
        hip: Optional[Any] = None,
        scoping_gate: Optional[Any] = None,
        qdrant_service: Optional[Any] = None,
        storage: Optional[Any] = None,
        max_context_tokens: int = 128_000,
    ) -> None:
        self._agents = agents
        self._llm_config = llm_config
        self._on_message = on_message
        self._api_key = llm_config.api_key or os.environ.get("AGENT_LLM_API_KEY", "")
        # New components (all optional for backward compat)
        self._tool_registry = tool_registry
        self._context_assembler = context_assembler
        self._output_validator = output_validator
        self._cost_tracker = cost_tracker
        self._retry_policy = retry_policy
        self._health_monitor = health_monitor
        self._task_queue = task_queue
        self._hip = hip
        self._scoping_gate = scoping_gate
        self._qdrant = qdrant_service
        self._storage = storage
        self._max_context_tokens = max_context_tokens
        # Per-task conversation history
        self._task_histories: dict[str, list[dict]] = {}

    async def _call_agent(self, agent: AgentWrapper, user_message: str, task_id: str = "") -> str:
        """Call a single agent via the OpenAI-compatible API."""
        # Use context-enriched system prompt if assembler available
        system_prompt = self._build_system_prompt(agent, task_id)

        try:
            from openai import AsyncOpenAI
        except ImportError:
            return await self._call_agent_httpx(agent, user_message, task_id)

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._llm_config.base_url)
        try:
            resp = await client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            content = "(no response)"
            if resp.choices:
                content = resp.choices[0].message.content or "(empty)"

            # Validate output
            content = self._validate_response(content, agent)

            return content
        except Exception as e:
            logger.error("LLM call failed for %s: %s", agent.agent_id, e)
            return f"(error: {e})"

    async def _call_agent_httpx(self, agent: AgentWrapper, user_message: str, task_id: str = "") -> str:
        """Fallback: call via urllib if openai package not installed."""
        import json as _json
        import urllib.request

        # Use context-enriched system prompt if assembler available
        system_prompt = self._build_system_prompt(agent, task_id)

        base = (self._llm_config.base_url or "").rstrip("/")
        url = f"{base}/chat/completions"
        payload = _json.dumps({
            "model": agent.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def _emit(self, agent_id: str, role: str, content: str) -> None:
        """Emit a message to the callback (for live CLI output)."""
        if self._on_message:
            self._on_message(agent_id, role, content)

    # ── Tool invocation ──────────────────────────────────────────

    async def _call_agent_with_tools(
        self,
        agent: AgentWrapper,
        user_message: str,
        task_id: str = "",
    ) -> str:
        """Call an agent with tool support. Handles tool_calls in responses."""
        messages = [
            {"role": "system", "content": self._build_system_prompt(agent, task_id)},
            {"role": "user", "content": user_message},
        ]

        # Add conversation history for this task (with summarization)
        if task_id and task_id in self._task_histories:
            history = self._maybe_summarize_history(task_id)
            messages = [messages[0]] + history + [messages[-1]]

        tools = self._get_tool_schemas(agent)

        for tool_round in range(MAX_TOOL_CALLS_PER_TURN):
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self._api_key, base_url=self._llm_config.base_url)

                kwargs: dict[str, Any] = {
                    "model": agent.model,
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools

                resp = await client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.error("LLM call failed for %s: %s", agent.agent_id, e)
                return f"(error: {e})"

            if not resp.choices:
                # Some LLM providers return empty choices when tools are included.
                # Retry without tools to get a text response.
                if tools:
                    logger.info(
                        "Empty response with tools for %s, retrying without tools",
                        agent.agent_id,
                    )
                    tools = []  # disable tools for subsequent rounds
                    try:
                        resp = await client.chat.completions.create(
                            model=agent.model, messages=messages,
                        )
                    except Exception as e:
                        logger.error("LLM retry without tools failed for %s: %s", agent.agent_id, e)
                        return f"(error: {e})"
                    if not resp.choices:
                        return "(no response)"
                else:
                    return "(no response)"

            choice = resp.choices[0]

            # Log token usage
            if self._cost_tracker and resp.usage:
                self._cost_tracker.log_usage(
                    agent.agent_id, task_id,
                    resp.usage.prompt_tokens or 0,
                    resp.usage.completion_tokens or 0,
                )

            # Emit heartbeat
            if self._health_monitor:
                from src.infrastructure.health_monitor import HeartbeatEvent
                self._health_monitor.record_heartbeat(HeartbeatEvent(
                    agent_id=agent.agent_id, task_id=task_id,
                    current_action="calling_llm",
                ))

            # Check for tool calls
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message.model_dump())
                for tc in choice.message.tool_calls:
                    tool_result = self._execute_tool_call(tc.function.name, tc.function.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
                    if self._health_monitor:
                        self._health_monitor.record_tool_call(agent.agent_id)
                continue

            # Text response
            content = choice.message.content or "(empty)"
            if self._health_monitor:
                self._health_monitor.reset_tool_calls(agent.agent_id)

            # Validate output against workspace reality
            content = self._validate_response(content, agent)

            # Store in task history
            if task_id:
                self._task_histories.setdefault(task_id, []).append(
                    {"role": "assistant", "content": content}
                )

            return content

        return "(max tool calls reached)"

    def _execute_tool_call(self, tool_name: str, arguments_json: str) -> str:
        """Execute a tool call and return the result as a string."""
        if not self._tool_registry:
            return f"Tool registry not configured, cannot execute {tool_name}"

        try:
            func = self._tool_registry.lookup(tool_name)
            args = json.loads(arguments_json) if arguments_json else {}
            result = func(**args)
            return json.dumps(result, default=str) if not isinstance(result, str) else result
        except KeyError as e:
            return f"Tool not found: {e}"
        except Exception as e:
            logger.warning("Tool %s execution error: %s", tool_name, e)
            return f"Tool execution error: {e}"

    def _get_tool_schemas(self, agent: AgentWrapper) -> list[dict]:
        """Get OpenAI function schemas for an agent's tools."""
        if not self._tool_registry or not agent.tools:
            return []
        return self._tool_registry.get_schemas_for_agent(agent.tools)

    def _validate_response(self, content: str, agent: AgentWrapper) -> str:
        """Run OutputValidator on agent response if available."""
        if not self._output_validator:
            return content

        try:
            for repo in (agent.assigned_repos or []):
                result = self._output_validator.validate(content, repo, agent.agent_id)
                if not result.is_valid and result.invalid_refs:
                    invalid_list = ", ".join(
                        r["reference"] for r in result.invalid_refs[:5]
                    )
                    content += (
                        f"\n\n[Validation note: some references could not be verified "
                        f"in workspace — {invalid_list}]"
                    )
                    logger.info(
                        "OutputValidator flagged %d invalid refs for %s: %s",
                        len(result.invalid_refs), agent.agent_id, invalid_list,
                    )
                    break
        except Exception as e:
            logger.warning("Output validation failed for %s: %s", agent.agent_id, e)

        return content

    def _build_system_prompt(self, agent: AgentWrapper, task_id: str = "") -> str:
        """Build system prompt with optional context assembly."""
        if self._context_assembler:
            try:
                return self._context_assembler.assemble(
                    task_description=task_id,
                    agent_role=agent.role,
                    agent_id=agent.agent_id,
                    assigned_repos=agent.assigned_repos,
                    system_prompt=agent.system_prompt,
                )
            except Exception as e:
                logger.warning("Context assembly failed: %s", e)
        return agent.system_prompt

    # ── Context window management ────────────────────────────────

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4

    def _maybe_summarize_history(self, task_id: str) -> list[dict]:
        """Return history, summarizing older messages if exceeding 80% of context window."""
        history = self._task_histories.get(task_id, [])
        if not history:
            return []

        threshold = int(self._max_context_tokens * 0.8)
        estimated = self._estimate_tokens(history)

        if estimated <= threshold:
            return list(history)

        # Keep the last 5 messages intact, summarize the rest
        keep_count = min(5, len(history))
        recent = history[-keep_count:]
        older = history[:-keep_count]

        if not older:
            return list(history)

        # Build a condensed summary of older messages
        summary_parts = []
        for msg in older:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Take first 200 chars of each old message
            snippet = content[:200].replace("\n", " ")
            if len(content) > 200:
                snippet += "..."
            summary_parts.append(f"[{role}]: {snippet}")

        summary_text = (
            "Summary of earlier conversation:\n" + "\n".join(summary_parts)
        )

        # Replace older messages with a single summary message
        summarized = [{"role": "system", "content": summary_text}] + recent

        # Update stored history with the compacted version
        self._task_histories[task_id] = summarized

        return summarized

    # ── Task persistence ─────────────────────────────────────────

    def persist_task_state(self, task_id: str) -> None:
        """Persist in-progress task conversation history."""
        if self._storage and task_id in self._task_histories:
            self._storage.save("task_conversations", task_id, {
                "task_id": task_id,
                "messages": self._task_histories[task_id],
            })

    def load_task_state(self, task_id: str) -> bool:
        """Load persisted task state. Returns True if found."""
        if self._storage:
            data = self._storage.load("task_conversations", task_id)
            if data:
                self._task_histories[task_id] = data.get("messages", [])
                return True
        return False

    # ── Agent lookups ────────────────────────────────────────────

    def _get_pm(self) -> Optional[AgentWrapper]:
        for a in self._agents.values():
            if a.role == "pm_agent":
                return a
        return None

    def _get_tech_leads(self) -> list[AgentWrapper]:
        return [a for a in self._agents.values() if a.role == "tech_lead_agent"]

    def _get_tech_lead_for_domain(self, domain: str) -> Optional[AgentWrapper]:
        for a in self._agents.values():
            if a.role == "tech_lead_agent" and domain in (a.domain or []):
                return a
        return None

    def _get_devs_for_domain(self, domain: str) -> list[AgentWrapper]:
        return [
            a for a in self._agents.values()
            if a.role == "developer_agent" and domain in (a.domain or [])
        ]

    def _get_all_domains(self) -> list[str]:
        domains = set()
        for a in self._agents.values():
            for d in (a.domain or []):
                domains.add(d)
        return sorted(domains)

    # ── Orchestration flows ──────────────────────────────────────

    async def _smart_call(self, agent: AgentWrapper, user_message: str, task_id: str = "") -> str:
        """Call agent with tools+context if available, otherwise plain LLM call.

        Falls back to plain LLM call (no tools) if the tool-enhanced call
        returns empty/no response — some LLM providers don't support function calling.
        """
        if self._tool_registry or self._context_assembler:
            try:
                result = await self._call_agent_with_tools(agent, user_message, task_id)
                # If the LLM returned empty/no response with tools, retry without tools
                if result in ("(no response)", "(empty)"):
                    logger.info(
                        "Tool-enhanced call returned empty for %s, retrying without tools",
                        agent.agent_id,
                    )
                    return await self._call_agent(agent, user_message, task_id)
                return result
            except Exception as e:
                logger.warning("Enhanced call failed for %s, falling back: %s", agent.agent_id, e)
        return await self._call_agent(agent, user_message, task_id)

    async def run_task(self, task_description: str) -> TaskResult:
        """Run a full PM → Tech Lead → Developer workflow.

        1. PM analyzes the task and decides which domains are affected
        2. Relevant Tech Lead(s) break it down into sub-tasks
        3. Developer agents provide technical analysis
        4. PM consolidates everything into a final summary
        """
        result = TaskResult(task=task_description)
        domains = self._get_all_domains()
        tl_list = ", ".join(f"{tl.agent_id} ({', '.join(tl.domain)})" for tl in self._get_tech_leads())
        dev_list = "\n".join(
            f"  - {a.agent_id}: role={a.role}, spec={a.specialization}, "
            f"domain={','.join(a.domain)}, repos={','.join(a.assigned_repos)}"
            for a in self._agents.values()
            if a.role == "developer_agent"
        )

        # ── Step 1: PM analyzes and routes ───────────────────────
        pm = self._get_pm()
        if not pm:
            result.final_summary = "No PM agent found."
            return result

        pm_prompt = (
            f"You are coordinating a multi-agent platform task. Here is the team:\n"
            f"Domains: {', '.join(domains)}\n"
            f"Tech Leads: {tl_list}\n"
            f"Developers:\n{dev_list}\n\n"
            f"Task: {task_description}\n\n"
            f"Analyze this task and respond with:\n"
            f"1. Which domain(s) are affected (choose from: {', '.join(domains)})\n"
            f"2. A brief analysis of what needs to happen, referencing the code context provided above\n"
            f"3. What you want each affected Tech Lead to do\n\n"
            f"IMPORTANT: Base your analysis on the code snippets and knowledge provided in your context. "
            f"If you don't have enough information, say so rather than guessing.\n\n"
            f"In your response, include a line starting with "
            f"'DOMAINS:' followed by a comma-separated list of affected domains "
            f"(e.g., 'DOMAINS: inbound, outbound'). This is used for routing."
        )

        self._emit(pm.agent_id, "PM Agent", "Analyzing task and routing...")
        pm_response = await self._smart_call(pm, pm_prompt, task_description)
        result.add(pm.agent_id, "PM Agent", pm_response)
        self._emit(pm.agent_id, "PM Agent", pm_response)

        # Parse affected domains from PM response
        affected_domains = self._parse_domains(pm_response, domains)
        if not affected_domains:
            affected_domains = domains  # fallback: involve all domains

        # ── Step 2: Tech Leads break down the work ───────────────
        tl_responses: dict[str, str] = {}
        for domain in affected_domains:
            tl = self._get_tech_lead_for_domain(domain)
            if not tl:
                continue

            devs_in_domain = self._get_devs_for_domain(domain)
            dev_info = "\n".join(
                f"  - {d.agent_id}: spec={d.specialization}, repos={','.join(d.assigned_repos)}"
                for d in devs_in_domain
            )

            tl_prompt = (
                f"The PM Agent has assigned you this task for the {domain} domain.\n\n"
                f"PM's analysis:\n{pm_response}\n\n"
                f"Original task: {task_description}\n\n"
                f"Your developer agents:\n{dev_info}\n\n"
                f"Use the code context provided above to understand the current codebase structure. "
                f"Break this down into specific sub-tasks for your developers. "
                f"For each sub-task, specify which developer agent should handle it "
                f"and what exactly they need to do. Reference actual code paths from the context "
                f"when possible."
            )

            self._emit(tl.agent_id, f"Tech Lead ({domain})", "Breaking down task...")
            tl_resp = await self._smart_call(tl, tl_prompt, task_description)
            tl_responses[domain] = tl_resp
            result.add(tl.agent_id, f"Tech Lead ({domain})", tl_resp)
            self._emit(tl.agent_id, f"Tech Lead ({domain})", tl_resp)

        # ── Step 3: Developer agents provide technical analysis ──
        dev_responses: list[tuple[str, str]] = []
        for domain in affected_domains:
            tl_resp = tl_responses.get(domain, "")
            devs = self._get_devs_for_domain(domain)

            for dev in devs:
                dev_prompt = (
                    f"Your Tech Lead has broken down a task for the {domain} domain.\n\n"
                    f"Tech Lead's breakdown:\n{tl_resp}\n\n"
                    f"Original task: {task_description}\n\n"
                    f"You are {dev.agent_id}, specialization: {dev.specialization}, "
                    f"assigned to repos: {', '.join(dev.assigned_repos)}.\n\n"
                    f"IMPORTANT: Before answering, use your tools to examine the actual code:\n"
                    f"  - Use search_code to find relevant code in your repos\n"
                    f"  - Use read_file to read specific files you find\n"
                    f"  - Use grep to search for patterns in the codebase\n"
                    f"  - Use find_references to trace symbol usage\n\n"
                    f"Base your analysis ONLY on what you find in the actual code. "
                    f"Do NOT guess file names, paths, or structures.\n\n"
                    f"Provide your technical analysis:\n"
                    f"1. What specific files/functions did you find that are relevant? (cite actual paths)\n"
                    f"2. What changes are needed and where exactly?\n"
                    f"3. Estimated complexity (low/medium/high)\n"
                    f"4. Any risks or dependencies to flag"
                )

                self._emit(dev.agent_id, f"Developer ({dev.specialization})", "Analyzing...")
                dev_resp = await self._smart_call(dev, dev_prompt, task_description)
                dev_responses.append((dev.agent_id, dev_resp))
                result.add(dev.agent_id, f"Developer ({dev.specialization})", dev_resp)
                self._emit(dev.agent_id, f"Developer ({dev.specialization})", dev_resp)

        # ── Step 4: PM consolidates ──────────────────────────────
        all_responses = "\n\n".join(
            f"--- {msg.role} ({msg.agent_id}) ---\n{msg.content}"
            for msg in result.messages
        )

        summary_prompt = (
            f"You asked your team to analyze this task: {task_description}\n\n"
            f"Here are all the responses from your team:\n\n{all_responses}\n\n"
            f"Now provide a consolidated summary. Flag any responses that appear to reference "
            f"files or structures that weren't confirmed by tool calls or code context.\n\n"
            f"1. Overall plan and timeline\n"
            f"2. Key decisions and assignments\n"
            f"3. Risks and dependencies\n"
            f"4. Recommended next steps"
        )

        self._emit(pm.agent_id, "PM Agent", "Consolidating team responses...")
        summary = await self._smart_call(pm, summary_prompt, task_description)
        result.final_summary = summary
        result.add(pm.agent_id, "PM Agent (Summary)", summary)
        self._emit(pm.agent_id, "PM Agent (Summary)", summary)

        return result

    @staticmethod
    def _parse_domains(pm_response: str, valid_domains: list[str]) -> list[str]:
        """Extract domain names from PM's response."""
        found = []
        for line in pm_response.split("\n"):
            if line.strip().upper().startswith("DOMAINS:"):
                parts = line.split(":", 1)[1]
                for token in parts.split(","):
                    token = token.strip().lower()
                    if token in valid_domains:
                        found.append(token)
                break

        # Fallback: scan entire response for domain names
        if not found:
            lower_resp = pm_response.lower()
            for d in valid_domains:
                if d in lower_resp:
                    found.append(d)

        return found
