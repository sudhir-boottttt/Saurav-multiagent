"""Core Pydantic data models for the Multi-Agent Platform."""

from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .enums import (
    AgentStatus,
    ApprovalDecision,
    BugSeverity,
    HIPUrgency,
    RepoStatus,
    TaskPriority,
    TaskState,
)


# --- Role & Agent Models ---


class RoleDefinition(BaseModel):
    """Defines an agent role loaded from the YAML manifest."""

    role_name: str = Field(..., min_length=1)
    tier: str = Field(..., min_length=1)  # "top", "middle", "bottom", "cross-domain"
    capabilities: list[str] = Field(..., min_length=1)
    tools: list[str] = Field(default_factory=list)
    system_prompt_template: str = Field(default="")
    allowed_parent_roles: list[str] = Field(default_factory=list)
    allowed_domains: Optional[list[str]] = None
    is_builtin: bool = False


class AgentRecord(BaseModel):
    """Registry record for a running agent instance."""

    agent_id: str = Field(default_factory=lambda: str(uuid4()))
    role: str = Field(..., min_length=1)
    specialization: Optional[str] = None
    domain: Optional[list[str]] = None
    assigned_repos: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    system_prompt_template: str = Field(default="")
    status: AgentStatus = AgentStatus.STARTING
    parent_agent_id: Optional[str] = None
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config: dict[str, Any] = Field(default_factory=dict)


# --- Message Bus Models ---


class BusMessage(BaseModel):
    """A message published to the event-driven message bus."""

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str = Field(..., min_length=1)
    sender_id: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    retry_count: int = Field(default=0, ge=0, le=3)


# --- Bug Detection Models ---


class BugReport(BaseModel):
    """A bug detected by a Developer Agent."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    line_start: int = Field(..., ge=1)
    line_end: int = Field(..., ge=1)
    category: str = Field(..., min_length=1)
    severity: BugSeverity
    description: str = Field(..., min_length=1)
    proposed_fix: Optional[str] = None
    discussion_thread_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Discussion Thread Models ---


class CodeReference(BaseModel):
    """A reference to a code location."""

    file_path: str = Field(..., min_length=1)
    line_start: int = Field(..., ge=1)
    line_end: int = Field(..., ge=1)


class Contribution(BaseModel):
    """A single agent contribution to a discussion thread."""

    agent_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content: str = Field(..., min_length=1)
    code_refs: list[CodeReference] = Field(default_factory=list)


class DiscussionThread(BaseModel):
    """A structured multi-agent conversation."""

    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str = Field(..., min_length=1)
    creator_id: str = Field(..., min_length=1)
    invited_agents: list[str] = Field(default_factory=list)
    contributions: list[Contribution] = Field(default_factory=list)
    is_cross_domain: bool = False
    moderator_id: str = Field(..., min_length=1)
    summary: Optional[str] = None
    status: str = Field(default="open")  # "open" | "synthesized" | "closed"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closed_at: Optional[datetime] = None


# --- Approval Gate Models ---


class ApprovalRequest(BaseModel):
    """A human-in-the-loop approval request."""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    action_type: str = Field(..., min_length=1)
    requester_id: str = Field(..., min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    discussion_thread_id: Optional[str] = None
    reviewer_id: str = Field(..., min_length=1)
    secondary_reviewer_id: Optional[str] = None
    decision: ApprovalDecision = ApprovalDecision.PENDING
    rejection_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    decided_at: Optional[datetime] = None
    deadline: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Knowledge Layer Models ---


class CodeChunk(BaseModel):
    """A code chunk stored in Qdrant Code_Index."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    repo: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    chunk_type: str = Field(..., min_length=1)  # "function" | "class" | "module"
    language: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    commit_hash: str = Field(..., min_length=1)
    embedding: list[float] = Field(default_factory=list)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Impact Analysis Models ---


class ImpactReport(BaseModel):
    """Cross-domain impact analysis report."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    trigger_agent_id: str = Field(..., min_length=1)
    change_description: str = Field(..., min_length=1)
    affected_repos: list[str] = Field(..., min_length=1)
    affected_endpoints: list[str] = Field(..., min_length=1)
    estimated_effort: str = Field(..., min_length=1)
    risk_level: str = Field(..., min_length=1)
    has_breaking_changes: bool = False
    migration_plan_thread_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Observability Models ---


class AuditLogEntry(BaseModel):
    """An audit log entry for agent actions."""

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    action_type: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    decision_rationale: Optional[str] = None
    retrieved_context_refs: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Task Queue Models ---


class TaskEntry(BaseModel):
    """A task in the task queue."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = Field(..., min_length=1)
    priority: TaskPriority = TaskPriority.NORMAL
    state: TaskState = TaskState.QUEUED
    depends_on: list[str] = Field(default_factory=list)
    repos_involved: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    messages: list[dict] = Field(default_factory=list)
    current_step: str = ""
    token_usage: int = 0


# --- Tool Registry Models ---


class ToolDefinition(BaseModel):
    """A registered tool with its schema."""

    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    parameters_schema: dict = Field(default_factory=dict)
    callable_ref: str = Field(default="")


# --- Verification Models ---


class VerificationResult(BaseModel):
    """Result of running validation commands on a code change."""

    passed: bool
    command_results: list[dict] = Field(default_factory=list)


# --- Repository State Models ---


class RepoState(BaseModel):
    """Tracks the state of a cloned repository."""

    repo_name: str = Field(..., min_length=1)
    repo_url: str = Field(default="")
    local_path: str = Field(default="")
    branch: str = Field(default="main")
    last_commit_hash: str = Field(default="")
    last_indexed_commit: str = Field(default="")
    status: RepoStatus = RepoStatus.CLONED
    language: str = Field(default="python")
    chunk_count: int = 0
    cloned_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None


class RuntimeConfig(BaseModel):
    """Per-repository language and toolchain configuration."""

    language: str = Field(..., min_length=1)
    package_manager: str = Field(default="")
    install_command: str = Field(default="")
    test_command: str = Field(default="")
    lint_command: str = Field(default="")
    build_command: str = Field(default="")


# --- Cost Tracking Models ---


class CostRecord(BaseModel):
    """Token usage record for a single LLM call."""

    record_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str = Field(..., min_length=1)
    task_id: str = Field(default="")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Human Interaction Models ---


class HIPRecord(BaseModel):
    """Persisted Human Interaction Point record."""

    hip_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str = Field(..., min_length=1)
    context: dict = Field(default_factory=dict)
    urgency: HIPUrgency = HIPUrgency.NORMAL
    agent_id: str = Field(..., min_length=1)
    task_id: str = Field(default="")
    response: Optional[str] = None
    responded_at: Optional[datetime] = None
    responded_via: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    timeout_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
