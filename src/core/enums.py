"""Core enums for the Multi-Agent Platform."""

from enum import Enum


class AgentStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPED = "stopped"


class BugSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    ESCALATED = "escalated"


class TaskState(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    WAITING_ON_HUMAN = "waiting_on_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALLED = "stalled"


class TaskPriority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    COMMENT = "comment"


class HIPUrgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class RepoStatus(str, Enum):
    CLONED = "cloned"
    INDEXING = "indexing"
    READY = "ready"
    DEPS_FAILED = "deps_failed"
    ERROR = "error"
