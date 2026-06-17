"""Hypothesis strategies (generators) for all core data models."""

from datetime import datetime, timedelta

from hypothesis import strategies as st

from src.core.enums import AgentStatus, ApprovalDecision, BugSeverity

# --- Reusable primitives ---

uuid_st = st.uuids().map(str)
non_empty_str = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P")))
short_str = st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))
timestamp_st = st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 1, 1))
embedding_st = st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1.0, max_value=1.0), min_size=3, max_size=10)


# --- RoleDefinition ---

role_definition_st = st.builds(
    dict,
    role_name=short_str,
    tier=st.sampled_from(["top", "middle", "bottom", "cross-domain"]),
    capabilities=st.lists(short_str, min_size=1, max_size=5),
    tools=st.lists(short_str, min_size=0, max_size=5),
    system_prompt_template=st.text(max_size=100),
    allowed_parent_roles=st.lists(short_str, min_size=0, max_size=3),
    allowed_domains=st.one_of(st.none(), st.lists(st.sampled_from(["inbound", "outbound"]), min_size=1, max_size=2)),
    is_builtin=st.booleans(),
)


# --- AgentRecord ---

agent_record_st = st.builds(
    dict,
    agent_id=uuid_st,
    role=short_str,
    specialization=st.one_of(st.none(), st.sampled_from(["frontend", "backend", "fullstack", "ui", "devops"])),
    domain=st.one_of(st.none(), st.lists(st.sampled_from(["inbound", "outbound"]), min_size=1, max_size=2)),
    assigned_repos=st.lists(non_empty_str, min_size=0, max_size=3),
    capabilities=st.lists(short_str, min_size=0, max_size=5),
    tools=st.lists(short_str, min_size=0, max_size=5),
    status=st.sampled_from(list(AgentStatus)),
    parent_agent_id=st.one_of(st.none(), uuid_st),
)


# --- BusMessage ---

topic_st = st.builds(
    lambda d, r, e: f"{d}.{r}.{e}",
    d=st.sampled_from(["inbound", "outbound"]),
    r=short_str,
    e=short_str,
)

bus_message_st = st.builds(
    dict,
    message_id=uuid_st,
    topic=topic_st,
    sender_id=uuid_st,
    payload=st.fixed_dictionaries({"key": non_empty_str}),
    timestamp=timestamp_st,
    correlation_id=uuid_st,
    retry_count=st.integers(min_value=0, max_value=3),
)


# --- BugReport ---

bug_report_st = st.builds(
    dict,
    report_id=uuid_st,
    agent_id=uuid_st,
    repo=short_str,
    file_path=st.builds(lambda n: f"src/{n}.py", n=short_str),
    line_start=st.integers(min_value=1, max_value=500),
    line_end=st.integers(min_value=1, max_value=1000),
    category=st.sampled_from(["null_reference", "unhandled_exception", "api_mismatch", "dependency_conflict"]),
    severity=st.sampled_from(list(BugSeverity)),
    description=non_empty_str,
    proposed_fix=st.one_of(st.none(), non_empty_str),
    discussion_thread_id=st.one_of(st.none(), uuid_st),
)


# --- CodeReference ---

code_reference_st = st.builds(
    dict,
    file_path=st.builds(lambda n: f"src/{n}.py", n=short_str),
    line_start=st.integers(min_value=1, max_value=500),
    line_end=st.integers(min_value=1, max_value=1000),
)


# --- Contribution ---

contribution_st = st.builds(
    dict,
    agent_id=uuid_st,
    timestamp=timestamp_st,
    content=non_empty_str,
    code_refs=st.lists(code_reference_st, min_size=0, max_size=3),
)


# --- DiscussionThread ---

discussion_thread_st = st.builds(
    dict,
    thread_id=uuid_st,
    topic=non_empty_str,
    creator_id=uuid_st,
    invited_agents=st.lists(uuid_st, min_size=1, max_size=5),
    contributions=st.lists(contribution_st, min_size=0, max_size=5),
    is_cross_domain=st.booleans(),
    moderator_id=uuid_st,
    summary=st.one_of(st.none(), non_empty_str),
    status=st.sampled_from(["open", "synthesized", "closed"]),
)


# --- ApprovalRequest ---

approval_request_st = st.builds(
    dict,
    request_id=uuid_st,
    action_type=st.sampled_from(["merge_pr", "create_prod_work_item", "cross_domain_migration", "agent_config_change"]),
    requester_id=uuid_st,
    context=st.fixed_dictionaries({"detail": non_empty_str}),
    discussion_thread_id=st.one_of(st.none(), uuid_st),
    reviewer_id=uuid_st,
    secondary_reviewer_id=st.one_of(st.none(), uuid_st),
    decision=st.sampled_from(list(ApprovalDecision)),
    rejection_reason=st.one_of(st.none(), non_empty_str),
)


# --- CodeChunk ---

code_chunk_st = st.builds(
    dict,
    chunk_id=uuid_st,
    repo=short_str,
    file_path=st.builds(lambda n: f"src/{n}.py", n=short_str),
    chunk_type=st.sampled_from(["function", "class", "module"]),
    language=st.sampled_from(["python", "csharp", "javascript", "typescript"]),
    content=non_empty_str,
    start_line=st.integers(min_value=1, max_value=500),
    end_line=st.integers(min_value=1, max_value=1000),
    commit_hash=st.text(min_size=7, max_size=40, alphabet="0123456789abcdef"),
    embedding=embedding_st,
)


# --- ImpactReport ---

impact_report_st = st.builds(
    dict,
    report_id=uuid_st,
    trigger_agent_id=uuid_st,
    change_description=non_empty_str,
    affected_repos=st.lists(short_str, min_size=1, max_size=5),
    affected_endpoints=st.lists(non_empty_str, min_size=1, max_size=5),
    estimated_effort=st.sampled_from(["low", "medium", "high"]),
    risk_level=st.sampled_from(["low", "medium", "high", "critical"]),
    has_breaking_changes=st.booleans(),
    migration_plan_thread_id=st.one_of(st.none(), uuid_st),
)


# --- AuditLogEntry ---

audit_log_entry_st = st.builds(
    dict,
    entry_id=uuid_st,
    correlation_id=uuid_st,
    agent_id=uuid_st,
    action_type=st.sampled_from(["message_send", "tool_invocation", "code_analysis", "work_item_op", "decision"]),
    payload=st.fixed_dictionaries({"data": non_empty_str}),
    decision_rationale=st.one_of(st.none(), non_empty_str),
    retrieved_context_refs=st.lists(uuid_st, min_size=0, max_size=5),
)
