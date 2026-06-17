"""Integration tests for end-to-end workflows.

Tests cross-component interactions using in-memory implementations.
"""

import os
import pytest

from src.agents.bug_scanner import BugScanner
from src.agents.context_manager import ContextManager
from src.agents.factory import AgentConfig, AgentFactory
from src.agents.feature_scoper import FeatureScoper
from src.agents.impact_analyzer import ImpactAnalyzer
from src.core.enums import ApprovalDecision, BugSeverity
from src.core.models import AgentRecord, BugReport
from src.infrastructure.agent_registry import AgentRegistry
from src.infrastructure.approval_gate import ApprovalGateService
from src.infrastructure.discussion_thread import DiscussionThreadManager
from src.infrastructure.message_bus import MessageBus
from src.infrastructure.provisioner import DynamicProvisioner
from src.infrastructure.role_registry import RoleRegistry
from src.knowledge.qdrant_service import QdrantService
from src.observability.audit import AuditTrail, MetricsCollector


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_MANIFEST = os.path.join(FIXTURE_DIR, "test_manifest.yaml")


@pytest.fixture
def platform():
    """Set up a complete platform with all components."""
    rr = RoleRegistry()
    ar = AgentRegistry()
    factory = AgentFactory(rr)
    bus = MessageBus()
    qdrant = QdrantService(vector_size=5)
    provisioner = DynamicProvisioner(rr, ar, factory, bus, qdrant)
    discussion = DiscussionThreadManager(qdrant_service=qdrant, pm_agent_id="pm-001")
    approval = ApprovalGateService()
    audit = AuditTrail()
    metrics = MetricsCollector()
    context_mgr = ContextManager(qdrant)

    # Provision from manifest
    agents = provisioner.provision_from_manifest(TEST_MANIFEST)

    return {
        "role_registry": rr,
        "agent_registry": ar,
        "factory": factory,
        "bus": bus,
        "qdrant": qdrant,
        "provisioner": provisioner,
        "discussion": discussion,
        "approval": approval,
        "audit": audit,
        "metrics": metrics,
        "context_mgr": context_mgr,
        "agents": agents,
    }


# ──────────────────────────────────────────────
# 18.1 End-to-end agent provisioning from YAML manifest
# ──────────────────────────────────────────────


class TestEndToEndProvisioning:

    def test_all_agents_provisioned_and_registered(self, platform):
        agents = platform["agents"]
        ar = platform["agent_registry"]
        assert len(agents) >= 4
        for agent in agents:
            record = ar.get_agent(agent.agent_id)
            assert record.role == agent.role

    def test_roles_loaded(self, platform):
        rr = platform["role_registry"]
        assert rr.get_role("pm_agent") is not None
        assert rr.get_role("developer_agent") is not None
        assert rr.get_role("db_admin") is not None


# ──────────────────────────────────────────────
# 18.2 Message bus publish → subscribe → deliver
# ──────────────────────────────────────────────


class TestMessageBusIntegration:

    def test_publish_subscribe_deliver(self, platform):
        bus = platform["bus"]
        received = []
        bus.subscribe("inbound.receiving-service.*", lambda msg: received.append(msg))
        bus.publish("inbound.receiving-service.bug_detected", {"bug": "null_ref"}, "dev-001")
        assert len(received) == 1
        assert received[0].payload["bug"] == "null_ref"

    def test_dead_letter_queue_on_failure(self, platform):
        bus = platform["bus"]
        bus.subscribe("inbound.test-svc.event", lambda msg: (_ for _ in ()).throw(RuntimeError("fail")))
        bus.publish("inbound.test-svc.event", {}, "dev-001")
        dlq = bus.get_dead_letter_queue()
        assert len(dlq) >= 1


# ──────────────────────────────────────────────
# 18.3 Qdrant index → search round-trip
# ──────────────────────────────────────────────


class TestQdrantIntegration:

    def test_code_index_round_trip(self, platform):
        qdrant = platform["qdrant"]
        chunks = [{
            "chunk_id": "c1", "file_path": "src/main.py", "language": "python",
            "commit_hash": "abc123", "chunk_type": "function", "content": "def hello(): pass",
            "start_line": 1, "end_line": 1, "embedding": [0.5, 0.3, 0.8, 0.1, 0.6],
        }]
        count = qdrant.index_code("receiving-service", chunks)
        assert count == 1
        results = qdrant.search_code([0.5, 0.3, 0.8, 0.1, 0.6], repo_filter="receiving-service")
        assert len(results) >= 1
        assert results[0].file_path == "src/main.py"

    def test_knowledge_base_store_search(self, platform):
        qdrant = platform["qdrant"]
        emb = [0.5, 0.3, 0.8, 0.1, 0.6]
        qdrant.store_knowledge(emb, {"type": "decision", "content": "Use Redis for bus"})
        results = qdrant.search_knowledge(emb, top_k=5)
        assert len(results) >= 1


# ──────────────────────────────────────────────
# 18.4 Bug detection → fix → PR → approval pipeline
# ──────────────────────────────────────────────


class TestBugFixPipeline:

    def test_full_bug_pipeline(self, platform):
        audit = platform["audit"]
        approval = platform["approval"]
        discussion = platform["discussion"]

        # 1. Scan code and detect bug
        scanner = BugScanner(agent_id="dev-recv-001")
        code = "try:\n    process()\nexcept:\n    pass\n"
        bugs = scanner.scan_content(code, "src/handler.py", "receiving-service")
        assert len(bugs) >= 1

        # 2. Log detection in audit trail
        cid = "corr-bug-001"
        audit.log_action("dev-recv-001", "code_analysis", cid, {"bugs_found": len(bugs)})

        # 3. Create discussion thread for high-severity bug
        bug = bugs[0]
        thread = discussion.create_thread(
            topic=f"Bug: {bug.category} in {bug.file_path}",
            creator_id="dev-recv-001",
            invited_agents=["dev-recv-001", "tl-inbound-001"],
        )

        # 4. Agents contribute
        discussion.add_contribution(thread.thread_id, "dev-recv-001", "Found bare except, proposing fix")
        discussion.add_contribution(thread.thread_id, "tl-inbound-001", "Approved, create PR")

        # 5. Synthesize
        summary = discussion.synthesize(thread.thread_id)
        assert summary

        # 6. Request approval for PR merge
        req = approval.request_approval(
            action_type="merge_pr",
            context={"pr_url": "https://github.com/org/repo/pull/42", "thread_id": thread.thread_id},
            requester_id="dev-recv-001",
            reviewer_id="human-reviewer-001",
        )
        assert req.decision == ApprovalDecision.PENDING

        # 7. Human approves
        approval.submit_response(req.request_id, ApprovalDecision.APPROVED)
        assert approval.get_request(req.request_id).decision == ApprovalDecision.APPROVED

        # 8. Log in audit
        audit.log_action("dev-recv-001", "tool_invocation", cid, {"action": "merge_pr", "pr_id": "PR-42"})


# ──────────────────────────────────────────────
# 18.5 Cross-domain impact analysis
# ──────────────────────────────────────────────


class TestCrossDomainImpact:

    def test_impact_analysis_flow(self, platform):
        analyzer = ImpactAnalyzer()
        discussion = platform["discussion"]

        # Detect cross-domain change
        assert analyzer.detect_cross_domain_change(["shared/models.py"])

        # Compile report
        report = analyzer.compile_impact_report(
            trigger_agent_id="dev-recv-001",
            change_description="Updated shared Order model",
            affected_repos=["receiving-service", "picking-service"],
            affected_endpoints=["/api/v1/orders"],
            estimated_effort="high",
            risk_level="critical",
            has_breaking_changes=True,
        )
        assert report.has_breaking_changes

        # Create cross-domain discussion
        thread = discussion.create_thread(
            topic="Breaking change: Order model update",
            creator_id="pm-001",
            invited_agents=["tl-inbound-001", "tl-outbound-001"],
            is_cross_domain=True,
        )
        assert thread.moderator_id == "pm-001"


# ──────────────────────────────────────────────
# 18.6 Feature scoping → story breakdown → DevOps
# ──────────────────────────────────────────────


class TestFeatureScopingPipeline:

    def test_feature_breakdown_to_devops(self, platform):
        scoper = FeatureScoper()
        approval = platform["approval"]

        # Create breakdown
        stories = [
            {"title": "Add receiving endpoint", "repo": "receiving-service",
             "assigned_agent_id": "dev-recv-001", "acceptance_criteria": ["Returns 200"]},
            {"title": "Update UI form", "repo": "web-app",
             "assigned_agent_id": "dev-ui-001", "acceptance_criteria": ["Form validates"]},
        ]
        bd = scoper.create_breakdown("New receiving feature", "tl-inbound-001", stories)
        assert len(bd.stories) == 2

        # PM approves
        scoper.approve_breakdown(bd.breakdown_id)
        assert bd.approved_by_pm

        # Send to DevOps
        scoper.send_to_devops(bd.breakdown_id)
        assert bd.sent_to_devops


# ──────────────────────────────────────────────
# 18.7 Dynamic provisioning hot-reload
# ──────────────────────────────────────────────


class TestHotReloadIntegration:

    def test_hot_reload_no_change(self, platform):
        provisioner = platform["provisioner"]
        new_agents = provisioner.check_hot_reload(TEST_MANIFEST)
        assert len(new_agents) == 0


# ──────────────────────────────────────────────
# 18.8 Custom role provisioning
# ──────────────────────────────────────────────


class TestCustomRoleProvisioning:

    def test_db_admin_provisioned(self, platform):
        agents = platform["agents"]
        db_admins = [a for a in agents if a.role == "db_admin"]
        assert len(db_admins) >= 1

    def test_custom_role_has_correct_tools(self, platform):
        rr = platform["role_registry"]
        tools = rr.get_tools_for_role("db_admin")
        assert "DBMigrationChecker.scan" in tools
