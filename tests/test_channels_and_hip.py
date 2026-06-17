"""Tests for CommunicationChannel, CLIChannel, NotificationProvider, HIP, and ScopingGate."""

from src.agents.scoping_gate import ScopingConfirmationGate
from src.infrastructure.channels import CLIChannel, CommunicationChannel
from src.infrastructure.human_interaction import HumanInteractionPoint
from src.infrastructure.notifications import ConsoleLogProvider, NotificationProvider


# --- CLIChannel ---

def test_cli_channel_protocol():
    assert isinstance(CLIChannel(), CommunicationChannel)


def test_cli_send_and_respond():
    ch = CLIChannel()
    msg_id = ch.send_question("Approve?", {}, "user", "normal")
    assert ch.poll_response(msg_id) is None
    ch.submit_response(msg_id, "yes")
    resp = ch.poll_response(msg_id)
    assert resp is not None
    assert resp.text == "yes"
    assert resp.channel_name == "cli"


def test_cli_pending_questions():
    ch = CLIChannel()
    ch.send_question("Q1?", {}, "user", "normal")
    ch.send_question("Q2?", {}, "user", "high")
    assert len(ch.get_pending_questions()) == 2


# --- NotificationProvider ---

def test_console_log_protocol():
    assert isinstance(ConsoleLogProvider(), NotificationProvider)


def test_console_log_notify():
    p = ConsoleLogProvider()
    p.notify("task_completed", {"task_id": "t1"})
    assert len(p.get_history()) == 1
    assert p.get_history()[0]["event_type"] == "task_completed"


# --- HumanInteractionPoint ---

def test_hip_request_and_respond():
    ch = CLIChannel()
    hip = HumanInteractionPoint(channels={"cli": ch})
    record = hip.request_input("Approve scope?", {"task": "test"}, "normal", "agent-1", "task-1")
    assert record.hip_id
    assert record.response is None

    # No response yet
    assert hip.check_response(record.hip_id) is None

    # Submit response via CLI
    pending = ch.get_pending_questions()
    assert len(pending) == 1
    ch.submit_response(pending[0]["message_id"], "approved")

    # Now check response
    response = hip.check_response(record.hip_id)
    assert response == "approved"


def test_hip_get_pending():
    ch = CLIChannel()
    hip = HumanInteractionPoint(channels={"cli": ch})
    hip.request_input("Q1?", {}, "normal", "a1", "t1")
    hip.request_input("Q2?", {}, "high", "a2", "t2")
    assert len(hip.get_pending()) == 2


def test_hip_no_channels():
    hip = HumanInteractionPoint(channels={})
    record = hip.request_input("Q?", {}, "normal", "a1", "t1")
    assert record.hip_id
    assert hip.check_response(record.hip_id) is None


# --- ScopingConfirmationGate ---

def test_scoping_gate_disabled():
    gate = ScopingConfirmationGate(enabled=False)
    result = gate.present_scope({"task": "test"}, "t1")
    assert result is None


def test_scoping_gate_no_hip():
    gate = ScopingConfirmationGate(hip=None, enabled=True)
    result = gate.present_scope({"task": "test"}, "t1")
    assert result is None  # auto-approve


def test_scoping_gate_approve():
    ch = CLIChannel()
    hip = HumanInteractionPoint(channels={"cli": ch})
    gate = ScopingConfirmationGate(hip=hip, enabled=True)

    hip_id = gate.present_scope({"task": "Add barcode scanning", "repos": ["receiving"]}, "t1")
    assert hip_id is not None

    # Respond
    pending = ch.get_pending_questions()
    ch.submit_response(pending[0]["message_id"], "approve")

    decision = gate.check_decision(hip_id)
    assert decision == "approve"


def test_scoping_gate_reject():
    ch = CLIChannel()
    hip = HumanInteractionPoint(channels={"cli": ch})
    gate = ScopingConfirmationGate(hip=hip, enabled=True)

    hip_id = gate.present_scope({"task": "test"}, "t1")
    pending = ch.get_pending_questions()
    ch.submit_response(pending[0]["message_id"], "reject")

    assert gate.check_decision(hip_id) == "reject"


def test_format_scope_summary():
    summary = ScopingConfirmationGate.format_scope_summary({
        "task": "Add barcode scanning",
        "domains": ["receiving"],
        "repos": ["receiving-service"],
        "subtasks": ["Parse barcode", "Update handler"],
    })
    assert "barcode scanning" in summary
    assert "receiving-service" in summary
    assert "Parse barcode" in summary
