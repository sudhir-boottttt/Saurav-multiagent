"""Property-based and unit tests for the Message Bus."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.infrastructure.message_bus import (
    MessageBus,
    RequestTimeoutError,
    TopicValidationError,
)


# --- Property Tests ---
# Feature: multi-agent-platform, Property 5: Topic Format Validation


@given(
    domain=st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
    repo=st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
    event=st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
)
@settings(max_examples=100)
def test_valid_topic_accepted(domain, repo, event):
    """Valid topics in {domain}.{repo}.{event} format should be accepted."""
    topic = f"{domain}.{repo}.{event}"
    assert MessageBus.validate_topic(topic) is True


@given(bad_topic=st.text(min_size=0, max_size=20).filter(lambda t: t.count(".") != 2))
@settings(max_examples=100)
def test_invalid_topic_rejected(bad_topic):
    """Topics not matching the 3-part format should be rejected."""
    assert MessageBus.validate_topic(bad_topic) is False


# Feature: multi-agent-platform, Property 6: Message Persistence Before Delivery


@given(
    domain=st.sampled_from(["inbound", "outbound"]),
    repo=st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
    event=st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
)
@settings(max_examples=100)
def test_message_persisted_before_delivery(domain, repo, event):
    """Published messages must be persisted and retrievable by ID."""
    bus = MessageBus()
    topic = f"{domain}.{repo}.{event}"
    msg_id = bus.publish(topic, {"data": "test"}, "sender-1")
    msg = bus.get_message(msg_id)
    assert msg is not None
    assert msg.topic == topic
    assert msg.sender_id == "sender-1"
    assert msg.payload == {"data": "test"}
    assert msg.timestamp is not None


# Feature: multi-agent-platform, Property 7: Message Delivery Failure Handling


def test_delivery_failure_moves_to_dlq():
    """After 3 failed retries, message should be in dead-letter queue."""
    bus = MessageBus()
    notified = []
    bus.set_pm_agent_id("pm-1")
    bus.set_notify_callback(lambda agent_id, msg: notified.append(agent_id))

    def failing_handler(msg):
        raise RuntimeError("subscriber down")

    bus.subscribe("inbound.svc.event", failing_handler)
    bus.publish("inbound.svc.event", {"data": "test"}, "sender-1")

    dlq = bus.get_dead_letter_queue()
    assert len(dlq) == 1
    assert dlq[0].retry_count == 3  # MAX_RETRIES
    assert "pm-1" in notified


def test_successful_delivery_no_dlq():
    """Successful delivery should not put message in DLQ."""
    bus = MessageBus()
    received = []
    bus.subscribe("inbound.svc.event", lambda msg: received.append(msg))
    bus.publish("inbound.svc.event", {"data": "hello"}, "sender-1")
    assert len(received) == 1
    assert bus.get_dead_letter_queue() == []


# Feature: multi-agent-platform, Property 8: Request-Reply Timeout


def test_request_reply_timeout():
    """Request-reply should raise timeout when no reply comes."""
    bus = MessageBus()
    with pytest.raises(RequestTimeoutError):
        bus.request_reply("inbound.svc.query", {"q": "test"}, "sender-1", timeout_ms=50)


def test_request_reply_success():
    """Request-reply should return reply payload when reply is sent."""
    bus = MessageBus()

    def echo_handler(msg):
        reply_topic = msg.payload.get("_reply_topic")
        if reply_topic:
            bus.publish(reply_topic, {"answer": 42}, "responder-1", _skip_validation=True)

    bus.subscribe("inbound.svc.query", echo_handler)
    result = bus.request_reply("inbound.svc.query", {"q": "test"}, "sender-1", timeout_ms=1000)
    assert result == {"answer": 42}


# --- Unit Tests ---


def test_publish_invalid_topic_raises():
    bus = MessageBus()
    with pytest.raises(TopicValidationError):
        bus.publish("bad-topic", {}, "sender-1")


def test_subscribe_and_receive():
    bus = MessageBus()
    received = []
    bus.subscribe("outbound.picking.bug_detected", lambda msg: received.append(msg.payload))
    bus.publish("outbound.picking.bug_detected", {"bug": "null_ref"}, "dev-1")
    assert len(received) == 1
    assert received[0]["bug"] == "null_ref"


def test_wildcard_subscription():
    bus = MessageBus()
    received = []
    bus.subscribe("inbound.*.bug_detected", lambda msg: received.append(msg.topic))
    bus.publish("inbound.receiving.bug_detected", {}, "dev-1")
    bus.publish("inbound.putaway.bug_detected", {}, "dev-2")
    assert len(received) == 2


def test_unsubscribe():
    bus = MessageBus()
    received = []
    sub = bus.subscribe("inbound.svc.event", lambda msg: received.append(1))
    bus.publish("inbound.svc.event", {}, "s1")
    assert len(received) == 1
    bus.unsubscribe(sub)
    bus.publish("inbound.svc.event", {}, "s1")
    assert len(received) == 1  # no new messages after unsubscribe


def test_multiple_subscribers():
    bus = MessageBus()
    r1, r2 = [], []
    bus.subscribe("inbound.svc.event", lambda msg: r1.append(1))
    bus.subscribe("inbound.svc.event", lambda msg: r2.append(1))
    bus.publish("inbound.svc.event", {}, "s1")
    assert len(r1) == 1
    assert len(r2) == 1
