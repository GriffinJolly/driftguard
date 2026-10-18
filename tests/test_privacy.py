"""What the capture layer keeps of prompt and reply text (driftguard/ingest/privacy.py)."""

import httpx
import pytest
from sqlmodel import Session, select

from driftguard.ingest.privacy import apply_text_policy, redact_text
from driftguard.ingest.transport_hook import DriftCaptureTransport
from driftguard.storage.store import DriftStore, LogRow


def test_redaction_replaces_common_identifiers():
    text = ("Mail jane.doe@example.com or call +91 98765 43210, card 4111 1111 1111 1111, "
            "server 10.0.0.12, key gsk_abcdefghijklmnopqrstuv1234")
    out = redact_text(text)
    for secret in ("jane.doe@example.com", "98765", "4111", "10.0.0.12", "gsk_abcdefghijklmnopqrstuv1234"):
        assert secret not in out
    for placeholder in ("[EMAIL]", "[PHONE]", "[CARD]", "[IP]", "[SECRET]"):
        assert placeholder in out
    assert redact_text("What is 17 * 23? Answer B.") == "What is 17 * 23? Answer B."


def test_policies():
    assert apply_text_policy("hello", "full") == "hello"
    h = apply_text_policy("hello", "hash")
    assert h.startswith("sha256:") and h == apply_text_policy("hello", "hash") and "hello" not in h
    assert apply_text_policy("hello", "none") == ""
    assert apply_text_policy(None, "hash") is None
    with pytest.raises(ValueError):
        apply_text_policy("x", "encrypt")


def _reply(request):
    return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "Reply to bob@example.com"}}]})


@pytest.mark.parametrize("policy", ["full", "redact", "hash", "none"])
def test_transport_applies_the_policy(tmp_path, policy):
    store = DriftStore(db_path=str(tmp_path / "p.db"))
    client = httpx.Client(transport=DriftCaptureTransport(store=store, wrapped=httpx.MockTransport(_reply),
                                                          text_policy=policy))
    r = client.post("https://api.groq.com/openai/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Email alice@example.com"}]})
    assert r.json()["choices"][0]["message"]["content"] == "Reply to bob@example.com"   # the caller sees the real reply
    with Session(store.engine) as s:
        row = s.exec(select(LogRow)).one()
    stored = row.prompt + " " + (row.response_text or "")
    if policy == "full":
        assert "alice@example.com" in row.prompt and row.response_text == "Reply to bob@example.com"
    elif policy == "redact":
        assert "@example.com" not in stored and row.prompt.count("[EMAIL]") == 1 and row.response_text == "Reply to [EMAIL]"
    elif policy == "hash":
        assert row.prompt.startswith("sha256:") and row.response_text.startswith("sha256:") and "@" not in stored
    else:
        assert row.prompt == "" and row.response_text == ""


def test_unknown_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        DriftCaptureTransport(store=DriftStore(db_path=str(tmp_path / "x.db")), text_policy="maybe")
