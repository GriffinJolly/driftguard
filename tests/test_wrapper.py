import httpx

from driftguard.ingest.wrapper import DriftGuardClient, make_wrapped_client


class FakeStore:
    def __init__(self):
        self.saved = []

    def write_log(self, entry):
        self.saved.append(entry)


def test_make_wrapped_client_creates_httpx_client():
    store = FakeStore()
    client = make_wrapped_client(
        store=store,
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
    )

    assert isinstance(client, httpx.Client)
    assert str(client.base_url) == "https://api.groq.com/openai/v1/"
    client.close()


def test_driftguard_client_wraps_transport_and_exposes_post():
    store = FakeStore()
    client = DriftGuardClient(
        provider="groq",
        store=store,
        base_url="https://api.groq.com/openai/v1",
    )

    assert isinstance(client.client, httpx.Client)
    assert hasattr(client, "post")
    client.close()
