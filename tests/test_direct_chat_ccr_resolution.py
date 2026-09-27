"""The direct chat-completions path must answer the retrieval tool it injects.

`headroom_retrieve` is injected into a non-streaming chat request by default
(`ccr_inject_tool=True`), but `handlers/openai.py`'s direct HTTP branch had no
CCR tool-call handling at all — the code said so itself. Every other path
resolves the call: `gateway_turn.py:1003`, `handlers/gemini.py:837`,
`handlers/anthropic.py:4286`, `handlers/openai.py` for the custom backend and
for Responses. Only the direct chat branch did not.

So the proxy advertised a tool it would not answer. A model that called it sent
the client a `tool_calls` entry for a tool the client never declared and cannot
implement, which stalls any agent loop driven off `finish_reason`.

The accounting pair is the part worth keeping honest. A resolved turn makes two
billed upstream calls, and the continuation resends the whole conversation, so
counting only the last one hides roughly half the turn. The split matters too:
the forwarded body keeps the provider's own usage untouched (byte-faithful
forwarding), while Headroom's cost tracking folds in the extra call — the same
division the turn-hook re-drive path already uses via `TurnHookUsage`.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache.compression_store import (  # noqa: E402
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.tool_injection import CCR_TOOL_NAME  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

ORIGINAL = "row 1: the original uncompressed rows\nrow 2: more of them"


@pytest.fixture(autouse=True)
def reset_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _stored_hash() -> str:
    return get_compression_store().store(
        original=ORIGINAL,
        compressed="[2 items compressed to 0]",
        original_item_count=2,
        compressed_item_count=0,
    )


def _config() -> ProxyConfig:
    # No backend configured -> the direct OpenAI HTTP path.
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        ccr_handle_responses=True,
        ccr_inject_tool=True,
    )


def _retrieve_call_response(hash_key: str) -> dict:
    """An upstream reply in which the model asks to retrieve."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": CCR_TOOL_NAME,
                                "arguments": json.dumps({"hash": hash_key}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def _final_response() -> dict:
    """The continuation reply, after the tool result was supplied."""
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "there are 2 rows"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 400, "completion_tokens": 10, "total_tokens": 410},
    }


def _run(upstream_sequence: list[dict]) -> tuple[httpx.Response, list[dict], list]:
    """Post one chat turn, serving `upstream_sequence` to successive calls.

    Returns the response, the request bodies that went upstream, and the
    RequestOutcome objects the proxy recorded.
    """
    sent: list[dict] = []
    outcomes: list = []
    remaining = list(upstream_sequence)

    async def fake_retry(method, url, headers, req_body, *args, **kwargs):
        sent.append(req_body)
        payload = remaining.pop(0) if remaining else upstream_sequence[-1]
        return httpx.Response(200, json=payload, headers={"content-type": "application/json"})

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = fake_retry

        # The continuation on this path may go out through the shared client
        # rather than _retry_request; capture both so the test does not depend
        # on which transport the implementation picks.
        class _FakeHTTPClient:
            async def post(self, url, content=None, headers=None, **kwargs):  # noqa: ANN001
                import json as _json

                sent.append(_json.loads(content) if content else {})
                payload = remaining.pop(0) if remaining else upstream_sequence[-1]
                return httpx.Response(
                    200, json=payload, headers={"content-type": "application/json"}
                )

        proxy.http_client = _FakeHTTPClient()

        _real_record = proxy._record_request_outcome

        async def _capture(outcome, *args, **kwargs):  # noqa: ANN001
            outcomes.append(outcome)
            return await _real_record(outcome, *args, **kwargs)

        proxy._record_request_outcome = _capture

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "how many rows?"}],
                "stream": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "Read", "description": "read a file"},
                    }
                ],
            },
            headers={"Authorization": "Bearer test-key", "x-api-key": "test-key"},
        )
        return resp, sent, outcomes


def test_a_retrieval_call_is_resolved_before_the_client_sees_it():
    """The client must never receive a headroom_retrieve tool call."""
    hash_key = _stored_hash()
    resp, sent, _outcomes = _run([_retrieve_call_response(hash_key), _final_response()])

    assert resp.status_code == 200, resp.text
    body = resp.json()
    message = body["choices"][0]["message"]

    tool_calls = message.get("tool_calls") or []
    names = [tc.get("function", {}).get("name") for tc in tool_calls]
    assert CCR_TOOL_NAME not in names, (
        "the proxy injected headroom_retrieve and then handed the model's call "
        "straight to the client, which cannot implement it"
    )
    assert message.get("content") == "there are 2 rows"
    assert len(sent) == 2, f"expected an original call plus a continuation, got {len(sent)}"


def test_the_continuation_carries_the_retrieved_content():
    """The retrieved original has to reach the model, not just be looked up."""
    hash_key = _stored_hash()
    _resp, sent, _outcomes = _run([_retrieve_call_response(hash_key), _final_response()])

    assert len(sent) == 2
    continuation = sent[1]
    serialized = str(continuation)
    assert ORIGINAL.split("\n")[0] in serialized, (
        "the continuation request did not carry the retrieved original content"
    )


def test_the_forwarded_body_keeps_the_provider_usage_untouched():
    """Byte-faithful forwarding: the client sees the provider's own final usage.

    Rewriting `usage` in the forwarded body would misreport what the upstream
    call actually returned. Extra calls belong in Headroom's own accounting,
    which the next test covers — this is the same split the turn-hook re-drive
    path already uses.
    """
    hash_key = _stored_hash()
    resp, _sent, _outcomes = _run([_retrieve_call_response(hash_key), _final_response()])
    assert (resp.json().get("usage") or {}).get("prompt_tokens") == 400


def test_headroom_accounting_counts_both_upstream_calls():
    """A retrieval turn makes two billed calls; cost tracking must see both.

    `handle_response` replaces the response rather than merging usage, so the
    pre-continuation call's tokens are invisible unless they are folded in
    explicitly. The continuation resends the whole conversation, so counting
    only the last call hides roughly half the turn — the same mistake the
    re-drive block warns about ("counting only the last one lets the feature
    hide its own overhead behind the saving it is claiming").
    """
    hash_key = _stored_hash()
    _resp, sent, outcomes = _run([_retrieve_call_response(hash_key), _final_response()])
    assert len(sent) == 2
    assert outcomes, "no RequestOutcome was recorded"

    recorded = outcomes[-1].provider_input_tokens
    assert recorded >= 100 + 400, (
        f"cost tracking saw {recorded} input tokens for a turn that made two "
        "upstream calls costing 100 + 400; the first call was dropped"
    )
