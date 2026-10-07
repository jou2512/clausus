"""The chat client retries a server error once, never a request error."""

import httpx
import pytest

from clausus.errors import ClaususError
from clausus.llm import ChatClient

OK = {"choices": [{"message": {"content": "fertig"}}]}


def client(*responses):
    calls = []
    queue = list(responses)

    def handler(req):
        calls.append(req)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    c = ChatClient("https://a.test/v1", "k", "m", "apertus", http=httpx.Client(transport=httpx.MockTransport(handler)))
    c.retry_pause_s = 0
    return c, calls


def test_a_server_error_is_retried_once():
    """07.10.2026: one HTTP 500 from the hackathon endpoint ended a whole benchmark run; with
    long texts split into sections a message makes more calls, so one hiccup costs more."""
    c, calls = client(httpx.Response(500), httpx.Response(200, json=OK))
    assert c.chat([{"role": "user", "content": "x"}]) == "fertig"
    assert len(calls) == 2


def test_a_lost_connection_is_retried_once():
    c, calls = client(httpx.ConnectError("down"), httpx.Response(200, json=OK))
    assert c.chat([{"role": "user", "content": "x"}]) == "fertig"
    assert len(calls) == 2


def test_two_server_errors_fail_closed():
    c, calls = client(httpx.Response(502), httpx.Response(500))
    with pytest.raises(ClaususError) as e:
        c.chat([{"role": "user", "content": "x"}])
    assert e.value.code == "bad_response" and len(calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403])
def test_request_errors_are_not_retried(status):
    c, calls = client(httpx.Response(status))
    with pytest.raises(ClaususError):
        c.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 1


def test_a_rate_limit_is_retried_once_after_a_longer_pause():
    """07.10.2026: four sections checked in parallel ran into HTTP 429 on the hackathon endpoint."""
    c, calls = client(httpx.Response(429), httpx.Response(200, json=OK))
    c.rate_limit_pause_s = 0
    assert c.chat([{"role": "user", "content": "x"}]) == "fertig"
    assert len(calls) == 2


def test_two_rate_limits_are_reported_as_such():
    c, calls = client(httpx.Response(429), httpx.Response(429))
    c.rate_limit_pause_s = 0
    with pytest.raises(ClaususError) as e:
        c.chat([{"role": "user", "content": "x"}])
    assert e.value.code == "rate_limited" and len(calls) == 2
