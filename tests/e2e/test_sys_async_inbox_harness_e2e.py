"""
End-to-end tests for the harness-path async-tool dispatch chain.

The companion file ``test_sys_async_inbox_e2e.py`` covers the
in-process drain via :meth:`SysReadInboxTool.dispatch_async`.
This file pins the OTHER routing path: when the LLM lives in an
external harness subprocess (claude-sdk here), AP's
the harness HTTP client checks
``Tool.is_async()`` on each action_required tool. Tools that
return ``True`` (today: ``sys_call_async`` and ``sys_read_inbox``;
``WebFetchTool`` / ``WebSearchTool`` also qualify but require
provider keys to exercise) are dispatched INLINE in the parent
agent workflow's DBOS context via :meth:`_dispatch_async_tool_inline`
— skipping the ``tool_dispatch_workflow`` child spawn so
``Tool.dispatch_async`` runs against the right mailbox.

Without the is_async branch, the LLM's ``sys_read_inbox`` /
``sys_call_async`` calls would either:

* Spawn ``tool_dispatch_workflow`` and call ``Tool.invoke`` —
  which is a routing-error stub for these tools (the LLM-mode
  path uses ``dispatch_async``), OR
* If invoke wasn't a stub, the child workflow's DBOS context
  couldn't read the parent's ``async_work_complete`` mailbox
  anyway — the drain would silently return empty.

The two tests in this file:

1. ``test_sys_call_async_harness_path_returns_handle_e2e`` —
   focused reproduction of the original sys_call_async harness
   regression. Asks the LLM to dispatch exactly one
   ``sys_call_async`` and asserts on its function_call_output
   directly: it MUST be a parseable handle JSON, NOT
   ``{"error": "internal_routing_error", ...}``. Would have
   failed under the prior commit's name-based short-circuit
   (which only matched ``sys_read_inbox`` — sys_call_async
   fell through to ``Tool.invoke``'s routing-error stub).
2. ``test_sys_read_inbox_harness_path_returns_drained_marker_e2e`` —
   end-to-end chain (dispatch → drain → marker quote) that
   exercises BOTH async tools together. Catches both regression
   modes (routing-error AND empty-drain) via the marker assertion.

Excluded from default ``pytest`` runs via ``--ignore=tests/e2e``.
Invoke with::

    pytest tests/e2e/test_sys_async_inbox_harness_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import (
    create_runner_bound_session,
    poll_session_until_terminal,
    send_user_message_to_session,
)
from tests.e2e.helpers import final_assistant_text, get_output_items

_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "_fixtures" / "agents" / "sys-async-inbox-harness-test"
)


@pytest.fixture(scope="session")
def sys_async_inbox_harness_agent(http_client: httpx.Client) -> str:
    """
    Upload the harness-flavored variant of the sys-async-inbox fixture.

    The bundled YAML pins ``executor.type: claude_sdk`` so the LLM
    runs in a real claude-sdk harness subprocess and ``sys_read_inbox``
    flows through the harness HTTP client
    instead of :meth:`SysReadInboxTool.dispatch_async`.

    :param http_client: HTTP client pointed at the live server.
    :returns: The agent's name (matches its config.yaml ``name``).
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(str(_FIXTURE_DIR), arcname=".")
        bundle_path = tmp.name
    try:
        with open(bundle_path, "rb") as f:
            resp = http_client.post(
                "/v1/sessions",
                data={"metadata": json.dumps({})},
                files={
                    "bundle": (
                        "agent.tar.gz",
                        f,
                        "application/gzip",
                    ),
                },
            )
        if resp.status_code == 409:
            # Already registered in the same session.
            return _FIXTURE_DIR.name
        resp.raise_for_status()
        session_id = resp.json()["session_id"]
        agent_resp = http_client.get(f"/v1/sessions/{session_id}/agent")
        agent_resp.raise_for_status()
        name: str = agent_resp.json()["name"]
        return name
    finally:
        Path(bundle_path).unlink(missing_ok=True)


def _create_response_blocking(
    http_client: httpx.Client,
    *,
    model: str,
    runner_id: str,
    user_text: str,
    timeout_s: float = 240.0,
) -> dict[str, object]:
    """
    POST a session event, poll until terminal, return the final response-like body.

    Same shape as :func:`tests.e2e.test_sys_async_inbox_e2e._create_response_blocking`.
    Duplicated here rather than promoted to a shared helper to
    keep the e2e files individually inspectable; if a third
    consumer lands, both should move to ``tests/e2e/helpers.py``.

    :param http_client: HTTP client pointed at the live server.
    :param model: Agent name to invoke.
    :param runner_id: Registered runner id from ``live_runner_id``.
    :param user_text: Plain-text input message for the agent.
    :param timeout_s: Max seconds to wait for the response to
        complete. The harness path can take longer than the
        in-process LLM path because the inner SDK subprocess
        adds ~5-10 s of cold-start overhead, so 240 s leaves
        comfortable margin.
    :returns: The final response JSON.
    """
    session_id = create_runner_bound_session(
        http_client,
        agent_name=model,
        runner_id=runner_id,
    )
    response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=user_text,
    )
    return poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=response_id,
        timeout=timeout_s,
    )


def _function_call_outputs_for(response_body: dict[str, object], call_ids: set[str]) -> list[str]:
    """
    Return the ``output`` strings of ``function_call_output`` items
    whose ``call_id`` is in ``call_ids``.

    Mirrors the helper of the same name in
    ``test_sys_async_inbox_e2e.py``; same rationale for keeping
    it local rather than promoting to ``helpers.py``.

    :param response_body: Response body from
        ``GET /v1/responses/{id}``.
    :param call_ids: Call IDs to match against function_call_output
        items.
    :returns: Matching string outputs in original order.
    """
    output_items = response_body.get("output", [])
    assert isinstance(output_items, list), (
        f"response.output must be a list; got {type(output_items)!r}"
    )
    matched: list[str] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if (
            item.get("type") == "function_call_output"
            and item.get("call_id") in call_ids
            and isinstance(item.get("output"), str)
        ):
            matched.append(item["output"])
    return matched


def test_sys_call_async_harness_path_returns_handle_e2e(
    http_client: httpx.Client,
    sys_async_inbox_harness_agent: str,
    live_runner_id: str,
) -> None:
    """
    The LLM's ``sys_call_async`` function_call_output MUST be a valid
    handle JSON when running on the harness path — not the
    ``{"error": "internal_routing_error", ...}`` envelope from
    ``SysCallAsyncTool.invoke``.

    Reproduces the original sys_call_async-on-harness regression:
    under the prior commit's name-based short-circuit (which only
    matched ``name == SysReadInboxTool.name()``), every other
    is_async tool — sys_call_async included — fell through to
    ``_spawn_dispatch_workflow_and_await``. That spawned
    ``tool_dispatch_workflow``, which calls ``Tool.invoke``.
    SysCallAsyncTool.invoke is a routing-error stub
    (``{"error": "internal_routing_error", ...}``), so the LLM saw
    a tool failure where it expected an async-handle JSON.

    The fix (generic ``tool.is_async()`` check ahead of the spawn)
    routes sys_call_async through ``_dispatch_async_tool_inline``,
    which calls ``SysCallAsyncTool.dispatch_async`` from the parent's
    DBOS context — that spawns ``background_tool_workflow`` and
    returns an ``_AsyncToolHandle``. The handle's
    ``to_handle_json()`` form (a dict with ``task_id``, ``tool_name``,
    ``status``, ``message`` keys) is what the LLM must see in the
    function_call_output.

    Why this test matters even with the chain-style test below: that
    test asserts on the marker in ``sys_read_inbox``'s output, which
    means a sys_call_async failure manifests indirectly (empty drain).
    This test fails LOUDLY at the source by checking sys_call_async's
    own output — so the diagnostic points at the right tool.

    :param http_client: HTTP client pointed at the live server.
    :param sys_async_inbox_harness_agent: Agent name from the
        session-scoped fixture.
    """
    body = _create_response_blocking(
        http_client,
        model=sys_async_inbox_harness_agent,
        runner_id=live_runner_id,
        user_text=(
            "Call sys_call_async EXACTLY ONCE with target='tag_label' "
            'and args=\'{"label": "epsilon"}\'. Then stop. Do not '
            "call sys_read_inbox. Do not call any other tool. Reply "
            "with just 'done' after the dispatch."
        ),
    )
    assert body["status"] == "completed", (
        f"harness sys_call_async turn did not complete: "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )

    call_async_calls = get_output_items(body, "function_call", name="sys_call_async")
    # 1+ = LLM did the dispatch (the test premise). If 0, the LLM
    # ignored the prompt and the regression-under-test wasn't
    # exercised — fail loudly so a flake doesn't masquerade as a
    # passing test.
    assert len(call_async_calls) >= 1, (
        f"LLM did not call sys_call_async; the regression-under-test "
        f"(sys_call_async returning routing-error on harness path) "
        f"wasn't exercised. function_calls in output: "
        f"{[i.get('name') for i in get_output_items(body, 'function_call')]!r}"
    )

    call_ids = {item["call_id"] for item in call_async_calls}
    outputs = _function_call_outputs_for(body, call_ids)
    assert len(outputs) == len(call_async_calls), (
        f"Expected one function_call_output per sys_call_async call, "
        f"got {len(outputs)} outputs for {len(call_async_calls)} calls. "
        f"A missing pairing would mean the inline dispatch's PATCH "
        f"didn't reach the response (PATCH failure or pairing-buffer "
        f"stamp regression)."
    )

    # Load-bearing: parse each sys_call_async output as JSON and
    # check that it's a handle, NOT the routing-error envelope.
    # This is the assertion that fails under the prior name-based
    # short-circuit and passes under the generic is_async routing.
    for raw in outputs:
        # routing-error sentinel detection — the explicit shape the
        # bug produced. Match before parsing so the diagnostic is
        # specific even if the LLM-formatted output isn't strict JSON.
        assert "internal_routing_error" not in raw, (
            f"sys_call_async output carries the routing-error sentinel: "
            f"{raw!r}. The harness path fell through to "
            f"SysCallAsyncTool.invoke instead of taking the is_async "
            f"inline branch in _dispatch_action_required. This is the "
            f"exact regression mode the generic is_async routing fix "
            f"closed."
        )
        # Positive shape check: a real ``_AsyncToolHandle.to_handle_json()``
        # produces a dict with task_id + tool_name + status + message.
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise AssertionError(
                f"sys_call_async output is not valid JSON: {raw!r}; "
                f"expected a handle dict from _AsyncToolHandle.to_handle_json(). "
                f"Parse error: {exc!r}"
            ) from exc
        assert isinstance(parsed, dict), (
            f"sys_call_async output parsed to {type(parsed).__name__}, expected dict. Raw: {raw!r}"
        )
        # Required keys per ``_AsyncToolHandle.to_handle_json``.
        for key in ("task_id", "tool_name", "status"):
            assert key in parsed, (
                f"sys_call_async handle missing {key!r}: {parsed!r}. "
                f"to_handle_json() must emit task_id+tool_name+status; "
                f"a missing key indicates a contract drift in "
                f"_AsyncToolHandle or the inline dispatch wrapped the "
                f"return in an unexpected shape."
            )
        # The handle is freshly created — status MUST be in_progress
        # (terminal status arrives via async_work_complete, not at
        # dispatch time).
        assert parsed["status"] == "in_progress", (
            f"Fresh sys_call_async handle should report status="
            f"'in_progress', got {parsed['status']!r}. Real result "
            f"auto-delivers separately; the handle is just the "
            f"dispatch acknowledgement."
        )
        # tool_name carries the dispatched target ("tag_label" here),
        # NOT "sys_call_async". The LLM uses this to correlate the
        # handle with its own tool_calls list.
        assert parsed["tool_name"] == "tag_label", (
            f"sys_call_async handle tool_name should be the dispatched "
            f"target 'tag_label', got {parsed['tool_name']!r}. The "
            f"agent_name parameter to dispatch_async sources this "
            f"field; a wrong value indicates the inline path threaded "
            f"the wrong name through."
        )


def test_sys_read_inbox_harness_path_returns_drained_marker_e2e(
    http_client: httpx.Client,
    sys_async_inbox_harness_agent: str,
    live_runner_id: str,
) -> None:
    """
    The LLM dispatches ``tag_label`` via ``sys_call_async``, then
    drains via ``sys_read_inbox`` — and the drain output carries
    the dispatched tool's literal marker.

    Pins the harness HTTP client
    end-to-end across both async tools the test exercises
    (``sys_call_async`` for the dispatch, ``sys_read_inbox`` for
    the drain):

    * The LLM emits a ``sys_call_async`` function_call.
    * AP's ``_dispatch_action_required`` resolves the tool, sees
      ``is_async() == True``, and routes to
      ``_dispatch_async_tool_inline`` — which calls
      :meth:`SysCallAsyncTool.dispatch_async` from the parent's
      DBOS context. That spawns ``background_tool_workflow`` and
      returns an ``_AsyncToolHandle``.
    * The LLM later emits a ``sys_read_inbox`` function_call.
      Same routing: is_async branch fires,
      :meth:`SysReadInboxTool.dispatch_async` reads the parent's
      ``async_work_complete`` mailbox via
      :func:`_drain_async_completions` (which works because we
      ARE in the parent's DBOS workflow context here).
    * The drained payload — ``tag_label``'s ``"SYS_ASYNC_TAG[..]"``
      marker — is rendered via :func:`_format_async_completion_text`
      and PATCHed back to the harness as the tool's output.

    A regression that removed the is_async branch (or scoped it
    incorrectly so it skipped one of these tools) would either:

    * Spawn ``tool_dispatch_workflow`` and call ``Tool.invoke`` —
      which for ``sys_call_async`` and ``sys_read_inbox`` is a
      routing-error stub, surfacing ``"internal_routing_error"``
      as the tool output, OR
    * Even if invoke weren't a stub, the child workflow's DBOS
      context can't read the parent's mailbox — the drain would
      silently return empty.

    Either failure mode means the marker doesn't surface in the
    ``sys_read_inbox`` function_call_output. We assert on the
    drain's literal output rather than the LLM's final text so
    the assertion is robust against the LLM choosing whether to
    quote the marker or paraphrase it.

    :param http_client: HTTP client pointed at the live server.
    :param sys_async_inbox_harness_agent: Agent name from the
        session-scoped fixture.
    """
    body = _create_response_blocking(
        http_client,
        model=sys_async_inbox_harness_agent,
        runner_id=live_runner_id,
        user_text=(
            "Step 1: dispatch the tag_label tool with "
            "label='omega' using sys_call_async. Step 2: wait a "
            "moment for it to complete. Step 3: call "
            "sys_read_inbox. Step 4: tell me the literal string "
            "the tool returned, character-for-character."
        ),
    )
    assert body["status"] == "completed", (
        f"harness sys_read_inbox turn did not complete: "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )

    # The LLM must have called both tools — otherwise the path
    # under test wasn't exercised.
    output_items = body.get("output", [])
    assert isinstance(output_items, list), (
        f"response.output must be a list; got {type(output_items)!r}"
    )
    fn_call_names = [
        i.get("name")
        for i in output_items
        if isinstance(i, dict) and i.get("type") == "function_call"
    ]
    call_async_calls = get_output_items(body, "function_call", name="sys_call_async")
    assert len(call_async_calls) >= 1, (
        f"LLM did not call sys_call_async; the test premise — "
        f"async dispatch followed by inbox drain — wasn't exercised. "
        f"function_calls in output: {fn_call_names!r}"
    )
    read_inbox_calls = get_output_items(body, "function_call", name="sys_read_inbox")
    assert len(read_inbox_calls) >= 1, (
        f"LLM did not call sys_read_inbox; this test specifically "
        f"verifies the inline drain on the harness path, so the "
        f"call MUST happen. function_calls in output: {fn_call_names!r}"
    )

    # Load-bearing assertion: the marker MUST appear in the
    # sys_read_inbox tool's output. That's the path the
    # is_async branch produces — drained payload, rendered via
    # _format_async_completion_text, surfaced to the LLM via
    # the function_call_output.
    read_inbox_call_ids = {item["call_id"] for item in read_inbox_calls}
    read_inbox_outputs = _function_call_outputs_for(body, read_inbox_call_ids)
    marker = "SYS_ASYNC_TAG[omega]"
    inbox_outputs_with_marker = [o for o in read_inbox_outputs if marker in o]
    assert inbox_outputs_with_marker, (
        f"Marker {marker!r} did not appear in any sys_read_inbox "
        f"function_call_output. read_inbox_outputs={read_inbox_outputs!r}. "
        f"If every output is the empty-inbox sentinel, either "
        f"sys_call_async didn't actually run (so nothing was queued) "
        f"or the inbox drain ran in the wrong DBOS context (child "
        f"workflow instead of parent — meaning the is_async branch "
        f"didn't fire). If every output is the routing-error sentinel "
        f"(``internal_routing_error``), the is_async branch in "
        f"_dispatch_action_required is missing and the harness path "
        f"fell through to SysReadInboxTool.invoke / "
        f"SysCallAsyncTool.invoke."
    )

    # Belt-and-suspenders: the LLM should have quoted the marker
    # back. Permissive — the LLM might paraphrase, so this is a
    # secondary signal. The load-bearing assertion is above.
    final = final_assistant_text(body)
    if marker not in final:
        pytest.skip(
            f"sys_read_inbox carried the marker (proven above), "
            f"but the LLM didn't quote it verbatim in the final "
            f"reply: {final!r}. This is an LLM-paraphrase "
            f"variation, not a regression in the dispatch path."
        )
