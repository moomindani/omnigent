"""End-to-end tests for ``sys_call_async`` dispatch of bundled
local Python tools against a real LLM.

Verifies the full pipeline against a live ``omnigent server``
+ real LLM calls:

* The LLM dispatches a slow ``@tool``-decorated function via
  ``sys_call_async`` and gets a JSON handle back as the
  ``sys_call_async`` tool result (not the inline tool result).
* ``background_tool_workflow`` runs the function in a subprocess,
  signals ``async_work_complete``.
* The parent's drain auto-delivers the result as a system message
  (or the LLM proactively drains via ``sys_read_inbox``).
* The LLM sees the result on the next iteration and references the
  literal marker.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_async_tools_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v

**TUI verification** (mandatory per CLAUDE.md before merge):
``omnigent run tests/_fixtures/agents/async-tools-test/``
then ask "dispatch delayed_echo with label='alpha' via
sys_call_async". The auto-delivered result must render as a dim
``⤵ [System: task ...]`` line.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import (
    create_runner_bound_session,
    poll_session_until_terminal,
    send_user_message_to_session,
    upload_agent,
)

_ASYNC_TOOLS_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "_fixtures" / "agents" / "async-tools-test"
)


@pytest.fixture(scope="session")
def async_tools_agent(
    http_client: httpx.Client,
    databricks_workspace_host: str | None,
) -> str:
    """
    Upload the async-tools-test fixture agent.

    Rewrites the YAML's ``executor.model`` to its Databricks-served
    equivalent only when ``--profile`` is set; the raw-OpenAI path
    keeps the original model name.

    :param http_client: HTTP client pointed at the live server.
    :param databricks_workspace_host: Workspace host URL when
        ``--profile`` is set, else ``None``.
    :returns: The agent's name (matches the YAML's ``name`` field).
    """
    return upload_agent(
        http_client,
        _ASYNC_TOOLS_FIXTURE_DIR,
        rewrite_model_for_databricks=databricks_workspace_host is not None,
    )


def _create_response_blocking(
    http_client: httpx.Client,
    *,
    model: str,
    runner_id: str,
    user_text: str,
    timeout_s: float = 180.0,
) -> dict:
    """
    POST a session event, poll until terminal, return the final response-like body.

    :param http_client: HTTP client pointed at the live server.
    :param model: Agent name to invoke.
    :param runner_id: Registered runner id from ``live_runner_id``.
    :param user_text: Plain-text input message for the agent.
    :param timeout_s: Max seconds to wait for the response to
        complete. Default 180 s — async tools sleep 2 s and the
        LLM may take a couple of turns to converge.
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


def _final_text(response_body: dict) -> str:
    """
    Extract the assistant's final text from a response.

    :param response_body: The response JSON returned from
        ``GET /v1/responses/{id}``.
    :returns: Concatenated assistant text. Empty string if no
        assistant message exists.
    """
    parts: list[str] = []
    for item in response_body.get("output", []):
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        for block in item.get("content", []):
            if block.get("type") == "output_text":
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n\n".join(parts)


def _conversation_items(http_client: httpx.Client, conversation_id: str) -> list[dict]:
    """
    Fetch the full ordered list of conversation items.

    :param http_client: HTTP client pointed at the live server.
    :param conversation_id: The conversation ID, e.g.
        ``"conv_abc..."``.
    :returns: Conversation items in store order.
    """
    resp = http_client.get(
        f"/v1/sessions/{conversation_id}/items",
        params={"limit": 100},
    )
    resp.raise_for_status()
    data: list[dict] = resp.json()["data"]
    return data


# ─── Tests ───────────────────────────────────────────────────


def test_async_tool_real_llm_e2e(
    http_client: httpx.Client,
    async_tools_agent: str,
    live_runner_id: str,
) -> None:
    """
    Real LLM dispatches an async tool, sees the auto-delivered
    result, and surfaces the literal marker in its final answer.

    What this catches end-to-end:
    * Schema derivation handed the LLM a usable tool spec.
    * Dispatch produced a handle (no inline result).
    * Background workflow ran in a subprocess.
    * Drain delivered the system message.
    * The LLM read the system message and quoted the marker.
    """
    body = _create_response_blocking(
        http_client,
        model=async_tools_agent,
        runner_id=live_runner_id,
        user_text=(
            "Dispatch delayed_echo with label='alpha' via "
            "sys_call_async. After it completes, tell me the "
            "literal string the tool returned."
        ),
    )
    assert body["status"] == "completed", (
        f"async-tools turn did not complete: status={body.get('status')!r}, "
        f"error={body.get('error')!r}"
    )
    final = _final_text(body)
    # The marker is distinctive enough that the LLM can't have
    # invented it. If absent, either the auto-delivered system
    # message was missing or the LLM ignored it.
    assert "ECHO_FROM_ASYNC[alpha]" in final, (
        f"Expected the tool's literal marker 'ECHO_FROM_ASYNC[alpha]' "
        f"in the final response. Got: {final!r}"
    )

    # Cross-check the conversation store: the auto-delivered
    # [System: task ... completed] message must be persisted.
    conv_id = body["conversation"]["id"]
    items = _conversation_items(http_client, conv_id)
    user_texts = [
        item["content"][0]["text"]
        for item in items
        if item.get("type") == "message"
        and item.get("role") == "user"
        and item.get("content")
        and item["content"][0].get("type") == "input_text"
    ]
    completion_messages = [t for t in user_texts if t.startswith("[System: task ")]
    assert len(completion_messages) == 1, (
        f"Expected exactly one auto-delivered completion message; "
        f"got {len(completion_messages)}. user_texts={user_texts}"
    )
    assert "ECHO_FROM_ASYNC[alpha]" in completion_messages[0], (
        f"The auto-delivered system message must carry the tool's "
        f"actual return value. Got: {completion_messages[0]!r}"
    )


def test_mixed_sync_and_async_tools_e2e(
    http_client: httpx.Client,
    async_tools_agent: str,
    live_runner_id: str,
) -> None:
    """
    The same turn dispatches both an async tool and a sync tool.

    Proves the runtime handles mixed-kind tool batches in
    ``_execute_tools``: the async dispatch returns immediately
    with a handle while the sync tool runs to completion inline,
    then the async result auto-delivers and the LLM references
    both.
    """
    body = _create_response_blocking(
        http_client,
        model=async_tools_agent,
        runner_id=live_runner_id,
        user_text=(
            "Run TWO tools in this turn: count_chars on the text "
            "'hello' (which is 5 characters) — call it directly "
            "for an inline result. AND dispatch delayed_echo with "
            "label='beta' via sys_call_async for background "
            "execution. After both finish, tell me both results "
            "verbatim — the count_chars number and the "
            "delayed_echo literal string."
        ),
    )
    assert body["status"] == "completed", (
        f"mixed-tools turn did not complete: "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )
    final = _final_text(body)
    # Sync tool result — straightforward integer assert.
    assert "5" in final, (
        f"Expected the count_chars result '5' in the final response. Got: {final!r}"
    )
    # Async tool result — distinctive marker.
    assert "ECHO_FROM_ASYNC[beta]" in final, (
        f"Expected the delayed_echo marker 'ECHO_FROM_ASYNC[beta]' "
        f"in the final response. Got: {final!r}"
    )


def test_async_tool_failure_surfaces_e2e(
    http_client: httpx.Client,
    async_tools_agent: str,
    live_runner_id: str,
) -> None:
    """
    Real LLM invokes the failing async tool, sees the failure
    system message, and acknowledges the error in its response.

    Without G86 the parent's drain would never wake — this test
    would time out at the polling loop instead of asserting on
    the LLM's text.
    """
    body = _create_response_blocking(
        http_client,
        model=async_tools_agent,
        runner_id=live_runner_id,
        user_text=(
            "Dispatch boom_async via sys_call_async. Then tell me "
            "what happened — include the literal error marker "
            "string from the system message in your reply so I "
            "can verify it."
        ),
    )
    # The agent's response itself must complete (only the tool
    # task fails). If status="failed" here, the failure was
    # incorrectly propagated as an agent-level error.
    assert body["status"] == "completed", (
        f"async failure must not fail the agent turn: "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )
    final = _final_text(body)
    # The exception message marker proves the failure traceback
    # survived format_failure_payload + truncate_traceback +
    # drain → system message → next LLM prompt.
    assert "ASYNC_TOOL_BOOM_MARKER" in final, (
        f"Expected the failure marker 'ASYNC_TOOL_BOOM_MARKER' in "
        f"the final response — failure path likely dropped the "
        f"exception detail somewhere between the tool body and "
        f"the LLM's view. Got: {final!r}"
    )
