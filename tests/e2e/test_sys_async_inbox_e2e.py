"""
End-to-end tests for ``sys_call_async`` + ``sys_read_inbox`` against
a real LLM.

Verifies the full Phase 2 step-11a pipeline:

* Top-level ``async: true`` registers the new LLM-callable builtins.
* The LLM receives the tool schemas and decides to dispatch
  ``sys_call_async`` with a target tool name + JSON args.
* ``SysCallAsyncTool.dispatch_async`` looks up the target via the
  runtime tool manager and calls
  ``_dispatch_local_python_tool_async``, which returns an
  ``_AsyncToolHandle`` (the existing handle path — exercises the
  ``isinstance(dispatched, _AsyncToolHandle)`` branch of
  ``_execute_tools``).
* The LLM follows up with ``sys_read_inbox``;
  ``SysReadInboxTool.dispatch_async`` calls
  ``_drain_async_completions`` and returns a ``str`` (the new
  inline-result path of ``_execute_tools`` — exercises the ``else``
  branch added in 11a.ii).
* The drained payload contains the ``tag_label`` tool's literal
  return marker; the LLM quotes it back to the user.

What this catches that the unit tests don't:

* The actual workflow.py dispatch routing of two different
  return shapes (``_AsyncToolHandle`` vs ``str``) from
  ``Tool.dispatch_async`` in real DBOS context.
* The ``sys_read_inbox`` drain reading the real
  ``async_work_complete`` topic populated by a real
  ``background_tool_workflow``.
* That the LLM understands the ``[System: task ... completed]``
  format well enough to extract the marker.

Excluded from default ``pytest`` runs via ``--ignore=tests/e2e``.
Invoke with::

    pytest tests/e2e/test_sys_async_inbox_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

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
    Path(__file__).resolve().parents[1] / "_fixtures" / "agents" / "sys-async-inbox-test"
)


@pytest.fixture(scope="session")
def sys_async_inbox_agent(http_client: httpx.Client) -> str:
    """
    Upload the sys-async-inbox-test fixture agent.

    :param http_client: HTTP client pointed at the live server.
    :returns: The agent's name (matches its config.yaml ``name``).
    """
    import json as _json

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(str(_FIXTURE_DIR), arcname=".")
        bundle_path = tmp.name
    try:
        with open(bundle_path, "rb") as f:
            resp = http_client.post(
                "/v1/sessions",
                data={"metadata": _json.dumps({})},
                files={
                    "bundle": (
                        "agent.tar.gz",
                        f,
                        "application/gzip",
                    ),
                },
            )
        if resp.status_code == 409:
            # Already registered earlier in the same session.
            return _FIXTURE_DIR.name
        resp.raise_for_status()
        session_id = resp.json()["session_id"]
        agent_resp = http_client.get(f"/v1/sessions/{session_id}/agent")
        agent_resp.raise_for_status()
        return agent_resp.json()["name"]
    finally:
        Path(bundle_path).unlink(missing_ok=True)


def _create_response_blocking(
    http_client: httpx.Client,
    *,
    model: str,
    runner_id: str,
    user_text: str,
    timeout_s: float = 240.0,
) -> dict:
    """
    POST a session event, poll until terminal, return the final response-like body.

    :param http_client: HTTP client pointed at the live server.
    :param model: Agent name to invoke.
    :param runner_id: Registered runner id from ``live_runner_id``.
    :param user_text: Plain-text input message for the agent.
    :param timeout_s: Max seconds to wait for the response to
        complete. Default 240 s — the LLM may take 2-3 turns
        (dispatch → optional wait → drain → final answer) and
        the tool itself sleeps 2 s.
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


def _function_call_outputs_for(response_body: dict, call_ids: set[str]) -> list[str]:
    """
    Return the ``output`` strings of ``function_call_output`` items
    whose ``call_id`` is in ``call_ids``.

    Sister to :func:`tests.e2e.helpers.get_output_items` — that one
    filters by item type and tool name; this one filters by call id
    (the only stable join key from a function_call to its
    function_call_output, since the LLM may dispatch the same tool
    multiple times in one turn). Local rather than promoted to
    ``helpers.py`` because it's the only consumer at the moment.

    :param response_body: Response body from
        ``GET /v1/responses/{id}``.
    :param call_ids: Call IDs to match against function_call_output
        items.
    :returns: Matching string outputs in original order. Items whose
        ``output`` field is non-string are skipped (defensive — the
        contract is string-typed but loose JSON could theoretically
        deviate, and dropping silently keeps the assertion site
        clean).
    """
    return [
        item["output"]
        for item in response_body.get("output", [])
        if item.get("type") == "function_call_output"
        and item.get("call_id") in call_ids
        and isinstance(item.get("output"), str)
    ]


# ─── Tests ───────────────────────────────────────────────────


def test_sys_call_async_dispatch_marker_reaches_llm_e2e(
    http_client: httpx.Client,
    sys_async_inbox_agent: str,
    live_runner_id: str,
) -> None:
    """
    The LLM dispatches a tool via ``sys_call_async`` and the
    tool's marker reaches the LLM's final response.

    This pins the **handle-branch** of the new
    ``_execute_tools`` dispatch routing (the ``isinstance(...,
    _AsyncToolHandle)`` arm): ``sys_call_async`` returns an
    ``_AsyncToolHandle``, the workflow serializes it via
    ``to_handle_json``, the child ``background_tool_workflow``
    runs ``tag_label``, the parent's between-iteration drain
    auto-collects the completion as a ``[System: task ...
    completed]`` user message, and the LLM quotes the marker.

    A regression that broke ``sys_call_async`` end-to-end (e.g.,
    the runtime ``get_tool_manager()`` lookup not finding the
    target tool because the agent spec parser dropped local
    Python tools when ``async: true`` was set) would manifest
    as the marker missing from the final response.

    Marker delivery is permissive across two paths because LLM
    timing is not deterministic:

    * If the LLM finalizes the turn after ``sys_call_async``,
      the framework's between-iteration drain (auto-collect)
      delivers the completion as a system-role user message in
      the next turn's prompt.
    * If the LLM also calls ``sys_read_inbox`` in the same or a
      following turn AFTER the child workflow completed, that
      call's output carries the marker.

    Either path proves the dispatched tool's result reached the
    LLM. We assert presence in conversation history rather than
    fixing the path, so the test isn't flaky against
    LLM-internal call-ordering decisions.
    """
    body = _create_response_blocking(
        http_client,
        model=sys_async_inbox_agent,
        runner_id=live_runner_id,
        user_text=(
            "Dispatch the tag_label tool with label='gamma' using "
            "sys_call_async. After it completes, tell me the "
            "literal string the tool returned, "
            "character-for-character. You can call sys_read_inbox "
            "to drain results explicitly, or wait for the system "
            "to deliver the result automatically."
        ),
    )
    assert body["status"] == "completed", (
        f"sys_call_async turn did not complete: "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )

    # The LLM must actually have invoked sys_call_async —
    # otherwise the marker could only have come from
    # prompt-leakage, and the whole pipeline is being bypassed.
    # This is the load-bearing assertion that proves the
    # registration gate, schema surfacing, and dispatch-routing
    # all worked.
    call_async_calls = get_output_items(body, "function_call", name="sys_call_async")
    fc_names_for_diag = [
        item.get("name") for item in body.get("output", []) if item.get("type") == "function_call"
    ]
    assert len(call_async_calls) >= 1, (
        f"LLM did not call sys_call_async; check that the schema "
        f"surfaced and the agent's instructions are clear. "
        f"function_calls in output: {fc_names_for_diag!r}"
    )

    # The tool's literal marker must surface SOMEWHERE in the
    # response — final assistant text, sys_read_inbox output, or
    # an auto-delivered ``[System: task ...]`` user message in
    # the conversation. Each path validates a different piece of
    # the pipeline; any single one is sufficient to prove the
    # dispatch + drain worked end-to-end.
    final = final_assistant_text(body)
    output_items = body.get("output", [])
    auto_delivered_user_texts = [
        block.get("text", "")
        for item in output_items
        if item.get("type") == "message" and item.get("role") == "user"
        for block in item.get("content", [])
        if block.get("type") == "input_text"
    ]
    read_inbox_call_ids = {
        item["call_id"] for item in get_output_items(body, "function_call", name="sys_read_inbox")
    }
    read_inbox_outputs = _function_call_outputs_for(body, read_inbox_call_ids)

    marker = "SYS_ASYNC_TAG[gamma]"
    paths_with_marker = {
        "final_assistant_text": marker in final,
        "auto_delivered_user_message": any(marker in t for t in auto_delivered_user_texts),
        "sys_read_inbox_output": any(marker in o for o in read_inbox_outputs),
    }
    assert any(paths_with_marker.values()), (
        f"Tool marker {marker!r} did not reach the LLM via any "
        f"path. final={final!r}; auto_delivered_user_texts="
        f"{auto_delivered_user_texts!r}; read_inbox_outputs="
        f"{read_inbox_outputs!r}."
    )


def test_sys_read_inbox_str_branch_routes_without_crash_e2e(
    http_client: httpx.Client,
    sys_async_inbox_agent: str,
    live_runner_id: str,
) -> None:
    """
    Calling ``sys_read_inbox`` round-trips through the ``str``
    arm of ``_execute_tools`` without crashing the response.

    This is the **other** half of the workflow.py change in
    11a.ii. The ``isinstance(dispatched, _AsyncToolHandle)``
    branch covers ``sys_call_async``; this test pins the
    ``else`` branch by forcing the LLM to call
    ``sys_read_inbox`` first. A regression that left the old
    ``handle.to_handle_json()`` call in place would crash with
    ``AttributeError: 'str' object has no attribute
    'to_handle_json'`` and surface as ``status="failed"``.

    We don't assert on what's in the inbox (might or might not
    be empty depending on LLM behaviour); we only assert that
    the call happened and the response completed. Inbox content
    correctness is covered by the unit tests in
    ``tests/tools/builtins/test_async_inbox.py``.
    """
    body = _create_response_blocking(
        http_client,
        model=sys_async_inbox_agent,
        runner_id=live_runner_id,
        user_text=("Call sys_read_inbox right now and tell me what it returned, exactly."),
    )
    assert body["status"] == "completed", (
        f"sys_read_inbox-only turn did not complete (a regression "
        f"in the str-arm of _execute_tools would surface here): "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )
    read_inbox_calls = get_output_items(body, "function_call", name="sys_read_inbox")
    fc_names_for_diag = [
        item.get("name") for item in body.get("output", []) if item.get("type") == "function_call"
    ]
    assert len(read_inbox_calls) >= 1, (
        f"LLM did not call sys_read_inbox even when prompted "
        f"directly; tool may not be reaching the schema list. "
        f"function_calls in output: {fc_names_for_diag!r}"
    )
    # The output must be the empty-inbox sentinel OR a real
    # formatted block — both are valid; we just want to confirm
    # the str passed through to the LLM as-is.
    read_inbox_call_ids = {item["call_id"] for item in read_inbox_calls}
    outputs = _function_call_outputs_for(body, read_inbox_call_ids)
    assert outputs, "sys_read_inbox produced no function_call_output items"
    sentinel = "Inbox is empty — no completed tasks."
    valid_shape = any(out == sentinel or "[System: task " in out for out in outputs)
    assert valid_shape, (
        f"sys_read_inbox output did not match the documented "
        f"shape (sentinel string or [System: task ...] block). "
        f"Outputs: {outputs!r}"
    )


def test_sys_cancel_async_aborts_in_flight_dispatch_e2e(
    http_client: httpx.Client,
    sys_async_inbox_agent: str,
    live_runner_id: str,
) -> None:
    """
    The LLM dispatches a slow tool, cancels it via
    ``sys_cancel_async``, and the cancel surfaces end-to-end.

    Pipeline pinned by this test:

    * ``sys_call_async`` dispatches ``sleep_label`` (sleeps long
      enough that cancel can fire mid-run) — handle-branch of
      ``_execute_tools``.
    * The LLM extracts ``handle_id`` from the handle JSON's
      ``task_id`` field and passes it to
      ``sys_cancel_async``.
    * :class:`SysCancelAsyncTool.invoke` rekeys
      ``handle_id`` → ``task_id`` and delegates to
      :class:`SysCancelTaskTool` — which marks the task
      cancelled in ``task_store`` and (for kind=tool) lets the
      child workflow observe the cancel at its next checkpoint.
    * The child emits an ``async_work_complete`` payload with
      ``status="cancelled"``; the parent's drain renders it as
      ``[System: task ... cancelled]``.
    * The ``SLEEP_LABEL_DONE[...]`` marker MUST NOT appear,
      because the cancel beat the sleep.

    Verification is permissive about the path the cancel
    surface arrives via (final assistant text, auto-delivered
    user message, or ``sys_read_inbox`` output) — LLM timing is
    not deterministic. A regression in any branch (the rekey,
    the parent's invoke, the drain rendering) would either
    surface the SLEEP marker (cancel never landed) OR flip
    ``status`` to ``failed`` (a crash in the alias).
    """
    body = _create_response_blocking(
        http_client,
        model=sys_async_inbox_agent,
        runner_id=live_runner_id,
        user_text=(
            "Step 1: dispatch sleep_label with label='delta' and "
            "seconds=8 via sys_call_async. Step 2: as soon as you "
            "see the handle, call sys_cancel_async passing the "
            "handle's task_id as handle_id. Step 3: wait for the "
            "cancellation to be confirmed (via sys_read_inbox or "
            "the auto-delivered system message), then tell me "
            "exactly what status the task ended in."
        ),
        # Bumped past the default 240s so the LLM has time to
        # complete a 2-3 turn dispatch → cancel → confirm cycle
        # without flaking on the polling deadline.
        timeout_s=300.0,
    )
    assert body["status"] == "completed", (
        f"sys_cancel_async turn did not complete (a regression "
        f"in the alias's invoke would flip this to 'failed'): "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )

    # The LLM must have actually used both new tools; otherwise
    # the cancel-pipeline wasn't exercised end-to-end.
    call_async_calls = get_output_items(body, "function_call", name="sys_call_async")
    cancel_async_calls = get_output_items(body, "function_call", name="sys_cancel_async")
    fc_names_for_diag = [
        item.get("name") for item in body.get("output", []) if item.get("type") == "function_call"
    ]
    assert len(call_async_calls) >= 1, (
        f"LLM did not call sys_call_async; pipeline not exercised. "
        f"function_calls in output: {fc_names_for_diag!r}"
    )
    assert len(cancel_async_calls) >= 1, (
        f"LLM did not call sys_cancel_async; the cancel surface "
        f"isn't reaching the schema list, OR the LLM ignored the "
        f"step-2 instruction. "
        f"function_calls in output: {fc_names_for_diag!r}"
    )

    # The cancel acknowledgement must surface via SOME path. We
    # accept the same three delivery paths as the dispatch test
    # (final text, auto-delivered system message, sys_read_inbox
    # output) because LLM timing is non-deterministic.
    final = final_assistant_text(body)
    output_items = body.get("output", [])
    auto_delivered_user_texts = [
        block.get("text", "")
        for item in output_items
        if item.get("type") == "message" and item.get("role") == "user"
        for block in item.get("content", [])
        if block.get("type") == "input_text"
    ]
    cancel_async_outputs = _function_call_outputs_for(
        body, {item["call_id"] for item in cancel_async_calls}
    )
    read_inbox_outputs = _function_call_outputs_for(
        body,
        {
            item["call_id"]
            for item in get_output_items(body, "function_call", name="sys_read_inbox")
        },
    )

    cancelled_marker_present = (
        "cancelled" in final.lower()
        or any("cancelled" in t.lower() for t in auto_delivered_user_texts)
        or any("cancelled" in o.lower() for o in cancel_async_outputs)
        or any("cancelled" in o.lower() for o in read_inbox_outputs)
    )
    assert cancelled_marker_present, (
        f"No 'cancelled' acknowledgement surfaced via any path. "
        f"final={final!r}; auto_delivered_user_texts="
        f"{auto_delivered_user_texts!r}; "
        f"cancel_async_outputs={cancel_async_outputs!r}; "
        f"read_inbox_outputs={read_inbox_outputs!r}."
    )

    # The dispatched sleep_label must NOT have completed —
    # cancel beat the sleep. A regression that no-op'd
    # sys_cancel_async would let the sleep finish and the
    # SLEEP_LABEL_DONE marker would surface alongside the
    # 'cancelled' wording.
    sleep_marker = "SLEEP_LABEL_DONE[delta]"
    surfaces_with_sleep = (
        sleep_marker in final
        or any(sleep_marker in t for t in auto_delivered_user_texts)
        or any(sleep_marker in o for o in read_inbox_outputs)
    )
    assert not surfaces_with_sleep, (
        f"The dispatched sleep_label appears to have completed — "
        f"the {sleep_marker!r} marker is present, which means "
        f"sys_cancel_async did not actually abort the task. "
        f"final={final!r}; user_texts={auto_delivered_user_texts!r}; "
        f"inbox_outputs={read_inbox_outputs!r}."
    )
