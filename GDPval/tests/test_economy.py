from gdpval_timing.models import Message, ToolResult
from gdpval_timing.runner import Harness


def test_context_compaction_preserves_recent_and_strips_old_payloads():
    messages = []
    for i in range(3):
        messages.append(Message(role="tool", tool_results=[ToolResult(str(i), "view_image", "x" * 20, image_data_url="data:image/png;base64,abc")]))
    Harness._compact_history(messages, {
        "retain_recent_tool_turns": 2,
        "max_historical_tool_result_chars": 5,
        "image_payload_retention_turns": 1,
    })
    assert messages[0].tool_results[0].output.startswith("xxxxx\n[historical")
    assert messages[0].tool_results[0].image_data_url is None
    assert messages[1].tool_results[0].output.startswith("x" * 20)
    assert messages[1].tool_results[0].image_data_url is None
    assert messages[2].tool_results[0].image_data_url is not None


def test_error_signature_ignores_changing_numbers():
    assert Harness._error_signature("exit 12 after 3.5 seconds") == Harness._error_signature("exit 99 after 8.2 seconds")


def test_finalize_prompt_is_artifact_aware():
    assert "validated" in Harness._finalize_prompt(True)
    assert "Create" in Harness._finalize_prompt(False)
