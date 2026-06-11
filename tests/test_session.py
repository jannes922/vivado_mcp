"""Unit tests for error classification and session helpers in vivado_session.py."""

from vivado_mcp.vivado_session import (
    classify_output_errors,
    CommandResult,
    VivadoSession,
)


# ---------------------------------------------------------------------------
# classify_output_errors
# ---------------------------------------------------------------------------

def test_vivado_error_detected():
    output = "ERROR: [Synth 8-87] cannot open file 'missing.v'"
    c = classify_output_errors(output, "read_verilog missing.v")
    assert c.is_vivado_error is True
    assert c.is_actual_failure is True
    assert "Synth 8-87" in c.error_messages[0]


def test_tcl_syntax_error_detected():
    output = 'invalid command name "opne_project"'
    c = classify_output_errors(output, "opne_project foo.xpr")
    assert c.is_tcl_error is True
    assert c.is_actual_failure is True


def test_wrong_args_error_detected():
    output = 'wrong # args: should be "open_project file"'
    c = classify_output_errors(output, "open_project")
    assert c.is_tcl_error is True


def test_report_content_not_an_error():
    # "ERROR" appearing as table data must not be classified as a failure
    output = "\n".join([
        "Design Timing Summary",
        "| Timing ERROR      |  0  |",
        "WNS(ns): -0.5  TNS ERROR: 0",
        "+---------+------+",
    ])
    c = classify_output_errors(output, "report_timing_summary")
    assert c.is_actual_failure is False


def test_error_must_start_line_with_bracket_code():
    # ERROR: without a [code] bracket is not a Vivado tool error line
    output = "the previous run printed ERROR: something informal"
    c = classify_output_errors(output, "puts ...")
    assert c.is_vivado_error is False


def test_clean_output_is_success():
    c = classify_output_errors("xc7z020clg400-1", "get_property PART [current_project]")
    assert c.is_actual_failure is False
    assert c.error_messages == []


# ---------------------------------------------------------------------------
# CommandResult / VivadoSession state behavior (no Vivado process needed)
# ---------------------------------------------------------------------------

def test_command_result_fields():
    r = CommandResult(
        command="puts hi", output="hi", return_value="0",
        success=True, elapsed_ms=1.5,
    )
    assert r.success is True
    assert r.timestamp  # auto-populated


def test_run_tcl_requires_running_session():
    session = VivadoSession()
    result = session.run_tcl("current_project")
    assert result.success is False
    assert "not running" in result.output


def test_stop_when_not_running_is_noop():
    session = VivadoSession()
    result = session.stop()
    assert result.success is True


def test_is_healthy_false_when_not_running():
    session = VivadoSession()
    assert session.is_healthy() is False
