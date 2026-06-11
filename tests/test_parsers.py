"""Unit tests for the Vivado report parsers in server.py."""

from vivado_mcp.server import (
    parse_timing_summary,
    parse_utilization,
    parse_messages,
    parse_timing_paths_summary,
    truncate_response,
    get_hierarchy_depth,
    verify_run_status,
    _hw_server_reachable,
)


# ---------------------------------------------------------------------------
# parse_timing_summary
# ---------------------------------------------------------------------------

TABULAR_TIMING_MET = """\
Design Timing Summary
| ---------------------

    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints
    -------      -------  ---------------------  -------------------      -------      -------  ---------------------  -------------------     --------     --------  ----------------------  --------------------
     14.111        0.000                      0                 5688        0.012        0.000                      0                 5688        9.238        0.000                       0                  1752


All user specified timing constraints are met.
"""

TABULAR_TIMING_VIOLATED = """\
Design Timing Summary
| ---------------------

    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints
    -------      -------  ---------------------  -------------------      -------      -------  ---------------------  -------------------     --------     --------  ----------------------  --------------------
     -0.512       -3.420                     12                 5688        0.012        0.000                      0                 5688        9.238        0.000                       0                  1752
"""

LEGACY_TIMING = """\
WNS(ns) :  1.234
TNS(ns) :  0.000
WHS(ns) :  0.056
THS(ns) :  0.000
There are 0 failing endpoints
"""


def test_tabular_timing_met():
    result = parse_timing_summary(TABULAR_TIMING_MET)
    assert result["wns"] == 14.111
    assert result["tns"] == 0.0
    assert result["whs"] == 0.012
    assert result["ths"] == 0.0
    assert result["wpws"] == 9.238
    assert result["failing_endpoints"] == 0
    assert result["met"] is True


def test_tabular_timing_violated():
    result = parse_timing_summary(TABULAR_TIMING_VIOLATED)
    assert result["wns"] == -0.512
    assert result["tns"] == -3.420
    assert result["failing_endpoints"] == 12
    assert result["met"] is False


def test_legacy_timing_format():
    result = parse_timing_summary(LEGACY_TIMING)
    assert result["wns"] == 1.234
    assert result["whs"] == 0.056
    assert result["met"] is True


def test_timing_unparseable_output():
    result = parse_timing_summary("no timing data here")
    assert result["wns"] is None
    assert result["met"] is False


# ---------------------------------------------------------------------------
# parse_utilization
# ---------------------------------------------------------------------------

UTILIZATION_2021 = """\
+----------------------------+------+-------+-----------+-------+
|          Site Type         | Used | Fixed | Available | Util% |
+----------------------------+------+-------+-----------+-------+
| Slice LUTs                 |  100 |     0 |     53200 |  0.19 |
| Slice Registers            |  200 |     0 |    106400 |  0.19 |
| Block RAM Tile             |  2.5 |     0 |       140 |  1.79 |
| DSPs                       |    3 |     0 |       220 |  1.36 |
| Bonded IOB                 |   10 |     0 |       200 |  5.00 |
+----------------------------+------+-------+-----------+-------+
"""

UTILIZATION_2022 = """\
+----------------------------+------+-------+------------+-----------+-------+
|          Site Type         | Used | Fixed | Prohibited | Available | Util% |
+----------------------------+------+-------+------------+-----------+-------+
| CLB LUTs                   |  100 |     0 |          0 |    230400 |  0.04 |
| CLB Registers              |  200 |     0 |          0 |    460800 |  0.04 |
| Block RAM Tile             |  2.5 |     0 |          0 |       312 |  0.80 |
| DSPs                       |    3 |     0 |          0 |      1728 |  0.17 |
| Bonded IOB                 |   10 |     0 |          0 |       360 |  2.78 |
+----------------------------+------+-------+------------+-----------+-------+
"""


def test_utilization_2021_format():
    result = parse_utilization(UTILIZATION_2021)
    assert result["lut"]["used"] == 100
    assert result["lut"]["available"] == 53200
    assert result["lut"]["percent"] == 0.19
    assert result["ff"]["used"] == 200
    assert result["bram"]["used"] == 2.5
    assert result["dsp"]["used"] == 3
    assert result["io"]["used"] == 10
    assert result["io"]["percent"] == 5.0


def test_utilization_2022_format_with_prohibited_column():
    result = parse_utilization(UTILIZATION_2022)
    assert result["lut"]["used"] == 100
    assert result["lut"]["available"] == 230400
    assert result["lut"]["percent"] == 0.04
    assert result["bram"]["used"] == 2.5
    assert result["bram"]["available"] == 312
    assert result["io"]["available"] == 360


def test_utilization_empty_output():
    result = parse_utilization("nothing useful")
    assert result["lut"]["used"] == 0
    assert result["lut"]["available"] == 0


# ---------------------------------------------------------------------------
# parse_messages
# ---------------------------------------------------------------------------

def test_parse_messages_categorizes_by_severity():
    output = "\n".join([
        "ERROR: [Synth 8-87] cannot open file",
        "CRITICAL WARNING: [Vivado 12-1411] something serious",
        "WARNING: [Synth 8-3331] design has unconnected port",
        "INFO: [Synth 8-7075] Helper message",
        "Just a normal line",
    ])
    result = parse_messages(output)
    assert len(result["errors"]) == 1
    assert len(result["critical_warnings"]) == 1
    assert len(result["warnings"]) == 1
    assert len(result["info"]) == 1
    assert "Synth 8-87" in result["errors"][0]


# ---------------------------------------------------------------------------
# parse_timing_paths_summary
# ---------------------------------------------------------------------------

TIMING_PATH_REPORT = """\
Slack (VIOLATED) :        -0.234ns  (required time - arrival time)
  Source:                 cpu/alu/reg_a_reg[3]/C
  Destination:            cpu/alu/result_reg[7]/D
  Source Clock:           clk
  Destination Clock:      clk
  Requirement:            10.000ns  (clk rise@10.000ns - clk rise@0.000ns)
  Data Path Delay:        9.876ns  (logic 3.210ns (32.5%)  route 6.666ns (67.5%))
  Logic Levels:           7  (LUT6=4 CARRY4=3)

Slack (MET) :             0.512ns  (required time - arrival time)
  Source:                 uart/tx_reg/C
  Destination:            uart/busy_reg/D
  Requirement:            10.000ns
  Data Path Delay:        8.123ns
  Logic Levels:           3
"""


def test_parse_timing_paths():
    paths = parse_timing_paths_summary(TIMING_PATH_REPORT)
    assert len(paths) == 2
    assert paths[0]["slack"] == -0.234
    assert paths[0]["source"] == "cpu/alu/reg_a_reg[3]/C"
    assert paths[0]["destination"] == "cpu/alu/result_reg[7]/D"
    assert paths[0]["source_clock"] == "clk"
    assert paths[0]["requirement"] == 10.0
    assert paths[0]["data_path_delay"] == 9.876
    assert paths[0]["logic_levels"] == 7
    assert paths[1]["slack"] == 0.512


def test_parse_timing_paths_respects_max():
    paths = parse_timing_paths_summary(TIMING_PATH_REPORT, max_paths=1)
    assert len(paths) == 1


def test_parse_timing_paths_empty():
    assert parse_timing_paths_summary("No timing paths found.") == []


# ---------------------------------------------------------------------------
# truncate_response
# ---------------------------------------------------------------------------

def test_truncate_short_content_passthrough():
    result = truncate_response("short", max_chars=100)
    assert result["truncated"] is False
    assert result["content"] == "short"


def test_truncate_long_content():
    content = "\n".join(f"line {i}" for i in range(1000))
    result = truncate_response(content, max_chars=500)
    assert result["truncated"] is True
    assert len(result["content"]) <= 500
    assert result["total_chars"] == len(content)
    # Should end on a line boundary
    assert not result["content"].endswith("lin")
    assert "truncation_message" in result


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------

def test_hierarchy_depth():
    assert get_hierarchy_depth("top") == 0
    assert get_hierarchy_depth("top/cpu") == 1
    assert get_hierarchy_depth("top/cpu/alu/adder") == 3


def test_hw_server_reachable_rejects_malformed_urls():
    assert _hw_server_reachable("not-a-url") is False
    assert _hw_server_reachable("TCP:host:notaport") is False


class _FakeResult:
    def __init__(self, output, success=True):
        self.output = output
        self.success = success


class _FakeSession:
    """Returns canned STATUS/PROGRESS responses for verify_run_status."""

    def __init__(self, status, progress="100%"):
        self._responses = {"STATUS": status, "PROGRESS": progress}

    def run_tcl(self, command, **kwargs):
        for key, value in self._responses.items():
            if key in command:
                return _FakeResult(value)
        return _FakeResult("", success=False)


def test_verify_run_status_success():
    info = verify_run_status(_FakeSession("route_design Complete!"), "impl_1")
    assert info["actually_succeeded"] is True
    assert info["actually_failed"] is False


def test_verify_run_status_failure():
    info = verify_run_status(_FakeSession("synth_design ERROR"), "synth_1")
    assert info["actually_succeeded"] is False
    assert info["actually_failed"] is True
