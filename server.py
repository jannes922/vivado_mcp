#!/usr/bin/env python3
"""
Vivado MCP Server - Direct integration with AMD/Xilinx Vivado.

This module implements a Model Context Protocol (MCP) server that provides
AI assistants (like Claude) with direct access to AMD/Xilinx Vivado FPGA
development tools. It enables:

- Session management: Start/stop persistent Vivado TCL sessions
- Project management: Open/close Vivado projects (.xpr files)
- Design flow: Run synthesis, implementation, and bitstream generation
- Reports: Get timing, utilization, and other analysis reports
- Design queries: Explore hierarchy, ports, nets, and cells
- Simulation: Control Vivado's behavioral simulator (xsim)
- Raw TCL: Execute arbitrary TCL commands for advanced operations

Architecture:
    The server maintains a singleton VivadoSession that keeps Vivado running
    in TCL mode. Commands are sent via pexpect and results are parsed and
    returned as structured JSON. This avoids the ~30 second startup time
    for each Vivado command.

MCP Protocol:
    The server uses the MCP stdio transport, communicating via stdin/stdout
    with JSON-RPC messages. Tools are exposed via the @server.list_tools()
    and @server.call_tool() decorators.

Usage:
    # Start the server (typically done by Claude Code or another MCP client)
    python -m vivado_mcp

    # Or via the console script (after pip install)
    vivado-mcp

Example workflow (from an AI assistant):
    1. start_session - Start Vivado
    2. open_project - Open your .xpr file
    3. run_synthesis - Synthesize the design
    4. get_timing_summary - Check timing results
    5. get_utilization - Check resource usage
    6. stop_session - Clean up when done

Author: Created with Claude (Anthropic)
License: MIT
"""

import asyncio
import json
import os
import re
import socket
import uuid
from datetime import datetime
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .vivado_session import get_session, VivadoSession


def _hw_server_reachable(hw_server_url: str, timeout: float = 1.0) -> bool:
    """Return True if hw_server is accepting TCP connections at the given URL.

    Vivado's `connect_hw_server` blocks for ~30s trying to reach hw_server if
    nothing is there, and that hang leaks into the persistent Tcl session —
    subsequent commands queue behind it and appear to hang forever. A cheap
    Python-side TCP check lets us bail out before the Vivado call and keep
    the session clean.

    Accepts URLs in either `TCP:host:port` or `host:port` form.
    """
    url = hw_server_url[4:] if hw_server_url.startswith("TCP:") else hw_server_url
    if ":" not in url:
        return False
    host, _, port_s = url.rpartition(":")
    try:
        port = int(port_s)
    except ValueError:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.error):
        return False


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Feature requests are stored persistently so users can track requested features.
# Stored under the XDG data dir, NOT the package directory — site-packages is
# read-only for non-editable installs (and ephemeral under Nix).
FEATURE_REQUESTS_FILE = (
    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "vivado_mcp" / "feature_requests.json"
)

# Legacy location (inside the package) — read as fallback so existing
# editable installs don't lose previously submitted requests
_LEGACY_FEATURE_REQUESTS_FILE = Path(__file__).parent / "data" / "feature_requests.json"

# Report management configuration
# Reports are written to temp files when they exceed inline size limits
REPORTS_DIR = Path("/tmp/vivado_mcp")

# Maximum characters to return inline in a response
# Larger reports should use generate_full_report + read_report_section
MAX_RESPONSE_CHARS = 8000  # ~8KB limit for inline responses

# How long to keep cached report files before cleanup (in hours)
REPORT_CACHE_HOURS = 1

# In-memory cache mapping report_id -> metadata (file path, type, etc.)
# This allows quick lookup of previously generated reports
_report_cache: dict[str, dict] = {}


# =============================================================================
# FEATURE REQUEST MANAGEMENT
# =============================================================================

def load_feature_requests() -> list[dict]:
    """
    Load feature requests from the persistent JSON file.

    Feature requests allow the AI assistant to record when it encounters
    limitations or wishes it had a tool that doesn't exist. This helps
    guide future development of the MCP server.

    Returns:
        List of feature request dictionaries, or empty list if file
        doesn't exist or can't be parsed.
    """
    for path in (FEATURE_REQUESTS_FILE, _LEGACY_FEATURE_REQUESTS_FILE):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, IOError):
                continue
    return []


def save_feature_request(request: dict) -> None:
    """
    Save a feature request to the persistent JSON file.

    Args:
        request: Dictionary containing the feature request with fields:
            - id: Auto-assigned sequential ID
            - title: Short description of the feature
            - description: Detailed explanation of what's needed
            - use_case: The specific task that prompted this request
            - priority: low/medium/high
            - timestamp: ISO format timestamp
            - status: "pending" (could be updated to "implemented" later)
    """
    requests = load_feature_requests()
    requests.append(request)
    # Ensure the data directory exists
    FEATURE_REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_REQUESTS_FILE.write_text(json.dumps(requests, indent=2))


# =============================================================================
# RESPONSE TRUNCATION
# =============================================================================

def truncate_response(content: str, max_chars: int = MAX_RESPONSE_CHARS) -> dict:
    """
    Truncate response content if it exceeds max_chars.

    Large Vivado reports can be tens of thousands of lines. Rather than
    overwhelming the AI context window, we truncate and provide metadata
    about what was cut. The user can then use generate_full_report to
    get the complete output to a file.

    Args:
        content: The full content string to potentially truncate
        max_chars: Maximum characters to return (default: MAX_RESPONSE_CHARS)

    Returns:
        Dictionary with:
            - content: The (possibly truncated) content
            - truncated: Boolean indicating if truncation occurred
            - total_chars: Original content length
            - total_lines: Original line count
            - returned_chars: Characters in truncated content (if truncated)
            - returned_lines: Lines in truncated content (if truncated)
            - truncation_message: Human-readable message about truncation
    """
    total_chars = len(content)
    total_lines = content.count('\n') + 1

    # If content fits, return it unchanged
    if total_chars <= max_chars:
        return {
            "content": content,
            "truncated": False,
            "total_chars": total_chars,
            "total_lines": total_lines
        }

    # Truncate to max_chars, but try to end at a line boundary
    # This makes the output more readable and avoids cutting mid-line
    truncated_content = content[:max_chars]
    last_newline = truncated_content.rfind('\n')

    # Only use the newline boundary if we keep >80% of the allowed content
    # Otherwise we might lose too much useful data
    if last_newline > max_chars * 0.8:
        truncated_content = truncated_content[:last_newline]

    truncated_lines = truncated_content.count('\n') + 1

    return {
        "content": truncated_content,
        "truncated": True,
        "total_chars": total_chars,
        "total_lines": total_lines,
        "returned_chars": len(truncated_content),
        "returned_lines": truncated_lines,
        "truncation_message": f"Output truncated ({total_chars:,} chars -> {len(truncated_content):,} chars). Use generate_full_report for complete output."
    }


def verify_run_status(session, run_name: str) -> dict:
    """
    Verify actual Vivado run status instead of relying on output parsing.

    Vivado run status is stored as properties on the run object. This function
    queries those properties directly, which is more reliable than parsing
    text output that may contain misleading strings.

    Args:
        session: VivadoSession instance
        run_name: Name of the run to check (e.g., "synth_1", "impl_1")

    Returns:
        Dictionary with:
        - run_name: The run that was checked
        - status: Vivado's STATUS property (e.g., "synth_design Complete!")
        - progress: Vivado's PROGRESS property (e.g., "100%")
        - actually_succeeded: True if run completed successfully
        - actually_failed: True if run failed
    """
    status_result = session.run_tcl(f"get_property STATUS [get_runs {run_name}]")
    progress_result = session.run_tcl(f"get_property PROGRESS [get_runs {run_name}]")

    status = status_result.output.strip() if status_result.success else "unknown"
    progress = progress_result.output.strip() if progress_result.success else "unknown"

    # Determine actual success/failure from status string
    # Successful runs have "Complete!" in status
    # Failed runs have "ERROR" in status
    status_lower = status.lower()
    return {
        "run_name": run_name,
        "status": status,
        "progress": progress,
        "actually_succeeded": "complete" in status_lower,
        "actually_failed": "error" in status_lower,
    }


# =============================================================================
# REPORT FILE MANAGEMENT
# =============================================================================

def ensure_reports_dir() -> Path:
    """
    Ensure the reports directory exists and clean up old reports.

    This function is called before generating new reports. It:
    1. Creates the reports directory if it doesn't exist
    2. Removes any report files older than REPORT_CACHE_HOURS
    3. Cleans up the in-memory cache for deleted files

    Returns:
        Path to the reports directory
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate cutoff timestamp for old reports
    cutoff = datetime.now().timestamp() - (REPORT_CACHE_HOURS * 3600)

    # Scan for and remove old report files
    for report_file in REPORTS_DIR.glob("*.txt"):
        try:
            if report_file.stat().st_mtime < cutoff:
                report_file.unlink()
                # Also remove from in-memory cache if present
                report_id = report_file.stem
                _report_cache.pop(report_id, None)
        except OSError:
            pass  # Ignore errors during cleanup

    return REPORTS_DIR


def generate_report_id() -> str:
    """
    Generate a unique 8-character report ID.

    Uses UUID4 for uniqueness, truncated to 8 chars for readability.
    The ID is used to reference reports across tool calls.

    Returns:
        8-character hexadecimal string (e.g., "a1b2c3d4")
    """
    return str(uuid.uuid4())[:8]


def get_hierarchy_depth(path: str) -> int:
    """
    Get the depth of a hierarchical path.

    Vivado uses "/" to separate hierarchy levels (e.g., "cpu/alu/adder").
    This function counts the depth to help filter hierarchy queries.

    Args:
        path: Hierarchical path string

    Returns:
        Depth as integer (0 for top level, 1 for first level children, etc.)
    """
    return path.count('/')


# =============================================================================
# MCP SERVER INSTANCE
# =============================================================================

# Create the MCP server instance
# The name "vivado" is used as the server identifier in MCP communications
server = Server("vivado")


# =============================================================================
# VIVADO OUTPUT PARSERS
# =============================================================================
# These functions parse Vivado's text-based reports into structured data
# that's easier for AI assistants to work with.

def parse_timing_summary(output: str) -> dict:
    """
    Parse a Vivado timing summary report into structured data.

    Timing summary reports contain critical information about whether
    the design meets timing requirements. Key metrics:

    - WNS (Worst Negative Slack): Most critical setup timing margin
      Positive = timing met, Negative = timing violation
    - TNS (Total Negative Slack): Sum of all negative setup slacks
    - WHS (Worst Hold Slack): Most critical hold timing margin
    - THS (Total Hold Slack): Sum of all negative hold slacks
    - WPWS (Worst Pulse Width Slack): For pulse width requirements
    - TPWS (Total Pulse Width Slack): Sum of pulse width violations

    Args:
        output: Raw text output from report_timing_summary

    Returns:
        Dictionary with parsed metrics and "met" boolean indicating
        if all timing is met (WNS >= 0 and WHS >= 0)
    """
    result = {
        "wns": None,   # Worst Negative Slack (setup)
        "tns": None,   # Total Negative Slack (setup)
        "whs": None,   # Worst Hold Slack
        "ths": None,   # Total Hold Slack
        "wpws": None,  # Worst Pulse Width Slack
        "tpws": None,  # Total Pulse Width Slack
        "failing_endpoints": 0,
        "met": False,
        "raw": output  # Keep raw output for detailed analysis
    }

    # Vivado's report_timing_summary prints the metrics in a tabular layout:
    #
    #     WNS(ns)   TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints   WHS(ns)   THS(ns)  THS Failing Endpoints  THS Total Endpoints   WPWS(ns)  TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints
    #     -------   -------  ---------------------  -------------------   -------   -------  ---------------------  -------------------   --------  --------  ----------------------  --------------------
    #      14.111     0.000                      0                 5688     0.012     0.000                      0                 5688      9.238     0.000                       0                  1752
    #
    # We locate the header line, skip the dashed separator, then parse the
    # first non-empty data row that follows. Twelve tokens are expected:
    #   WNS, TNS, TNS_fail, TNS_total, WHS, THS, THS_fail, THS_total,
    #   WPWS, TPWS, TPWS_fail, TPWS_total
    #
    # The older "WNS(ns) :  1.234" colon-separated form is still supported as
    # a fallback for pre-2022 Vivado releases.
    lines = output.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"WNS\(ns\)\s+TNS\(ns\)", line):
            header_idx = i
            break

    parsed_table = False
    if header_idx is not None:
        # The data row is the first line after the separator that has numeric tokens
        for j in range(header_idx + 1, min(header_idx + 6, len(lines))):
            row = lines[j].strip()
            # Skip the dashed separator row
            if not row or set(row) <= set("- "):
                continue
            tokens = row.split()
            # Need at least 8 numeric columns to cover setup + hold
            if len(tokens) >= 8 and re.match(r"^-?\d+\.?\d*$", tokens[0]):
                try:
                    result["wns"] = float(tokens[0])
                    result["tns"] = float(tokens[1])
                    setup_failing = int(tokens[2])
                    result["whs"] = float(tokens[4])
                    result["ths"] = float(tokens[5])
                    hold_failing = int(tokens[6])
                    if len(tokens) >= 12:
                        result["wpws"] = float(tokens[8])
                        result["tpws"] = float(tokens[9])
                    # "failing_endpoints" historically reports setup failures
                    result["failing_endpoints"] = setup_failing + hold_failing
                    parsed_table = True
                    break
                except (ValueError, IndexError):
                    continue

    if not parsed_table:
        # Legacy colon-separated fallback (pre-2022 Vivado)
        for key in ("wns", "tns", "whs", "ths", "wpws", "tpws"):
            m = re.search(rf"{key.upper()}\(ns\)\s*:\s*([-\d.]+)", output)
            if m:
                result[key] = float(m.group(1))
        fail_match = re.search(r"(\d+)\s+failing\s+endpoint", output, re.IGNORECASE)
        if fail_match:
            result["failing_endpoints"] = int(fail_match.group(1))

    # Timing is met when both setup and hold slacks are non-negative
    if result["wns"] is not None and result["whs"] is not None:
        result["met"] = result["wns"] >= 0 and result["whs"] >= 0

    return result


def parse_utilization(output: str) -> dict:
    """
    Parse a Vivado utilization report into structured data.

    Utilization reports show how much of each FPGA resource type is used.
    This is critical for understanding if a design will fit and for
    optimization decisions.

    Resource types tracked:
    - LUT: Look-Up Tables (combinational logic)
    - FF: Flip-Flops (registers/sequential logic)
    - BRAM: Block RAM (on-chip memory)
    - DSP: DSP slices (multipliers, MACs)
    - IO: Input/Output pins

    Args:
        output: Raw text output from report_utilization

    Returns:
        Dictionary with each resource type containing:
        - used: Number of resources used
        - available: Total resources on the device
        - percent: Utilization percentage
    """
    result = {
        "lut": {"used": 0, "available": 0, "percent": 0},
        "ff": {"used": 0, "available": 0, "percent": 0},
        "bram": {"used": 0, "available": 0, "percent": 0},
        "dsp": {"used": 0, "available": 0, "percent": 0},
        "io": {"used": 0, "available": 0, "percent": 0},
        "raw": output  # Keep raw output for detailed analysis
    }

    # Regex patterns for each resource type.
    #
    # Vivado's table format varies between releases:
    #   2021 and earlier:  Used | Fixed |              Available | Util%
    #   2022 and later:    Used | Fixed | Prohibited | Available | Util%
    #
    # The optional (?:\s*\d+\.?\d*\s*\|)? group matches the Prohibited column
    # when present so the same pattern works across versions.
    patterns = {
        "lut": r"(?:Slice LUTs|CLB LUTs)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|(?:\s*\d+\s*\|)?\s*(\d+)\s*\|\s*([\d.]+)",
        "ff": r"(?:Slice Registers|CLB Registers)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|(?:\s*\d+\s*\|)?\s*(\d+)\s*\|\s*([\d.]+)",
        "bram": r"Block RAM Tile\s*\|\s*(\d+\.?\d*)\s*\|\s*\d+\.?\d*\s*\|(?:\s*\d+\.?\d*\s*\|)?\s*(\d+\.?\d*)\s*\|\s*([\d.]+)",
        "dsp": r"DSPs?\s*\|\s*(\d+)\s*\|\s*\d+\s*\|(?:\s*\d+\s*\|)?\s*(\d+)\s*\|\s*([\d.]+)",
        "io": r"(?:Bonded IOB|Bonded User I/O)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|(?:\s*\d+\s*\|)?\s*(\d+)\s*\|\s*([\d.]+)"
    }

    # Apply each pattern and extract values
    for resource, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            result[resource]["used"] = float(match.group(1))
            result[resource]["available"] = float(match.group(2))
            result[resource]["percent"] = float(match.group(3))

    return result


def parse_messages(output: str) -> dict:
    """
    Parse Vivado messages into categorized lists.

    Vivado outputs messages with severity prefixes:
    - ERROR: Design or tool errors that must be fixed
    - CRITICAL WARNING: Serious issues that may cause problems
    - WARNING: Potential issues to review
    - INFO: Informational messages

    Args:
        output: Raw text output containing Vivado messages

    Returns:
        Dictionary with lists of messages by category
    """
    result = {
        "errors": [],
        "critical_warnings": [],
        "warnings": [],
        "info": [],
        "raw": output
    }

    # Categorize each line by its severity prefix
    for line in output.split("\n"):
        line = line.strip()
        if re.match(r"ERROR:", line):
            result["errors"].append(line)
        elif re.match(r"CRITICAL WARNING:", line):
            result["critical_warnings"].append(line)
        elif re.match(r"WARNING:", line):
            result["warnings"].append(line)
        elif re.match(r"INFO:", line):
            result["info"].append(line)

    return result


def parse_timing_paths_summary(output: str, max_paths: int = 5) -> list[dict]:
    """
    Extract structured summary of timing paths from report_timing output.

    Parses Vivado's timing path reports to extract key information about
    each path without the verbose detailed breakdown.

    Args:
        output: Raw text output from report_timing command
        max_paths: Maximum number of paths to return (default: 5)

    Returns:
        List of dictionaries, each containing:
        - slack: Path slack in ns (negative = failing)
        - source: Source register/port name
        - destination: Destination register/port name
        - source_clock: Source clock domain (if applicable)
        - dest_clock: Destination clock domain (if applicable)
        - requirement: Timing requirement in ns
        - data_path_delay: Data path delay in ns
        - logic_levels: Number of logic levels
    """
    paths = []

    # Split output into individual path blocks
    # Each path starts with "Slack" line
    path_blocks = re.split(r'\n(?=Slack\s*(?:\([A-Z]+\))?\s*:)', output)

    for block in path_blocks:
        if not block.strip() or 'Slack' not in block:
            continue

        path_info = {}

        # Extract slack value
        slack_match = re.search(r'Slack\s*(?:\([A-Z]+\))?\s*:\s*([-\d.]+)\s*ns', block)
        if slack_match:
            path_info['slack'] = float(slack_match.group(1))

        # Extract source (startpoint)
        source_match = re.search(r'Source:\s*(\S+)', block)
        if source_match:
            path_info['source'] = source_match.group(1)

        # Extract destination (endpoint)
        dest_match = re.search(r'Destination:\s*(\S+)', block)
        if dest_match:
            path_info['destination'] = dest_match.group(1)

        # Extract source clock
        src_clk_match = re.search(r'Source Clock:\s*(\S+)', block)
        if src_clk_match:
            path_info['source_clock'] = src_clk_match.group(1)

        # Extract destination clock
        dst_clk_match = re.search(r'Destination Clock:\s*(\S+)', block)
        if dst_clk_match:
            path_info['dest_clock'] = dst_clk_match.group(1)

        # Extract requirement
        req_match = re.search(r'Requirement:\s*([-\d.]+)\s*ns', block)
        if req_match:
            path_info['requirement'] = float(req_match.group(1))

        # Extract data path delay
        data_delay_match = re.search(r'Data Path Delay:\s*([-\d.]+)\s*ns', block)
        if data_delay_match:
            path_info['data_path_delay'] = float(data_delay_match.group(1))

        # Extract logic levels
        levels_match = re.search(r'Logic Levels:\s*(\d+)', block)
        if levels_match:
            path_info['logic_levels'] = int(levels_match.group(1))

        # Only add if we got meaningful data
        if 'slack' in path_info:
            paths.append(path_info)

        if len(paths) >= max_paths:
            break

    return paths


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
# MCP tools are the interface exposed to AI assistants. Each tool has:
# - name: Unique identifier for the tool
# - description: What the tool does (shown to the AI)
# - inputSchema: JSON Schema defining the parameters

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available Vivado tools.

    This function is called by MCP clients to discover available tools.
    Tools are organized into categories:

    1. Session Management: start_session, stop_session, session_status
    2. Project Management: open_project, close_project, get_project_info
    3. Design Flow: run_synthesis, run_implementation, generate_bitstream
    4. Reports/Analysis: get_timing_summary, get_timing_paths, get_utilization, etc.
    5. Design Queries: get_design_hierarchy, get_ports, get_nets, get_cells
    6. Raw TCL: run_tcl for advanced operations
    7. Simulation: launch_simulation, run_simulation, get_signal_value, etc.
    8. Feature Requests: request_feature, list_feature_requests
    9. Report Management: generate_full_report, read_report_section

    Returns:
        List of Tool objects with name, description, and inputSchema
    """
    return [
        # =====================================================================
        # SESSION MANAGEMENT TOOLS
        # =====================================================================
        # These tools control the Vivado process lifecycle

        Tool(
            name="start_session",
            description="Start a persistent Vivado TCL session. Must be called before other commands.",
            inputSchema={
                "type": "object",
                "properties": {
                    "vivado_path": {
                        "type": "string",
                        "description": "Path to Vivado executable (default: 'vivado' from PATH)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="stop_session",
            description="Stop the Vivado TCL session and free resources",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="session_status",
            description="Get status and statistics of the current Vivado session",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="check_session_health",
            description="Check if Vivado session is responsive and recover if needed. Use this if commands are timing out or behaving unexpectedly.",
            inputSchema={
                "type": "object",
                "properties": {
                    "auto_recover": {
                        "type": "boolean",
                        "description": "Restart session if unhealthy (default: true)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_host_status",
            description="Get status of this Vivado MCP server host including hostname, free memory, and session state. If free memory is below 64GB, use vivado-snoke instead.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # =====================================================================
        # PROJECT MANAGEMENT TOOLS
        # =====================================================================
        # These tools work with Vivado project files (.xpr)

        Tool(
            name="open_project",
            description="Open a Vivado project (.xpr file)",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to .xpr project file"
                    }
                },
                "required": ["project_path"]
            }
        ),
        Tool(
            name="close_project",
            description="Close the current project",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_project_info",
            description="Get information about the currently open project",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # =====================================================================
        # DESIGN FLOW TOOLS
        # =====================================================================
        # These tools run the major FPGA design flow steps

        Tool(
            name="run_synthesis",
            description="Run synthesis on the current project",
            inputSchema={
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "integer",
                        "description": "Number of parallel jobs (default: 4)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 1800 = 30 minutes). Increase for large designs."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="run_implementation",
            description="Run implementation (place and route) on the current project",
            inputSchema={
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "integer",
                        "description": "Number of parallel jobs (default: 4)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 3600 = 60 minutes). Increase for large designs."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="generate_bitstream",
            description="Generate bitstream for the implemented design",
            inputSchema={
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "integer",
                        "description": "Number of parallel jobs (default: 4)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 1800 = 30 minutes). Increase for large designs."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="launch_run_async",
            description="Launch a Vivado run (synth_1, impl_1, etc.) without blocking on completion. Use get_run_progress afterwards to poll. Ideal for long impl/bitstream flows that would otherwise freeze the MCP session for 5-15 minutes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run": {
                        "type": "string",
                        "description": "Run name (e.g. 'synth_1', 'impl_1')"
                    },
                    "jobs": {
                        "type": "integer",
                        "description": "Parallel jobs (default 4)"
                    },
                    "to_step": {
                        "type": "string",
                        "description": "Optional stop-step within the run, e.g. 'write_bitstream' for impl_1 to include bitstream generation"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Reset and relaunch if the run is already complete or in progress (default false)"
                    }
                },
                "required": ["run"]
            }
        ),
        Tool(
            name="get_run_progress",
            description="Poll a Vivado run for its current STATUS and PROGRESS without blocking. Pair with launch_run_async for non-blocking impl/synth/bitstream flows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run": {
                        "type": "string",
                        "description": "Run name (e.g. 'synth_1', 'impl_1')"
                    }
                },
                "required": ["run"]
            }
        ),

        # =====================================================================
        # HARDWARE MANAGER TOOLS
        # =====================================================================
        # These tools drive Vivado's hardware manager to program physical
        # FPGAs over JTAG / USB. hw_server must be reachable (localhost:3121
        # by default). On Linux, hw_server is usually installed with Vivado
        # at <vivado>/bin/hw_server and can be launched with `hw_server`
        # from the terminal if it's not already running.

        Tool(
            name="list_hw_targets",
            description="Connect to hw_server and list physical JTAG targets + devices. Use this before programming to confirm the board is visible.",
            inputSchema={
                "type": "object",
                "properties": {
                    "hw_server_url": {
                        "type": "string",
                        "description": "hw_server URL (default: TCP:localhost:3121)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="program_device",
            description="Program a .bit file onto a physical FPGA via hw_server. This is the last mile: bitstream -> silicon. Safer than raw TCL because it verifies the bitfile exists and runs the full open_hw_manager/connect/program flow in one shot.",
            inputSchema={
                "type": "object",
                "properties": {
                    "bitfile": {
                        "type": "string",
                        "description": "Absolute path to the .bit file to program"
                    },
                    "hw_server_url": {
                        "type": "string",
                        "description": "hw_server URL (default: TCP:localhost:3121)"
                    },
                    "target_index": {
                        "type": "integer",
                        "description": "Index into get_hw_targets if multiple are connected (default 0)"
                    },
                    "device_index": {
                        "type": "integer",
                        "description": "Index of device within the chosen target (default 0 — typically the FPGA; SoCs may have multiple devices on the scan chain)"
                    }
                },
                "required": ["bitfile"]
            }
        ),
        Tool(
            name="close_hw_manager",
            description="Close the Vivado hardware manager and disconnect from hw_server. Frees the JTAG cable for other tools.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # =====================================================================
        # REPORTS AND ANALYSIS TOOLS
        # =====================================================================
        # These tools generate and parse Vivado's analysis reports

        Tool(
            name="get_timing_summary",
            description="Get timing summary (WNS, TNS, WHS, THS). Returns parsed metrics only by default. Use generate_full_report for raw output.",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "description": "Type: 'summary' (default), 'setup', 'hold', or 'all'"
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["summary", "standard", "full"],
                        "description": "Detail level: 'summary' (default, parsed metrics only), 'standard' (+ truncated raw), 'full' (+ complete raw)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_timing_paths",
            description="Get timing paths for failing or critical paths. Returns structured summary (slack, source, dest, clocks) by default. Use generate_full_report for verbose path details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_paths": {
                        "type": "integer",
                        "description": "Number of paths to report (default: 10)"
                    },
                    "slack_threshold": {
                        "type": "number",
                        "description": "Only show paths with slack less than this (default: 0 for failing paths)"
                    },
                    "path_type": {
                        "type": "string",
                        "description": "Type: 'setup' (default) or 'hold'"
                    },
                    "from_pin": {
                        "type": "string",
                        "description": "Filter paths starting from this pin/cell pattern (Vivado -from option)"
                    },
                    "to_pin": {
                        "type": "string",
                        "description": "Filter paths ending at this pin/cell pattern (Vivado -to option)"
                    },
                    "through": {
                        "type": "string",
                        "description": "Filter paths going through this pin/cell pattern (Vivado -through option)"
                    },
                    "clock": {
                        "type": "string",
                        "description": "Filter paths by clock domain name"
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["summary", "standard", "full"],
                        "description": "Detail level: 'summary' (default, structured only), 'standard' (+ truncated raw), 'full' (+ complete raw)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_utilization",
            description="Get resource utilization (LUT, FF, BRAM, DSP, IO). Returns parsed metrics only by default. Use generate_full_report for hierarchical details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "hierarchical": {
                        "type": "boolean",
                        "description": "Include hierarchical breakdown (default: false)"
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["summary", "standard", "full"],
                        "description": "Detail level: 'summary' (default, parsed only), 'standard' (+ truncated raw), 'full' (+ complete raw)"
                    },
                    "module_filter": {
                        "type": "string",
                        "description": "Wildcard pattern to filter modules in hierarchical report"
                    },
                    "threshold_percent": {
                        "type": "number",
                        "description": "Only show resources above this utilization percentage (0-100)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_clocks",
            description="Get clock information and constraints",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_messages",
            description="Get synthesis/implementation messages (errors, warnings) from a run's log file (runme.log). Use this to diagnose why a run failed. Returns the log path so read_report_section can fetch surrounding context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run": {
                        "type": "string",
                        "description": "Run whose log to read (default: 'synth_1'; use 'impl_1' for implementation)"
                    },
                    "severity": {
                        "type": "string",
                        "description": "Filter by severity: 'all' (default), 'error', 'critical', 'warning'"
                    },
                    "max_per_category": {
                        "type": "integer",
                        "description": "Cap messages returned per severity category (default: 50)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_drc_violations",
            description="Run report_drc and return structured violations: rule ID, severity, message, and affected objects. Replaces grepping through the DRC report text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "severity_filter": {
                        "type": "string",
                        "description": "Filter by severity. Comma-separated list of any of: 'error', 'critical_warning', 'warning', 'advisory', 'info'. Default 'all' returns everything."
                    },
                    "ruledecks": {
                        "type": "string",
                        "description": "Optional ruledeck name to restrict the check (e.g. 'default', 'methodology', 'timing'). Default runs the tool's standard DRC set."
                    },
                    "max_violations": {
                        "type": "integer",
                        "description": "Cap the number of violations returned (default 500). Prevents the response from exploding on designs with thousands of warnings."
                    }
                },
                "required": []
            }
        ),

        # =====================================================================
        # DESIGN QUERY TOOLS
        # =====================================================================
        # These tools explore the elaborated/synthesized design structure

        Tool(
            name="get_design_hierarchy",
            description="Get the design hierarchy (modules and instances)",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum hierarchy depth to return (default: 3)"
                    },
                    "instance_pattern": {
                        "type": "string",
                        "description": "Wildcard pattern to filter instances (e.g., '*cpu*', 'core/alu/*')"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_ports",
            description="Get top-level ports of the design",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_nets",
            description="Search for nets in the design",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Wildcard pattern to match net names (default: '*')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 100)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_cells",
            description="Search for cells (instances) in the design",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Wildcard pattern to match cell names (default: '*')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 100)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="check_constraints",
            description="Audit every top-level port for PACKAGE_PIN and IOSTANDARD assignments. Reports which ports are unconstrained — the condition that triggers UCIO-1 DRC and produces broken bitstreams when DRC is downgraded. Optionally generate a ready-to-edit XDC skeleton for the missing ports.",
            inputSchema={
                "type": "object",
                "properties": {
                    "generate_skeleton": {
                        "type": "boolean",
                        "description": "If true, include an XDC skeleton (as a string) for every unconstrained port. Defaults to false."
                    },
                    "skeleton_iostandard": {
                        "type": "string",
                        "description": "IOSTANDARD placeholder to emit in the skeleton (default 'LVCMOS18'). Users must still verify this matches their board's bank voltage."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Optional path. If set and generate_skeleton is true, the skeleton is also written to this file."
                    }
                },
                "required": []
            }
        ),

        # =====================================================================
        # RAW TCL TOOL
        # =====================================================================
        # Escape hatch for advanced operations not covered by specific tools

        Tool(
            name="run_tcl",
            description="Execute a raw TCL command in Vivado. Use for advanced operations not covered by other tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "TCL command to execute"
                    }
                },
                "required": ["command"]
            }
        ),

        # =====================================================================
        # SIMULATION TOOLS
        # =====================================================================
        # These tools control Vivado's integrated simulator (xsim)

        Tool(
            name="launch_simulation",
            description="Launch behavioral simulation (xsim). Opens the simulator and loads the design.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["behavioral", "post_synth_func", "post_synth_timing", "post_impl_func", "post_impl_timing"],
                        "description": "Simulation mode (default: behavioral)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="run_simulation",
            description="Run the simulation for a specified time",
            inputSchema={
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Time to run (e.g., '100ns', '1us', '10ms', 'all')"
                    }
                },
                "required": ["time"]
            }
        ),
        Tool(
            name="restart_simulation",
            description="Restart the simulation from time 0",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="close_simulation",
            description="Close the current simulation",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_simulation_time",
            description="Get the current simulation time",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_signal_value",
            description="Get the current value of a signal in simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "string",
                        "description": "Full hierarchical signal path (e.g., '/tb/dut/clk', '/tb/dut/data_out')"
                    },
                    "radix": {
                        "type": "string",
                        "enum": ["bin", "hex", "dec", "unsigned", "ascii"],
                        "description": "Display radix (default: hex)"
                    }
                },
                "required": ["signal"]
            }
        ),
        Tool(
            name="get_signal_values",
            description="Get current values of multiple signals matching a pattern",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Signal pattern with wildcards (e.g., '/tb/dut/*', '/tb/dut/data*')"
                    },
                    "radix": {
                        "type": "string",
                        "enum": ["bin", "hex", "dec", "unsigned", "ascii"],
                        "description": "Display radix (default: hex)"
                    }
                },
                "required": ["pattern"]
            }
        ),
        Tool(
            name="add_signals_to_wave",
            description="Add signals to the waveform viewer",
            inputSchema={
                "type": "object",
                "properties": {
                    "signals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of signal paths to add (e.g., ['/tb/dut/clk', '/tb/dut/rst'])"
                    }
                },
                "required": ["signals"]
            }
        ),
        Tool(
            name="set_simulation_top",
            description="Set the top module for simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "top_module": {
                        "type": "string",
                        "description": "Name of the testbench module"
                    },
                    "fileset": {
                        "type": "string",
                        "description": "Simulation fileset (default: sim_1)"
                    }
                },
                "required": ["top_module"]
            }
        ),
        Tool(
            name="get_simulation_objects",
            description="List simulation objects (signals, variables) in a scope",
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Hierarchical scope (e.g., '/tb', '/tb/dut'). Default is root."
                    },
                    "filter": {
                        "type": "string",
                        "enum": ["all", "signals", "ports", "internal"],
                        "description": "Filter by object type (default: all)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_scopes",
            description="List available scopes (hierarchy) in the simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent": {
                        "type": "string",
                        "description": "Parent scope to list children of (default: root)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="step_simulation",
            description="Step the simulation by a delta cycle or time step",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of steps (default: 1)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="add_breakpoint",
            description="Add a simulation breakpoint on a signal condition",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "string",
                        "description": "Signal to monitor"
                    },
                    "condition": {
                        "type": "string",
                        "enum": ["posedge", "negedge", "change"],
                        "description": "Trigger condition (default: change)"
                    }
                },
                "required": ["signal"]
            }
        ),
        Tool(
            name="remove_breakpoints",
            description="Remove all breakpoints",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_simulation_messages",
            description="Get simulation log messages (errors, warnings, info)",
            inputSchema={
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["all", "error", "warning", "info"],
                        "description": "Filter by severity (default: all)"
                    }
                },
                "required": []
            }
        ),

        # =====================================================================
        # FEATURE REQUEST TOOLS
        # =====================================================================
        # Allow AI assistants to request new features

        Tool(
            name="request_feature",
            description="Request a new feature or capability for the Vivado MCP server. Use this when you encounter a limitation or wish you had a tool that doesn't exist.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title for the feature request"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of what you need and why"
                    },
                    "use_case": {
                        "type": "string",
                        "description": "The specific use case or task you were trying to accomplish"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "How important is this feature? (default: medium)"
                    }
                },
                "required": ["title", "description"]
            }
        ),
        Tool(
            name="list_feature_requests",
            description="List all feature requests that have been submitted",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # =====================================================================
        # REPORT FILE MANAGEMENT TOOLS
        # =====================================================================
        # Handle large reports that exceed inline response limits

        Tool(
            name="generate_full_report",
            description="Generate a full Vivado report to a file. Use when inline reports are truncated or you need the complete output.",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "enum": ["timing", "timing_summary", "utilization", "hierarchy", "clocks", "power", "drc"],
                        "description": "Type of report to generate"
                    },
                    "options": {
                        "type": "object",
                        "description": "Report-specific options (e.g., {'hierarchical': true} for utilization)"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Optional custom output path. Default: /tmp/vivado_mcp/<type>_<id>.txt"
                    }
                },
                "required": ["report_type"]
            }
        ),
        Tool(
            name="read_report_section",
            description="Read a section of a previously generated report file",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_id": {
                        "type": "string",
                        "description": "Report ID returned by generate_full_report"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Alternative: direct file path to read"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed, default: 1)"
                    },
                    "num_lines": {
                        "type": "integer",
                        "description": "Number of lines to read (default: 100)"
                    },
                    "search_pattern": {
                        "type": "string",
                        "description": "Regex pattern to find a section (returns lines around first match)"
                    }
                },
                "required": []
            }
        )
    ]


# =============================================================================
# TOOL IMPLEMENTATION
# =============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Handle tool calls from MCP clients.

    Dispatches to the synchronous implementation in a worker thread so the
    asyncio event loop stays responsive while long Vivado commands (e.g. a
    30-minute synthesis) block on pexpect. Without this, a single blocking
    call would freeze the whole MCP server — no pings, no other requests.
    """
    return await asyncio.to_thread(_call_tool_sync, name, arguments)


def _call_tool_sync(name: str, arguments: dict) -> list[TextContent]:
    """
    Synchronous tool dispatcher.

    This is the main dispatcher that routes tool calls to their implementations.
    Each tool returns a list containing a single TextContent with JSON-formatted
    results.

    Args:
        name: The tool name being called
        arguments: Dictionary of arguments passed to the tool

    Returns:
        List containing one TextContent with JSON response

    Response format:
        All tools return JSON with at minimum:
        - success: Boolean indicating if the operation succeeded
        - Additional fields specific to each tool

        On error:
        - error: Error message string
        - success: False
    """
    # Get the singleton Vivado session
    session = get_session()

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    if name == "start_session":
        # Start Vivado TCL session
        # This spawns a persistent Vivado process that stays running
        vivado_path = arguments.get("vivado_path", "vivado")
        session.vivado_path = vivado_path
        result = session.start()
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "message": result.output,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "stop_session":
        # Stop Vivado session gracefully
        result = session.stop()
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "message": result.output
        }, indent=2))]

    elif name == "session_status":
        # Get session statistics (commands run, errors, timing, etc.)
        stats = session.get_stats()
        return [TextContent(type="text", text=json.dumps(stats, indent=2))]

    elif name == "check_session_health":
        # Check if session is responsive and optionally recover
        auto_recover = arguments.get("auto_recover", True)

        if not session.is_running:
            if auto_recover:
                result = session.start()
                return [TextContent(type="text", text=json.dumps({
                    "healthy": result.success,
                    "action": "started",
                    "message": "Session was not running, started new session",
                    "elapsed_ms": result.elapsed_ms
                }, indent=2))]
            else:
                return [TextContent(type="text", text=json.dumps({
                    "healthy": False,
                    "action": "none",
                    "message": "Session not running (auto_recover=false)"
                }, indent=2))]

        # Session thinks it's running, check if actually responsive
        is_healthy = session.is_healthy()

        if is_healthy:
            return [TextContent(type="text", text=json.dumps({
                "healthy": True,
                "action": "none",
                "message": "Session is healthy and responsive"
            }, indent=2))]

        # Session is unresponsive
        if auto_recover:
            result = session.ensure_healthy()
            return [TextContent(type="text", text=json.dumps({
                "healthy": result.success,
                "action": "restarted",
                "message": "Session was unresponsive, restarted",
                "elapsed_ms": result.elapsed_ms
            }, indent=2))]
        else:
            return [TextContent(type="text", text=json.dumps({
                "healthy": False,
                "action": "none",
                "message": "Session is unresponsive (auto_recover=false)"
            }, indent=2))]

    elif name == "get_host_status":
        # Get host system status for memory-based server selection
        import psutil

        hostname = socket.gethostname()
        mem = psutil.virtual_memory()
        mem_free_gb = mem.available / (1024 ** 3)
        mem_total_gb = mem.total / (1024 ** 3)

        # Build suggestion based on free memory (64GB threshold)
        suggestion = None
        if mem_free_gb < 64:
            suggestion = f"Low memory ({mem_free_gb:.1f}GB free). Use vivado-snoke instead."

        return [TextContent(type="text", text=json.dumps({
            "hostname": hostname,
            "memory_free_gb": round(mem_free_gb, 1),
            "memory_total_gb": round(mem_total_gb, 1),
            "memory_percent_used": mem.percent,
            "vivado_session_active": session.is_running,
            "suggestion": suggestion
        }, indent=2))]

    # =========================================================================
    # SESSION CHECK
    # =========================================================================
    # All remaining commands require an active Vivado session

    if not session.is_running:
        return [TextContent(type="text", text=json.dumps({
            "error": "Vivado session not running. Call start_session first.",
            "success": False
        }, indent=2))]

    # =========================================================================
    # PROJECT MANAGEMENT
    # =========================================================================

    if name == "open_project":
        # Open a Vivado project file (.xpr)
        project_path = arguments.get("project_path", "")
        # Use braces to handle paths with spaces
        result = session.run_tcl(f"open_project {{{project_path}}}")
        if result.success:
            session.current_project = project_path
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "output": result.output,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "close_project":
        # Close the current project
        result = session.run_tcl("close_project")
        session.current_project = None
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "output": result.output
        }, indent=2))]

    elif name == "get_project_info":
        # Get various project properties
        commands = [
            "current_project",                                    # Project name
            "get_property PART [current_project]",               # Target FPGA part
            "get_property TARGET_LANGUAGE [current_project]",    # Verilog/VHDL
            "get_property DIRECTORY [current_project]"           # Project directory
        ]
        results = {}
        for cmd in commands:
            r = session.run_tcl(cmd)
            results[cmd] = r.output
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    # =========================================================================
    # DESIGN FLOW
    # =========================================================================

    elif name == "run_synthesis":
        # Run synthesis with optional parallel jobs
        # reset_run clears previous results, launch_runs starts synthesis,
        # wait_on_run blocks until complete
        jobs = arguments.get("jobs", 4)
        timeout = arguments.get("timeout", 1800)  # 30 min default

        result = session.run_tcl(
            f"reset_run synth_1; launch_runs synth_1 -jobs {jobs}; wait_on_run synth_1",
            timeout_override=timeout
        )

        # Verify actual run status (more reliable than output parsing)
        verification = verify_run_status(session, "synth_1")
        actual_success = verification["actually_succeeded"]

        response = {
            "success": actual_success,
            "output": result.output,
            "elapsed_ms": result.elapsed_ms,
            "run_status": verification["status"],
            "run_progress": verification["progress"],
        }

        # Note if there was a mismatch between output parsing and actual status
        if not result.success and actual_success:
            response["note"] = "Output contained error-like strings but run completed successfully"

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    elif name == "run_implementation":
        # Run place and route
        # reset_run is needed to re-launch a previously completed impl_1;
        # wrapped in catch so a never-launched run doesn't abort the chain
        jobs = arguments.get("jobs", 4)
        timeout = arguments.get("timeout", 3600)  # 60 min default

        result = session.run_tcl(
            f"catch {{reset_run impl_1}}; launch_runs impl_1 -jobs {jobs}; wait_on_run impl_1",
            timeout_override=timeout
        )

        # Verify actual run status (more reliable than output parsing)
        verification = verify_run_status(session, "impl_1")
        actual_success = verification["actually_succeeded"]

        response = {
            "success": actual_success,
            "output": result.output,
            "elapsed_ms": result.elapsed_ms,
            "run_status": verification["status"],
            "run_progress": verification["progress"],
        }

        # Note if there was a mismatch between output parsing and actual status
        if not result.success and actual_success:
            response["note"] = "Output contained error-like strings but run completed successfully"

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    elif name == "generate_bitstream":
        # Generate bitstream (programming file)
        # If impl_1 already completed route_design, launch_runs resumes from
        # write_bitstream. The launch is wrapped in catch so an impl_1 that
        # already wrote its bitstream (relaunch error) still falls through to
        # the status check, which then reports success.
        jobs = arguments.get("jobs", 4)
        timeout = arguments.get("timeout", 1800)  # 30 min default

        result = session.run_tcl(
            f"catch {{launch_runs impl_1 -to_step write_bitstream -jobs {jobs}}}; "
            "catch {wait_on_run impl_1}",
            timeout_override=timeout
        )

        # Verify actual run status; bitstream success means the run reached
        # the write_bitstream step, not just any "Complete!" state
        verification = verify_run_status(session, "impl_1")
        actual_success = (
            verification["actually_succeeded"]
            and "write_bitstream" in verification["status"].lower()
        )

        return [TextContent(type="text", text=json.dumps({
            "success": actual_success,
            "output": result.output,
            "elapsed_ms": result.elapsed_ms,
            "run_status": verification["status"],
            "run_progress": verification["progress"],
        }, indent=2))]

    elif name == "launch_run_async":
        # Launch a run without blocking on completion. Vivado's launch_runs
        # spawns the actual synth/impl as a subprocess and returns quickly
        # (usually under 5s); the blocking part is the wait_on_run that
        # run_implementation / run_synthesis normally chain afterwards. By
        # skipping wait_on_run we hand the session back immediately and let
        # the caller poll get_run_progress.
        run_name = arguments.get("run", "").strip()
        if not run_name:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "Missing 'run' argument (e.g. 'synth_1', 'impl_1')"
            }, indent=2))]

        jobs = int(arguments.get("jobs", 4))
        to_step = arguments.get("to_step", "").strip()
        force = bool(arguments.get("force", False))

        cmd_parts = []
        if force:
            # reset_runs returns an error if the run hasn't been created yet;
            # wrap in catch so a fresh project doesn't fail this step.
            cmd_parts.append(f"catch {{reset_run {run_name}}}")
        launch = f"launch_runs {run_name} -jobs {jobs}"
        if to_step:
            launch += f" -to_step {to_step}"
        cmd_parts.append(launch)
        # Immediately query status so the response can tell the caller what
        # state the run is in right after launch (usually "Queued" or
        # "Running synth_design").
        cmd_parts.append(
            f'puts stdout "LAUNCH_STATUS=[get_property STATUS [get_runs {run_name}]]"'
        )
        cmd_parts.append("flush stdout")

        result = session.run_tcl("; ".join(cmd_parts), timeout_override=60)

        # Parse the trailing status line
        launch_status = ""
        for line in result.output.splitlines():
            if line.startswith("LAUNCH_STATUS="):
                launch_status = line.split("=", 1)[1].strip()
                break

        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "run": run_name,
            "jobs": jobs,
            "to_step": to_step or None,
            "launch_status": launch_status,
            "hint": "Call get_run_progress repeatedly (every 10-30s) to track progress without blocking.",
            "output": result.output,
            "elapsed_ms": result.elapsed_ms,
        }, indent=2))]

    elif name == "get_run_progress":
        # Non-blocking progress query. Safe to call at any time; Vivado just
        # reads the run object's properties without affecting the background
        # subprocess doing the actual work.
        run_name = arguments.get("run", "").strip()
        if not run_name:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "Missing 'run' argument"
            }, indent=2))]

        info = verify_run_status(session, run_name)
        status_lower = info.get("status", "").lower()
        finished = (
            info.get("actually_succeeded", False)
            or info.get("actually_failed", False)
            or status_lower.endswith("complete!")
        )

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "run": run_name,
            "status": info.get("status"),
            "progress": info.get("progress"),
            "finished": finished,
            "succeeded": info.get("actually_succeeded", False),
            "failed": info.get("actually_failed", False),
        }, indent=2))]

    # =========================================================================
    # HARDWARE MANAGER
    # =========================================================================

    elif name == "list_hw_targets":
        # Enumerate physical JTAG targets reachable via hw_server. This opens
        # the hardware manager if it isn't already open and connects to the
        # server URL. Safe to call multiple times — Vivado is idempotent here.
        hw_url = arguments.get("hw_server_url", "TCP:localhost:3121")

        # Bail fast if hw_server isn't reachable — Vivado's connect_hw_server
        # will otherwise hang the entire Tcl session for ~30s and poison
        # subsequent commands.
        if not _hw_server_reachable(hw_url):
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "hw_server_url": hw_url,
                "error": "hw_server not reachable",
                "hint": "Start hw_server (`<vivado>/bin/hw_server &`) and verify a JTAG cable is connected.",
                "targets_found": 0,
                "targets": [],
            }, indent=2))]

        # Run open + connect defensively. If hw_manager is already open or the
        # connection already exists, Vivado prints a warning but keeps going.
        # Wrap each step in catch so one prior state doesn't abort the whole
        # query.
        probe = (
            "catch {open_hw_manager}; "
            f"catch {{connect_hw_server -url {hw_url}}}; "
            "set __mcp_rows__ [list]; "
            "foreach __mcp_t__ [get_hw_targets] { "
            "  set __mcp_devs__ [list]; "
            "  catch {current_hw_target $__mcp_t__; open_hw_target -quiet}; "
            "  foreach __mcp_d__ [get_hw_devices] { "
            "    lappend __mcp_devs__ \"[get_property PART $__mcp_d__]\""
            "  }; "
            "  lappend __mcp_rows__ \"$__mcp_t__||[join $__mcp_devs__ ,]\" "
            "}; "
            "puts stdout [join $__mcp_rows__ \"\\n\"]; "
            "flush stdout"
        )
        result = session.run_tcl(probe, timeout_override=30)

        targets = []
        if result.success and result.output.strip():
            for line in result.output.splitlines():
                line = line.strip()
                if "||" not in line:
                    continue
                tname, devs = line.split("||", 1)
                targets.append({
                    "target": tname,
                    "devices": [d for d in devs.split(",") if d],
                })

        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "hw_server_url": hw_url,
            "targets_found": len(targets),
            "targets": targets,
            "raw": result.output if not targets else None,
            "elapsed_ms": result.elapsed_ms,
        }, indent=2))]

    elif name == "program_device":
        # Program a .bit file onto a physical FPGA. This is the terminal
        # step that takes a verified bitstream and puts it on silicon. We
        # verify the file exists up front so the agent doesn't spend 5s in
        # Vivado only to get "file not found" at the last step.
        bitfile = arguments.get("bitfile", "")
        if not bitfile or not os.path.isfile(bitfile):
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Bitfile not found: {bitfile}",
                "hint": "Pass an absolute path to an existing .bit file."
            }, indent=2))]

        hw_url = arguments.get("hw_server_url", "TCP:localhost:3121")
        target_idx = int(arguments.get("target_index", 0))
        device_idx = int(arguments.get("device_index", 0))

        # Fail fast if hw_server isn't accepting connections. Without this
        # check Vivado's connect_hw_server hangs the session for ~30s and
        # poisons follow-up commands.
        if not _hw_server_reachable(hw_url):
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "bitfile": bitfile,
                "hw_server_url": hw_url,
                "error": "hw_server not reachable",
                "hint": "Start hw_server (`<vivado>/bin/hw_server &`) and verify a JTAG cable is connected.",
            }, indent=2))]

        # Run the whole program flow in one TCL shot so any intermediate
        # failure (cable disconnected, multiple boards, wrong device) is
        # caught by a single round-trip. Uses catch to capture the error
        # message but still report structured output.
        flow = (
            "catch {open_hw_manager}; "
            f"catch {{connect_hw_server -url {hw_url}}}; "
            "set __mcp_err__ \"\"; "
            "if {[catch { "
            f"  set __mcp_tgts__ [get_hw_targets]; "
            f"  if {{[llength $__mcp_tgts__] == 0}} {{error \"no hw_targets visible — is the cable connected and hw_server running?\"}}; "
            f"  current_hw_target [lindex $__mcp_tgts__ {target_idx}]; "
            "  catch {open_hw_target}; "
            f"  set __mcp_devs__ [get_hw_devices]; "
            f"  if {{[llength $__mcp_devs__] == 0}} {{error \"no hw_devices on target\"}}; "
            f"  set __mcp_dev__ [lindex $__mcp_devs__ {device_idx}]; "
            f"  current_hw_device $__mcp_dev__; "
            f"  set_property PROGRAM.FILE {{{bitfile}}} $__mcp_dev__; "
            "  program_hw_devices $__mcp_dev__; "
            "  refresh_hw_device $__mcp_dev__; "
            "  puts stdout \"PROGRAMMED $__mcp_dev__\"; "
            "  flush stdout "
            "} __mcp_err__]} { "
            "  puts stdout \"FAILED: $__mcp_err__\"; flush stdout "
            "}"
        )
        result = session.run_tcl(flow, timeout_override=120)

        programmed = "PROGRAMMED" in result.output and "FAILED" not in result.output
        return [TextContent(type="text", text=json.dumps({
            "success": programmed,
            "bitfile": bitfile,
            "hw_server_url": hw_url,
            "target_index": target_idx,
            "device_index": device_idx,
            "output": result.output,
            "elapsed_ms": result.elapsed_ms,
        }, indent=2))]

    elif name == "close_hw_manager":
        # Release the JTAG cable and close the hardware manager so other
        # tools (e.g. openocd, openFPGALoader, a second Vivado) can use it.
        result = session.run_tcl("catch {close_hw_target}; catch {disconnect_hw_server}; catch {close_hw_manager}; puts stdout CLOSED; flush stdout")
        return [TextContent(type="text", text=json.dumps({
            "success": "CLOSED" in result.output,
            "output": result.output,
            "elapsed_ms": result.elapsed_ms,
        }, indent=2))]

    # =========================================================================
    # REPORTS AND ANALYSIS
    # =========================================================================

    elif name == "get_timing_summary":
        # Get timing summary with parsed metrics
        report_type = arguments.get("report_type", "summary")
        detail_level = arguments.get("detail_level", "summary")

        # Run Vivado timing summary report
        result = session.run_tcl("report_timing_summary -no_header -return_string")

        # Parse the raw output into structured data
        parsed = parse_timing_summary(result.output)
        parsed["success"] = result.success
        parsed["elapsed_ms"] = result.elapsed_ms

        # Control output verbosity based on detail_level
        if detail_level == "summary":
            # Only return parsed metrics, no raw output
            parsed.pop("raw", None)
        elif detail_level == "standard":
            # Truncate raw output if too large (half of max to leave room for other data)
            if "raw" in parsed and len(parsed["raw"]) > MAX_RESPONSE_CHARS // 2:
                truncated = truncate_response(parsed["raw"], MAX_RESPONSE_CHARS // 2)
                parsed["raw"] = truncated["content"]
                if truncated["truncated"]:
                    parsed["raw_truncated"] = True
                    parsed["raw_total_chars"] = truncated["total_chars"]
        elif detail_level == "full":
            # Keep complete raw output but apply safety truncation
            if "raw" in parsed:
                truncated = truncate_response(parsed["raw"], MAX_RESPONSE_CHARS)
                parsed["raw"] = truncated["content"]
                if truncated["truncated"]:
                    parsed["raw_truncated"] = True
                    parsed["raw_total_chars"] = truncated["total_chars"]
                    parsed["truncation_message"] = truncated["truncation_message"]

        return [TextContent(type="text", text=json.dumps(parsed, indent=2))]

    elif name == "get_timing_paths":
        # Get detailed timing path information
        # Useful for debugging timing violations
        num_paths = arguments.get("num_paths", 10)
        slack_threshold = arguments.get("slack_threshold", 0)  # 0 = failing paths only
        path_type = arguments.get("path_type", "setup")
        from_pin = arguments.get("from_pin")
        to_pin = arguments.get("to_pin")
        through = arguments.get("through")
        clock = arguments.get("clock")
        detail_level = arguments.get("detail_level", "summary")

        # Build the report_timing command
        delay_type = "max" if path_type == "setup" else "min"
        cmd = f"report_timing -delay_type {delay_type} -max_paths {num_paths} -slack_lesser_than {slack_threshold}"

        # Add optional path filters
        if from_pin:
            cmd += f" -from {{{from_pin}}}"
        if to_pin:
            cmd += f" -to {{{to_pin}}}"
        if through:
            cmd += f" -through {{{through}}}"
        if clock:
            # report_timing has no -filter option; each clock forms a path
            # group of the same name, so -group restricts to that domain
            cmd += f" -group {{{clock}}}"

        cmd += " -return_string"
        result = session.run_tcl(cmd)

        # Build response with filter information
        response = {
            "success": result.success,
            "elapsed_ms": result.elapsed_ms,
            "filters_applied": {
                "path_type": path_type,
                "num_paths": num_paths,
                "slack_threshold": slack_threshold
            }
        }

        # Include any filters that were used
        if from_pin:
            response["filters_applied"]["from_pin"] = from_pin
        if to_pin:
            response["filters_applied"]["to_pin"] = to_pin
        if through:
            response["filters_applied"]["through"] = through
        if clock:
            response["filters_applied"]["clock"] = clock

        # Handle output based on detail level
        if result.success:
            # Always parse paths into structured format
            parsed_paths = parse_timing_paths_summary(result.output, max_paths=num_paths)
            response["paths"] = parsed_paths
            response["path_count"] = len(parsed_paths)

            if detail_level == "summary":
                # Only return structured data, no raw output
                pass
            elif detail_level == "standard":
                # Include truncated raw for reference
                truncated = truncate_response(result.output, MAX_RESPONSE_CHARS // 2)
                response["raw"] = truncated["content"]
                if truncated["truncated"]:
                    response["raw_truncated"] = True
                    response["raw_total_chars"] = truncated["total_chars"]
            elif detail_level == "full":
                # Include complete raw output
                truncated = truncate_response(result.output, MAX_RESPONSE_CHARS)
                response["raw"] = truncated["content"]
                if truncated["truncated"]:
                    response["raw_truncated"] = True
                    response["raw_total_chars"] = truncated["total_chars"]
                    response["truncation_message"] = truncated["truncation_message"]
        else:
            response["error"] = result.output

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    elif name == "get_utilization":
        # Get resource utilization with parsed metrics
        hierarchical = arguments.get("hierarchical", False)
        detail_level = arguments.get("detail_level", "summary")
        module_filter = arguments.get("module_filter")
        threshold_percent = arguments.get("threshold_percent")

        # Build utilization report command
        cmd = "report_utilization -return_string"
        if hierarchical:
            cmd += " -hierarchical"
            if module_filter:
                cmd += f" -hierarchical_pattern {{{module_filter}}}"

        result = session.run_tcl(cmd)

        # Parse into structured data
        parsed = parse_utilization(result.output)
        parsed["success"] = result.success
        parsed["elapsed_ms"] = result.elapsed_ms

        # Apply threshold filter if specified
        if threshold_percent is not None:
            for resource in ["lut", "ff", "bram", "dsp", "io"]:
                if resource in parsed and parsed[resource]["percent"] < threshold_percent:
                    parsed[resource]["below_threshold"] = True

        # Control output verbosity
        if detail_level == "summary":
            parsed.pop("raw", None)
        elif detail_level == "standard":
            if "raw" in parsed and len(parsed["raw"]) > MAX_RESPONSE_CHARS // 2:
                truncated = truncate_response(parsed["raw"], MAX_RESPONSE_CHARS // 2)
                parsed["raw"] = truncated["content"]
                if truncated["truncated"]:
                    parsed["raw_truncated"] = True
                    parsed["raw_total_chars"] = truncated["total_chars"]
        elif detail_level == "full":
            if "raw" in parsed:
                truncated = truncate_response(parsed["raw"], MAX_RESPONSE_CHARS)
                parsed["raw"] = truncated["content"]
                if truncated["truncated"]:
                    parsed["raw_truncated"] = True
                    parsed["raw_total_chars"] = truncated["total_chars"]
                    parsed["truncation_message"] = truncated["truncation_message"]

        return [TextContent(type="text", text=json.dumps(parsed, indent=2))]

    elif name == "get_clocks":
        # Get clock information from the design
        result = session.run_tcl("report_clocks -return_string")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "clocks": result.output,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "get_messages":
        # Get synthesis/implementation messages from the run's log file.
        # There is no TCL command that dumps the message log in -mode tcl
        # (get_msg_config only manages message *configuration rules*), so we
        # read runme.log from the run directory instead — that's where
        # launch_runs writes all ERROR/WARNING/INFO lines.
        run_name = arguments.get("run", "synth_1")
        severity = arguments.get("severity", "all")
        max_per_category = arguments.get("max_per_category", 50)

        dir_result = session.run_tcl(f"get_property DIRECTORY [get_runs {{{run_name}}}]")
        if not dir_result.success or not dir_result.output.strip():
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Could not resolve directory for run '{run_name}': {dir_result.output}"
            }, indent=2))]

        log_file = Path(dir_result.output.strip().splitlines()[-1]) / "runme.log"
        if not log_file.exists():
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Log file not found: {log_file}. Has the run been launched?",
                "run": run_name
            }, indent=2))]

        parsed = parse_messages(log_file.read_text(errors="replace"))
        parsed.pop("raw", None)

        counts = {
            "errors": len(parsed["errors"]),
            "critical_warnings": len(parsed["critical_warnings"]),
            "warnings": len(parsed["warnings"]),
            "info": len(parsed["info"])
        }

        # Apply severity filter, then cap each category to keep responses small
        if severity != "all":
            key = {
                "error": "errors",
                "critical": "critical_warnings",
                "warning": "warnings"
            }.get(severity)
            parsed = {key: parsed[key]} if key else {}
        for key in list(parsed.keys()):
            if len(parsed[key]) > max_per_category:
                parsed[key] = parsed[key][:max_per_category]
                parsed[f"{key}_truncated"] = True

        parsed["success"] = True
        parsed["run"] = run_name
        parsed["counts"] = counts
        parsed["log_file"] = str(log_file)
        parsed["hint"] = "Use read_report_section with file_path to read the log around a specific message."
        return [TextContent(type="text", text=json.dumps(parsed, indent=2))]

    elif name == "get_drc_violations":
        # Run report_drc and return structured violations. Agents routinely
        # grep through the DRC report text for specific rule IDs; this tool
        # surfaces that data as JSON so it can be filtered and counted
        # without parsing.
        severity_filter = arguments.get("severity_filter", "all").lower().strip()
        ruledecks = arguments.get("ruledecks", "").strip()
        max_viol = int(arguments.get("max_violations", 500))

        # Build the report_drc command. Using -name lets us reference the
        # result set, but the violation objects live directly under
        # get_drc_violations after report_drc runs.
        drc_cmd_parts = ["catch {report_drc"]
        if ruledecks:
            drc_cmd_parts.append(f"-ruledecks {ruledecks}")
        drc_cmd_parts.append("-quiet}")
        drc_cmd = " ".join(drc_cmd_parts)

        # After DRC runs, enumerate violation objects. Escape newlines and
        # pipes in messages so we can safely parse on the Python side.
        # drc_violation objects expose NAME (rule name), SEVERITY, DESCRIPTION
        # (the free-form text), CHECK (check id), CLASS (ruledeck), and ID.
        # Older docs and other Vivado object types use MESSAGE, but
        # drc_violation uses DESCRIPTION.
        probe = (
            f"{drc_cmd}; "
            "set __mcp_rows__ [list]; "
            "foreach __mcp_v__ [get_drc_violations] { "
            "  set __mcp_nm__ [get_property NAME $__mcp_v__]; "
            "  set __mcp_sv__ [get_property SEVERITY $__mcp_v__]; "
            "  set __mcp_msg__ [get_property DESCRIPTION $__mcp_v__]; "
            "  set __mcp_rule__ \"\"; "
            "  catch {set __mcp_rule__ [get_property CLASS $__mcp_v__]}; "
            "  set __mcp_msg__ [string map {\"\\n\" {\\n} | {\\|}} $__mcp_msg__]; "
            "  set __mcp_sv__ [string map {| {\\|}} $__mcp_sv__]; "
            "  lappend __mcp_rows__ \"$__mcp_nm__|$__mcp_sv__|$__mcp_rule__|$__mcp_msg__\" "
            "}; "
            "puts stdout [join $__mcp_rows__ \"\\n\"]; "
            "flush stdout"
        )
        result = session.run_tcl(probe, timeout_override=120)

        if not result.success:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "DRC command failed — is a synthesized or opened design in memory?",
                "output": result.output,
                "elapsed_ms": result.elapsed_ms,
            }, indent=2))]

        violations = []
        counts_by_severity = {}
        allowed = None
        if severity_filter != "all":
            allowed = {s.strip() for s in severity_filter.split(",") if s.strip()}

        def _norm_sev(sev: str) -> str:
            return sev.lower().replace(" ", "_")

        for line in result.output.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            # Rejoin escaped pipes before splitting
            # The Tcl string map replaced real `|` with `\|`, so split on the
            # literal `|` and then unescape each field.
            raw_fields = line.split("|")
            # The message may still contain \| markers from the escape — we
            # splitted too eagerly. Fix: greedy rejoin on first three
            # unescaped separators; everything else is the message.
            # Walk the fields and merge any that ended with an odd number of
            # backslashes (escape was trailing).
            merged = []
            buf = ""
            for f in raw_fields:
                if buf:
                    buf = buf + "|" + f
                else:
                    buf = f
                # Count trailing backslashes — odd means escaped pipe
                trailing = len(buf) - len(buf.rstrip("\\"))
                if trailing % 2 == 0:
                    merged.append(buf)
                    buf = ""
            if buf:
                merged.append(buf)
            fields = merged

            if len(fields) < 4:
                continue
            name_f, sev_f, rule_f = fields[0], fields[1], fields[2]
            msg_f = "|".join(fields[3:])
            # Unescape the \n, \| placeholders
            msg_f = msg_f.replace("\\|", "|").replace("\\n", "\n")

            norm_sev = _norm_sev(sev_f)
            counts_by_severity[norm_sev] = counts_by_severity.get(norm_sev, 0) + 1

            if allowed is not None and norm_sev not in allowed:
                continue
            if len(violations) >= max_viol:
                continue

            violations.append({
                "rule": name_f,
                "severity": sev_f,
                "ruledeck": rule_f,
                "message": msg_f,
            })

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "total_violations": sum(counts_by_severity.values()),
            "by_severity": counts_by_severity,
            "returned": len(violations),
            "truncated": sum(counts_by_severity.values()) > len(violations) and allowed is None,
            "severity_filter": severity_filter,
            "violations": violations,
            "elapsed_ms": result.elapsed_ms,
        }, indent=2))]

    # =========================================================================
    # DESIGN QUERIES
    # =========================================================================

    elif name == "get_design_hierarchy":
        # Get the design hierarchy (instances and modules)
        max_depth = arguments.get("max_depth", 3)
        instance_pattern = arguments.get("instance_pattern", "*")

        # Get all hierarchical cells matching the pattern
        cmd = f"get_cells -hierarchical {{{instance_pattern}}}"
        result = session.run_tcl(cmd)

        if result.success and result.output.strip():
            cells = result.output.strip().split()

            # Filter by hierarchy depth (count '/' separators)
            filtered_cells = []
            for cell in cells:
                depth = get_hierarchy_depth(cell)
                if depth <= max_depth:
                    filtered_cells.append(cell)

            # Get module reference for each cell in a single TCL round-trip
            # (limited to 100 cells for response size)
            cell_refs = {}
            sample_cells = filtered_cells[:100]
            if sample_cells:
                cell_list = " ".join(f"{{{c}}}" for c in sample_cells)
                probe = (
                    "set __mcp_refs__ [list]; "
                    f"foreach __mcp_c__ [list {cell_list}] {{ "
                    "catch {lappend __mcp_refs__ \"$__mcp_c__|[get_property REF_NAME [get_cells $__mcp_c__]]\"} "
                    "}; "
                    'puts stdout [join $__mcp_refs__ "\\n"]; flush stdout'
                )
                ref_result = session.run_tcl(probe)
                if ref_result.success:
                    for line in ref_result.output.splitlines():
                        if "|" in line:
                            cell, ref = line.strip().split("|", 1)
                            if ref:
                                cell_refs[cell] = ref

            response = {
                "success": True,
                "cells": filtered_cells[:500],  # Limit for response size
                "cell_count": len(filtered_cells),
                "cell_modules": cell_refs,
                "max_depth": max_depth,
                "elapsed_ms": result.elapsed_ms
            }

            if len(filtered_cells) > 500:
                response["truncated"] = True
                response["total_cells"] = len(filtered_cells)
                response["message"] = "Cell list truncated. Use instance_pattern to filter or generate_full_report for complete hierarchy."
        else:
            response = {
                "success": result.success,
                "cells": [],
                "cell_count": 0,
                "error": result.output if not result.success else "No cells found",
                "elapsed_ms": result.elapsed_ms
            }

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    elif name == "get_ports":
        # Get top-level I/O ports
        result = session.run_tcl("get_ports *")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "ports": result.output.split() if result.success else [],
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "get_nets":
        # Search for nets by pattern
        pattern = arguments.get("pattern", "*")
        limit = arguments.get("limit", 100)
        # Use lrange to limit results
        result = session.run_tcl(f"lrange [get_nets {{{pattern}}}] 0 {limit-1}")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "nets": result.output.split() if result.success else [],
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "get_cells":
        # Search for cells/instances by pattern
        pattern = arguments.get("pattern", "*")
        limit = arguments.get("limit", 100)
        result = session.run_tcl(f"lrange [get_cells {{{pattern}}}] 0 {limit-1}")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "cells": result.output.split() if result.success else [],
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "check_constraints":
        # Audit every top-level port for physical pin + iostandard constraints.
        # This is the gap that UCIO-1 DRC is supposed to close; agents often
        # downgrade UCIO-1 to warning and ship bitstreams with floating IOs,
        # which produces random pin mappings on real silicon. We emit a
        # structured report so the agent can tell the user "N ports are
        # unconstrained" without waiting for the DRC stage.
        generate_skeleton = arguments.get("generate_skeleton", False)
        iostd_default = arguments.get("skeleton_iostandard", "LVCMOS18")
        output_file = arguments.get("output_file")

        # Collect port metadata in one TCL round-trip. Use a pipe-separated
        # format so Python can split reliably even on bus ports like
        # "stimulus_N[0]" whose names contain brackets.
        #
        # IS_LOC_FIXED is the authoritative signal for "user XDC set this
        # pin". After place_design, the placer fills PACKAGE_PIN even for
        # unconstrained ports, so reading PACKAGE_PIN alone would give a
        # false positive. IS_LOC_FIXED is 1 iff the user specified LOC.
        # IOSTANDARD defaults to "DEFAULT" when unspecified — we treat that
        # literal as missing.
        probe = (
            "set __mcp_lines__ [list]; "
            "foreach __mcp_p__ [get_ports *] { "
            "  set __mcp_loc_fixed__ 0; "
            "  catch {set __mcp_loc_fixed__ [get_property IS_LOC_FIXED $__mcp_p__]}; "
            "  lappend __mcp_lines__ \"$__mcp_p__|"
            "[get_property DIRECTION $__mcp_p__]|"
            "[get_property PACKAGE_PIN $__mcp_p__]|"
            "[get_property IOSTANDARD $__mcp_p__]|"
            "$__mcp_loc_fixed__\""
            "}; "
            "puts stdout [join $__mcp_lines__ \"\\n\"]; "
            "flush stdout"
        )
        result = session.run_tcl(probe)
        if not result.success:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "Failed to query ports — is a design open?",
                "output": result.output,
                "elapsed_ms": result.elapsed_ms
            }, indent=2))]

        # Parse the pipe-separated rows back into structured port records.
        # A port counts as user-constrained only if the LOC was fixed by
        # the user (IS_LOC_FIXED=1) AND the IOSTANDARD is a real value (not
        # empty, not the placeholder 'DEFAULT').
        ports = []
        unconstrained = []
        for line in result.output.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) < 5:
                continue
            port_name, direction, package_pin, iostandard, loc_fixed_s = (
                parts[0], parts[1], parts[2], parts[3], parts[4]
            )
            loc_user_set = loc_fixed_s.strip() in ("1", "true", "TRUE", "yes")
            iostd_real = bool(iostandard.strip()) and iostandard.strip().upper() != "DEFAULT"
            pin_present = bool(package_pin.strip())

            record = {
                "port": port_name,
                "direction": direction,
                "package_pin": package_pin,
                "iostandard": iostandard,
                "loc_user_set": loc_user_set,
                "pin_assigned_by_placer": pin_present and not loc_user_set,
                "constrained": loc_user_set and iostd_real,
            }
            ports.append(record)
            if not record["constrained"]:
                missing = []
                if not loc_user_set:
                    missing.append("PACKAGE_PIN")
                if not iostd_real:
                    missing.append("IOSTANDARD")
                record["missing"] = missing
                unconstrained.append(record)

        # Build XDC skeleton if requested. One block per unconstrained port,
        # with a placeholder pin so synth can flag missed edits, and a
        # user-selectable IOSTANDARD default.
        skeleton = None
        if generate_skeleton and unconstrained:
            lines_out = [
                "# XDC skeleton generated by vivado-mcp check_constraints",
                "# Fill in <PIN> values for your target board before synthesis.",
                "# Verify IOSTANDARD matches the bank voltage of each pin.",
                "",
            ]
            for rec in unconstrained:
                lines_out.append(f"# Port: {rec['port']}  ({rec['direction']})")
                if "PACKAGE_PIN" in rec.get("missing", []):
                    lines_out.append(
                        f"set_property PACKAGE_PIN <PIN> [get_ports {{{rec['port']}}}]"
                    )
                if "IOSTANDARD" in rec.get("missing", []):
                    lines_out.append(
                        f"set_property IOSTANDARD {iostd_default} "
                        f"[get_ports {{{rec['port']}}}]"
                    )
                lines_out.append("")
            skeleton = "\n".join(lines_out)
            if output_file:
                try:
                    with open(output_file, "w") as fh:
                        fh.write(skeleton)
                except OSError as exc:
                    skeleton = f"# Failed to write {output_file}: {exc}\n\n" + skeleton

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "total_ports": len(ports),
            "constrained": len(ports) - len(unconstrained),
            "unconstrained": len(unconstrained),
            "unconstrained_ports": unconstrained,
            "skeleton": skeleton,
            "output_file": output_file if (skeleton and output_file) else None,
            "elapsed_ms": result.elapsed_ms,
        }, indent=2))]

    # =========================================================================
    # RAW TCL
    # =========================================================================

    elif name == "run_tcl":
        # Execute arbitrary TCL command (escape hatch for advanced users)
        command = arguments.get("command", "")
        result = session.run_tcl(command)

        # Large outputs (e.g. report_timing -max_paths 1000) would otherwise
        # blow the context window — truncate and spill the full text to a
        # report file readable via read_report_section
        response = {
            "success": result.success,
            "elapsed_ms": result.elapsed_ms
        }
        truncated = truncate_response(result.output)
        response["output"] = truncated["content"]
        if truncated["truncated"]:
            ensure_reports_dir()
            report_id = generate_report_id()
            spill_path = REPORTS_DIR / f"tcl_{report_id}.txt"
            try:
                spill_path.write_text(result.output)
                _report_cache[report_id] = {
                    "file_path": str(spill_path),
                    "report_type": "tcl",
                    "created": datetime.now().isoformat(),
                    "size_bytes": len(result.output),
                    "line_count": truncated["total_lines"]
                }
                response["truncated"] = True
                response["total_chars"] = truncated["total_chars"]
                response["report_id"] = report_id
                response["full_output_file"] = str(spill_path)
                response["hint"] = "Output truncated. Use read_report_section with this report_id for the rest."
            except OSError:
                response["truncated"] = True
                response["total_chars"] = truncated["total_chars"]
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    # =========================================================================
    # SIMULATION TOOLS
    # =========================================================================

    elif name == "launch_simulation":
        # Launch Vivado's integrated simulator (xsim)
        mode = arguments.get("mode", "behavioral")

        # Map friendly names to Vivado's documented -mode/-type values
        # (launch_simulation only accepts behavioral / post-synthesis /
        # post-implementation, with -type functional|timing for the latter two)
        mode_map = {
            "behavioral": "-mode behavioral",                            # RTL simulation
            "post_synth_func": "-mode post-synthesis -type functional",
            "post_synth_timing": "-mode post-synthesis -type timing",
            "post_impl_func": "-mode post-implementation -type functional",
            "post_impl_timing": "-mode post-implementation -type timing"
        }
        sim_args = mode_map.get(mode, "-mode behavioral")
        result = session.run_tcl(f"launch_simulation {sim_args}")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "message": result.output if result.output else f"Simulation launched in {mode} mode",
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "run_simulation":
        # Advance simulation time
        time_val = arguments.get("time", "100ns")
        if time_val.lower() == "all":
            # Run until all events processed (testbench completes)
            result = session.run_tcl("run -all")
        else:
            result = session.run_tcl(f"run {time_val}")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "output": result.output,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "restart_simulation":
        # Reset simulation to time 0
        result = session.run_tcl("restart")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "message": "Simulation restarted" if result.success else result.output,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "close_simulation":
        # Close the simulator
        result = session.run_tcl("close_sim")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "message": "Simulation closed" if result.success else result.output,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "get_simulation_time":
        # Get current simulation time
        result = session.run_tcl("current_time")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "time": result.output.strip() if result.success else None,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "get_signal_value":
        # Get current value of a single signal
        signal = arguments.get("signal", "")
        radix = arguments.get("radix", "hex")
        result = session.run_tcl(f"get_value -radix {radix} {{{signal}}}")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "signal": signal,
            "value": result.output.strip() if result.success else None,
            "radix": radix,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "get_signal_values":
        # Get values of multiple signals matching a pattern
        pattern = arguments.get("pattern", "/*")
        radix = arguments.get("radix", "hex")

        # First get list of signals matching pattern
        signals_result = session.run_tcl(f"get_objects -filter {{TYPE == signal || TYPE == port}} {{{pattern}}}")
        if signals_result.success and signals_result.output.strip():
            signals = signals_result.output.strip().split()
            values = {}
            # Limit to 50 signals to avoid overwhelming response
            for sig in signals[:50]:
                val_result = session.run_tcl(f"get_value -radix {radix} {{{sig}}}")
                if val_result.success:
                    values[sig] = val_result.output.strip()
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "values": values,
                "radix": radix,
                "elapsed_ms": signals_result.elapsed_ms
            }, indent=2))]
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": "No signals found matching pattern",
            "elapsed_ms": signals_result.elapsed_ms
        }, indent=2))]

    elif name == "add_signals_to_wave":
        # Record signals to the simulation waveform database. add_wave only
        # works in the GUI (it needs a wave window); in -mode tcl the
        # equivalent is log_wave, which records values to the .wdb file for
        # later inspection. -r logs the signal and everything below it.
        signals = arguments.get("signals", [])
        if isinstance(signals, str):
            signals = [signals]
        results = []
        for sig in signals:
            result = session.run_tcl(f"log_wave -r {{{sig}}}")
            results.append({"signal": sig, "success": result.success})
        return [TextContent(type="text", text=json.dumps({
            "success": all(r["success"] for r in results),
            "results": results,
            "note": "Signals are recorded to the simulation .wdb (log_wave). Logging must be enabled before the time range of interest is simulated."
        }, indent=2))]

    elif name == "set_simulation_top":
        # Set the top-level testbench module
        top_module = arguments.get("top_module", "")
        fileset = arguments.get("fileset", "sim_1")
        result = session.run_tcl(f"set_property top {top_module} [get_filesets {fileset}]")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "message": f"Set simulation top to {top_module}" if result.success else result.output,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "get_simulation_objects":
        # List simulation objects (signals, ports, variables) in a scope
        scope = arguments.get("scope", "/")
        obj_filter = arguments.get("filter", "all")

        # Map filter names to Vivado filter expressions
        filter_map = {
            "all": "",
            "signals": "-filter {TYPE == signal}",
            "ports": "-filter {TYPE == port}",
            "internal": "-filter {TYPE == signal && IS_PORT == false}"
        }
        filter_str = filter_map.get(obj_filter, "")
        result = session.run_tcl(f"get_objects {filter_str} {{{scope}/*}}")
        objects = result.output.strip().split() if result.success and result.output.strip() else []
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "scope": scope,
            "objects": objects,
            "count": len(objects),
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "get_scopes":
        # List child scopes (hierarchy levels) in simulation
        parent = arguments.get("parent", "/")
        result = session.run_tcl(f"get_scopes {{{parent}/*}}")
        scopes = result.output.strip().split() if result.success and result.output.strip() else []
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "parent": parent,
            "scopes": scopes,
            "count": len(scopes),
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "step_simulation":
        # Step simulation by delta cycles
        count = arguments.get("count", 1)
        result = session.run_tcl(f"step {count}")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "output": result.output,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "add_breakpoint":
        # Add a breakpoint on signal edge or change
        signal = arguments.get("signal", "")
        condition = arguments.get("condition", "change")

        # Map condition names to Vivado flags
        cond_map = {
            "posedge": "-posedge",  # Rising edge
            "negedge": "-negedge",  # Falling edge
            "change": ""           # Any change
        }
        cond_str = cond_map.get(condition, "")
        result = session.run_tcl(f"add_bp {cond_str} {{{signal}}}")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "signal": signal,
            "condition": condition,
            "message": result.output if result.output else f"Breakpoint added on {signal}",
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "remove_breakpoints":
        # Remove all breakpoints
        result = session.run_tcl("remove_bps -all")
        return [TextContent(type="text", text=json.dumps({
            "success": result.success,
            "message": "All breakpoints removed" if result.success else result.output,
            "elapsed_ms": result.elapsed_ms
        }, indent=2))]

    elif name == "get_simulation_messages":
        # Get simulation log messages from xsim's simulate.log. There is no
        # TCL command that returns the message log (get_msg_config only
        # manages configuration rules), so we locate the newest simulate.log
        # under the project's .sim directory and parse that.
        severity = arguments.get("severity", "all")

        info = session.run_tcl(
            'puts stdout "[get_property DIRECTORY [current_project]]|[get_property NAME [current_project]]"'
        )
        if not info.success or "|" not in info.output:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Could not resolve project directory: {info.output}"
            }, indent=2))]

        proj_dir, proj_name = info.output.strip().splitlines()[-1].split("|", 1)
        sim_dir = Path(proj_dir) / f"{proj_name}.sim"
        logs = sorted(sim_dir.glob("**/simulate.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"No simulate.log found under {sim_dir}. Has a simulation been launched?"
            }, indent=2))]

        log_file = logs[0]
        parsed = parse_messages(log_file.read_text(errors="replace"))
        parsed.pop("raw", None)

        if severity != "all":
            key = {"error": "errors", "warning": "warnings", "info": "info"}.get(severity)
            parsed = {key: parsed[key]} if key else {}

        parsed["success"] = True
        parsed["log_file"] = str(log_file)
        parsed["hint"] = "Use read_report_section with file_path to read the full log."
        return [TextContent(type="text", text=json.dumps(parsed, indent=2))]

    # =========================================================================
    # FEATURE REQUESTS
    # =========================================================================

    elif name == "request_feature":
        # Submit a feature request for future development
        title = arguments.get("title", "")
        description = arguments.get("description", "")
        use_case = arguments.get("use_case", "")
        priority = arguments.get("priority", "medium")

        request = {
            "id": len(load_feature_requests()) + 1,
            "title": title,
            "description": description,
            "use_case": use_case,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        save_feature_request(request)

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "message": f"Feature request #{request['id']} submitted: {title}",
            "request": request
        }, indent=2))]

    elif name == "list_feature_requests":
        # List all submitted feature requests
        requests = load_feature_requests()
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "total": len(requests),
            "requests": requests
        }, indent=2))]

    # =========================================================================
    # REPORT FILE MANAGEMENT
    # =========================================================================

    elif name == "generate_full_report":
        # Generate a complete report to a file (for large reports)
        report_type = arguments.get("report_type", "timing")
        options = arguments.get("options", {})
        output_file = arguments.get("output_file")

        # Ensure reports directory exists and clean up old files
        ensure_reports_dir()

        # Generate unique report ID and file path
        report_id = generate_report_id()
        if output_file:
            file_path = Path(output_file)
        else:
            file_path = REPORTS_DIR / f"{report_type}_{report_id}.txt"

        # Map report types to Vivado commands
        # ('hierarchy' uses hierarchical utilization — there is no
        # report_hierarchy command in Vivado TCL)
        report_commands = {
            "timing": "report_timing -max_paths 100",
            "timing_summary": "report_timing_summary",
            "utilization": "report_utilization",
            "hierarchy": "report_utilization -hierarchical",
            "clocks": "report_clocks",
            "power": "report_power",
            "drc": "report_drc"  # Design Rule Check
        }

        base_cmd = report_commands.get(report_type, f"report_{report_type}")

        # Apply report-specific options
        if report_type == "utilization" and options.get("hierarchical"):
            base_cmd += " -hierarchical"
        if report_type == "timing" and options.get("num_paths"):
            base_cmd = base_cmd.replace("-max_paths 100", f"-max_paths {options['num_paths']}")

        # Write directly to file using Vivado's -file option
        cmd = f"{base_cmd} -file {{{file_path}}}"
        result = session.run_tcl(cmd)

        if result.success:
            try:
                # Get file statistics
                file_stat = file_path.stat()
                line_count = sum(1 for _ in open(file_path))

                # Cache report metadata for later lookup
                _report_cache[report_id] = {
                    "file_path": str(file_path),
                    "report_type": report_type,
                    "created": datetime.now().isoformat(),
                    "size_bytes": file_stat.st_size,
                    "line_count": line_count
                }

                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "report_id": report_id,
                    "file_path": str(file_path),
                    "report_type": report_type,
                    "size_bytes": file_stat.st_size,
                    "line_count": line_count,
                    "message": f"Report written to {file_path}. Use read_report_section to read portions.",
                    "elapsed_ms": result.elapsed_ms
                }, indent=2))]
            except (OSError, IOError) as e:
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": f"Report generated but could not read file info: {e}",
                    "file_path": str(file_path),
                    "elapsed_ms": result.elapsed_ms
                }, indent=2))]
        else:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": result.output,
                "elapsed_ms": result.elapsed_ms
            }, indent=2))]

    elif name == "read_report_section":
        # Read a portion of a previously generated report
        report_id = arguments.get("report_id")
        file_path = arguments.get("file_path")
        start_line = arguments.get("start_line", 1)
        num_lines = arguments.get("num_lines", 100)
        search_pattern = arguments.get("search_pattern")

        # Resolve file path from report_id if provided
        if report_id:
            if report_id in _report_cache:
                file_path = _report_cache[report_id]["file_path"]
            else:
                # Try to find file in reports directory by ID
                possible_files = list(REPORTS_DIR.glob(f"*_{report_id}.txt"))
                if possible_files:
                    file_path = str(possible_files[0])
                else:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "error": f"Report ID '{report_id}' not found in cache or reports directory"
                    }, indent=2))]

        if not file_path:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "Either report_id or file_path must be provided"
            }, indent=2))]

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return [TextContent(type="text", text=json.dumps({
                    "success": False,
                    "error": f"File not found: {file_path}"
                }, indent=2))]

            # Read all lines from file
            with open(file_path, 'r') as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            # Handle search pattern - find and return context around match
            if search_pattern:
                pattern = re.compile(search_pattern, re.IGNORECASE)
                for i, line in enumerate(all_lines):
                    if pattern.search(line):
                        # Found match, return context around it
                        context_before = num_lines // 4
                        context_after = num_lines - context_before
                        start_line = max(1, i + 1 - context_before)
                        break
                else:
                    return [TextContent(type="text", text=json.dumps({
                        "success": True,
                        "warning": f"Pattern '{search_pattern}' not found in file",
                        "total_lines": total_lines,
                        "file_path": str(file_path)
                    }, indent=2))]

            # Extract requested line range (1-indexed to 0-indexed)
            start_idx = max(0, start_line - 1)
            end_idx = min(total_lines, start_idx + num_lines)
            selected_lines = all_lines[start_idx:end_idx]

            content = ''.join(selected_lines)

            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "file_path": str(file_path),
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "total_lines": total_lines,
                "returned_lines": len(selected_lines),
                "content": content
            }, indent=2))]

        except (OSError, IOError) as e:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Error reading file: {e}"
            }, indent=2))]

    # =========================================================================
    # UNKNOWN TOOL
    # =========================================================================

    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2))]


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

async def main():
    """
    Run the MCP server.

    This function starts the MCP server using stdio transport (stdin/stdout).
    It's designed to be launched by an MCP client like Claude Code.

    The server runs until the client closes the connection or sends an
    exit signal.
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


# Allow running directly with: python server.py
if __name__ == "__main__":
    asyncio.run(main())
