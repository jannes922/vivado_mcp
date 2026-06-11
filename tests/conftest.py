"""
Test fixtures for vivado_mcp.

The parsers under test are pure functions, but they live in modules that
import `mcp` and `pexpect` at the top. To keep the tests runnable without
those runtime dependencies installed (e.g. in a bare CI container), minimal
stubs are injected into sys.modules before the package is imported. When the
real packages are available they are used instead.

The repo root *is* the package (package-dir maps "vivado_mcp" to "."), so the
package is loaded explicitly from __init__.py regardless of what the checkout
directory happens to be called.
"""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _stub_mcp():
    mcp = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    stdio_mod = types.ModuleType("mcp.server.stdio")
    types_mod = types.ModuleType("mcp.types")

    class Server:
        def __init__(self, name):
            self.name = name

        def list_tools(self):
            return lambda fn: fn

        def call_tool(self):
            return lambda fn: fn

        def create_initialization_options(self):
            return None

        async def run(self, *args, **kwargs):
            pass

    class _Record:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    server_mod.Server = Server
    stdio_mod.stdio_server = None
    types_mod.Tool = _Record
    types_mod.TextContent = _Record

    mcp.server = server_mod
    sys.modules["mcp"] = mcp
    sys.modules["mcp.server"] = server_mod
    sys.modules["mcp.server.stdio"] = stdio_mod
    sys.modules["mcp.types"] = types_mod


def _stub_pexpect():
    pexpect = types.ModuleType("pexpect")

    class TIMEOUT(Exception):
        pass

    class EOF(Exception):
        pass

    pexpect.TIMEOUT = TIMEOUT
    pexpect.EOF = EOF
    pexpect.spawn = None
    sys.modules["pexpect"] = pexpect


try:
    import mcp  # noqa: F401
except ImportError:
    _stub_mcp()

try:
    import pexpect  # noqa: F401
except ImportError:
    _stub_pexpect()


if "vivado_mcp" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "vivado_mcp",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["vivado_mcp"] = pkg
    spec.loader.exec_module(pkg)
