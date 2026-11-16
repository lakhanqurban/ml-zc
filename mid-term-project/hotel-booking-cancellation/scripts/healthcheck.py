"""Container health check.

Reads the port from ``$PORT`` (falling back to 9696) so the same image works
locally and on platforms that assign their own port (Render, Fly.io, Cloud
Run).
"""

from __future__ import annotations

import os
import sys
import urllib.request

PORT = os.getenv("PORT", "9696")

try:
    with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
