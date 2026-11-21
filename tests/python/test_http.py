"""Black-box tests of `synapstore serve --listen` (stdlib only).

Starts the binary on an ephemeral port, then talks Streamable HTTP with
urllib exactly like a remote MCP client would. Run after `cargo build`:

    python3 -m unittest discover -s tests/python -v
"""

import json
import os
import re
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = os.environ.get("SYNAPSTORE_BIN", str(ROOT / "target" / "debug" / "synapstore"))


class HttpServer:
    def __init__(self, db: Path, *extra: str):
        self.proc = subprocess.Popen(
            [BIN, "serve", str(db), "--listen", "127.0.0.1:0", *extra],
            stderr=subprocess.PIPE,
            text=True,
        )
        self.token = None
        deadline = time.time() + 30
        while time.time() < deadline:
            line = self.proc.stderr.readline()
            if not line:
                break
            if m := re.search(r"(syn_[A-Za-z0-9_-]{43})", line):
                self.token = m.group(1)
            if m := re.search(r"serving MCP on (http://\S+/mcp)", line):
                self.url = m.group(1)
                return
        raise RuntimeError("server did not start")

    def post(self, payload, token=None, session=None, headers=None):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if token:
            h["Authorization"] = f"Bearer {token}"
        if session:
            h["Mcp-Session-Id"] = session
        h.update(headers or {})
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=h, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                return r.status, dict(r.headers), json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            body = e.read()
            return e.code, dict(e.headers), json.loads(body) if body else None

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.proc.stderr.close()


def init():
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "py", "version": "1"}},
    }


def call(i, name, **args):
    return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}


class HttpTransportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "mem.redb"
        self.server = HttpServer(self.db)

    def tearDown(self):
        self.server.stop()
        self.tmp.cleanup()

    def test_bootstrap_token_and_session(self):
        s = self.server
        self.assertIsNotNone(s.token, "first start prints a bootstrap token")
        status, headers, _ = s.post(init())
        self.assertEqual(status, 401)
        self.assertIn("Bearer", headers["WWW-Authenticate"])
        status, headers, body = s.post(init(), token=s.token)
        self.assertEqual(status, 200)
        self.assertEqual(body["result"]["protocolVersion"], "2025-11-25")
        sid = headers["Mcp-Session-Id"]
        status, _, _ = s.post({"jsonrpc": "2.0", "method": "notifications/initialized"}, s.token, sid)
        self.assertEqual(status, 202)
        _, _, body = s.post(call(2, "memorize", text="The deploy key rotates every 90 days"), s.token, sid)
        self.assertFalse(body["result"]["isError"])
        _, _, body = s.post(call(3, "recall", query="how often does the deploy key rotate"), s.token, sid)
        self.assertIn("90 days", body["result"]["structuredContent"]["hits"][0]["text"])

    def test_cli_tokens_work_against_a_running_server(self):
        s = self.server
        out = subprocess.run(
            [BIN, "token", "create", str(self.db), "--name", "ro", "--scope", "read"],
            capture_output=True, text=True, check=True,
        )
        ro = out.stdout.strip()
        self.assertTrue(ro.startswith("syn_"))
        _, _, body = s.post(call(1, "memorize", text="nope"), token=ro)
        self.assertEqual(body["result"]["structuredContent"]["error"]["kind"], "forbidden")
        listing = subprocess.run([BIN, "token", "list", str(self.db)], capture_output=True, text=True, check=True)
        self.assertIn("ro", listing.stdout)
        self.assertNotIn(ro, listing.stdout, "secrets are never listed")
        subprocess.run([BIN, "token", "revoke", str(self.db), "ro"], capture_output=True, check=True)
        status, _, _ = s.post(call(2, "recall", query="x"), token=ro)
        self.assertEqual(status, 401)

    def test_foreign_origin_is_rejected(self):
        status, _, _ = self.server.post(init(), self.server.token, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
