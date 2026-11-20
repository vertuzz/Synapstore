"""Black-box MCP integration tests (stdlib only).

Spawns the `synapstore` binary, writes JSON-RPC lines to its stdin and
asserts on stdout. Run after `cargo build`:

    python3 -m unittest discover -s tests/python -v

The binary is taken from $SYNAPSTORE_BIN, else target/debug/synapstore.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = os.environ.get("SYNAPSTORE_BIN", str(ROOT / "target" / "debug" / "synapstore"))


class McpClient:
    def __init__(self, db: Path, *extra: str):
        self.proc = subprocess.Popen(
            [BIN, "--mcp", str(db), *extra],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.next_id = 0

    def send(self, payload) -> dict | list | None:
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line) if line else None

    def request(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        reply = self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}})
        assert reply["id"] == self.next_id, reply
        return reply

    def notify(self, method: str) -> None:
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def tool(self, name: str, **arguments) -> tuple[bool, dict]:
        result = self.request("tools/call", {"name": name, "arguments": arguments})["result"]
        return result["isError"], result["structuredContent"]

    def close(self) -> int:
        self.proc.stdin.close()
        return self.proc.wait(timeout=30)


class McpProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "agent_memory.redb"
        self.client = McpClient(self.db)

    def tearDown(self):
        self.assertEqual(self.client.close(), 0)
        self.tmp.cleanup()

    def test_handshake_and_tool_catalogue(self):
        init = self.client.request(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "py", "version": "1"}},
        )["result"]
        self.assertEqual(init["protocolVersion"], "2025-06-18")
        self.assertIn("tools", init["capabilities"])
        self.client.notify("notifications/initialized")
        tools = {t["name"]: t for t in self.client.request("tools/list")["result"]["tools"]}
        for name in ("memorize", "recall", "graph_query"):
            self.assertIn(name, tools)
            self.assertEqual(tools[name]["inputSchema"]["type"], "object")

    def test_memorize_recall_graph_roundtrip(self):
        err, out = self.client.tool(
            "memorize",
            text="Marie Curie won the Nobel Prize in Physics in 1903.",
            facts=[["Marie Curie", "won", "Nobel Prize in Physics"], ["Marie Curie", "born_in", "Warsaw"]],
            tags=["science"],
        )
        self.assertFalse(err, out)
        self.assertEqual(out["created"], 1)
        self.client.tool("memorize", text="Bread, eggs and coffee are on the shopping list.")
        err, out = self.client.tool("recall", query="Which prize did Marie Curie win?", top_k=1)
        self.assertFalse(err, out)
        self.assertIn("Marie Curie", out["hits"][0]["text"])
        self.assertGreater(out["hits"][0]["decay_factor"], 1.0)
        _, graph = self.client.tool("graph_query", subject="Warsaw", depth=2)
        predicates = sorted(f["predicate"] for f in graph["facts"])
        self.assertEqual(predicates, ["born_in", "won"])
        # Fact subjects/objects are linked to the memory that asserted them.
        self.assertEqual(len(graph["memories"]), 1)
        _, nothing = self.client.tool("graph_query", subject="Atlantis")
        self.assertFalse(nothing["found"])

    def test_idempotent_memorize(self):
        _, a = self.client.tool("memorize", text="The deploy key rotates monthly.")
        _, b = self.client.tool("memorize", text="  the DEPLOY key rotates   monthly. ")
        self.assertEqual(a["chunk_ids"], b["chunk_ids"])
        self.assertEqual((b["created"], b["updated"]), (0, 1))
        _, stats = self.client.tool("stats")
        self.assertEqual(stats["counts"]["chunks"], 1)

    def test_errors_do_not_kill_the_server(self):
        self.assertEqual(self.client.send("not an object")["error"]["code"], -32600)
        self.assertEqual(self.client.request("bogus/method")["error"]["code"], -32601)
        err, out = self.client.tool("memorize")
        self.assertTrue(err)
        self.assertEqual(out["error"]["kind"], "invalid_input")
        err, out = self.client.tool("recall", vector=[0, 0])
        self.assertTrue(err)
        self.assertEqual(self.client.request("ping")["result"], {})

    def test_episodic_log_and_explain(self):
        self.client.tool("log_event", kind="tool_result", text="pytest: 42 passed", timestamp_ms=1_000)
        self.client.tool("log_event", kind="thought", text="should refactor the parser", timestamp_ms=2_000)
        _, ev = self.client.tool("recent_events", limit=5)
        self.assertEqual([e["kind"] for e in ev["events"]], ["thought", "tool_result"])
        _, mem = self.client.tool("memorize", text="The parser lives in src/mcp/jsonrpc.rs")
        _, ex = self.client.tool("explain", target_id=mem["chunk_ids"][0], query="where is the parser?")
        self.assertTrue(ex["verdict"].startswith("retrieved"), ex)


if __name__ == "__main__":
    unittest.main()
