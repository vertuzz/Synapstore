#!/usr/bin/env python3
"""Performance gate (PRD §10): reject a change that degrades recall speed,
recall quality, RAM consumption or the zero-copy contract.

Usage:
    perf_gate.py HEAD.json [BASE.json]

HEAD.json is the `bench/results/synthetic.json` of the candidate. BASE.json,
when given, is the same benchmark run on the target branch *on the same
machine*; relative latency/RSS/throughput checks are only applied then.
Absolute quality invariants are always enforced. Tolerance for relative
checks: $PERF_TOLERANCE (default 0.25 = 25 %).
"""

import json
import os
import sys

TOL = float(os.environ.get("PERF_TOLERANCE", "0.25"))

# Absolute invariants: (metric, predicate, description)
ABSOLUTE = [
    ("recall_at_10", lambda v: v >= 0.95, ">= 0.95"),
    ("graph_mismatches", lambda v: v == 0, "== 0"),
    ("decay_pass_rate", lambda v: v == 1.0, "== 1.0"),
    ("unreachable_nodes", lambda v: v == 0, "== 0"),
    ("zero_copy_alloc_bytes_per_read", lambda v: v == 0, "== 0 (rkyv zero-copy contract)"),
]

# Relative checks against the base run: (metric, direction, slack)
LOWER_IS_BETTER = ["recall_p50_us", "recall_p95_us", "recall_decay_p50_us", "graph_p50_us", "events_p50_us"]


def get(d: dict, dotted: str):
    for part in dotted.split("."):
        d = d[part]
    return d


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    head = json.load(open(sys.argv[1]))
    base = json.load(open(sys.argv[2])) if len(sys.argv) == 3 else None
    failures = []

    for metric, ok, desc in ABSOLUTE:
        v = get(head, metric)
        status = "ok" if ok(v) else "FAIL"
        print(f"[{status}] {metric} = {v} (must be {desc})")
        if status == "FAIL":
            failures.append(metric)

    if base is not None:
        def rel(metric: str, head_v: float, base_v: float, worse: bool, limit: str):
            status = "FAIL" if worse else "ok"
            print(f"[{status}] {metric}: head {head_v:.1f} vs base {base_v:.1f} (limit {limit})")
            if worse:
                failures.append(metric)

        for m in LOWER_IS_BETTER:
            h, b = get(head, m), get(base, m)
            # 50 us absolute slack absorbs timer noise on very fast operations.
            rel(m, h, b, h > b * (1 + TOL) + 50, f"<= base x {1 + TOL:.2f} + 50us")
        h, b = head["peak_rss_mb"], base["peak_rss_mb"]
        rel("peak_rss_mb", h, b, h > b * 1.10 + 8, "<= base x 1.10 + 8 MiB")
        h, b = get(head, "load.memories_per_sec"), get(base, "load.memories_per_sec")
        rel("load.memories_per_sec", h, b, h < b * (1 - TOL), f">= base x {1 - TOL:.2f}")
        h, b = head["recall_at_10"], base["recall_at_10"]
        rel("recall_at_10 (vs base)", h, b, h < b - 0.005, ">= base - 0.005")

    if failures:
        print(f"\nPerformance gate FAILED: {', '.join(failures)}")
        return 1
    print("\nPerformance gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
