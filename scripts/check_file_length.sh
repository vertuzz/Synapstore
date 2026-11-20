#!/usr/bin/env bash
# PRD §10: every Rust module must be at most 300 lines.
set -euo pipefail
limit=${1:-300}
status=0
while IFS= read -r f; do
  n=$(wc -l < "$f")
  if [ "$n" -gt "$limit" ]; then
    echo "::error file=$f::$n lines (limit $limit)"
    status=1
  fi
done < <(find src tests benches -name '*.rs' | sort)
[ "$status" -eq 0 ] && echo "all modules <= $limit lines"
exit "$status"
