#!/usr/bin/env bash
# internal: run the Warp build and print only diagnostics
cd "$(dirname "$0")/.." && bash scripts/build_warp.sh "$@" > logs/build_warp.log 2>&1; rc=$?
grep -E "error|Error|undefined|warning: unused" logs/build_warp.log | grep -v "^/usr/bin/g++\|hipcc\" -x hip\|-Werror -W" | sort -u | head -40
echo "exit=$rc"
