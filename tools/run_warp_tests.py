"""Run the Warp test suite one module per process so native crashes stay contained.

Usage: python tools/run_warp_tests.py [-j JOBS] [--timeout SEC] [--filter SUBSTR] [--out DIR]
Writes <out>/summary.json and one log per module.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
WARP = ROOT / "third_party" / "warp-rocm"


def discover(filter_substr):
    mods = []
    for p in sorted((WARP / "warp" / "tests").rglob("test_*.py")):
        mod = ".".join(p.relative_to(WARP).with_suffix("").parts)
        if filter_substr and filter_substr not in mod:
            continue
        mods.append(mod)
    return mods


def run_one(mod, timeout, out_dir):
    log = out_dir / f"{mod}.log"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", mod],
            cwd=WARP, capture_output=True, text=True, timeout=timeout,
        )
        rc, text = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        rc, text = "timeout", (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
    log.write_text(text)
    m = re.search(r"Ran (\d+) tests?", text)
    ran = int(m.group(1)) if m else 0
    fails = len(re.findall(r"^(?:FAIL|ERROR): ", text, re.M))
    skips = len(re.findall(r"\.\.\. skipped", text))
    crashed = rc not in (0, 1) or (rc == 1 and not m)
    return {"module": mod, "rc": rc, "ran": ran, "failed": fails, "skipped": skips,
            "crashed": crashed, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--jobs", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--filter", default="")
    ap.add_argument("--out", default=str(ROOT / "test_results" / "warp"))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    mods = discover(args.filter)
    results = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(run_one, m, args.timeout, out_dir): m for m in mods}
        for f in concurrent.futures.as_completed(futs):
            r = f.result(); results.append(r)
            flag = "CRASH" if r["crashed"] else ("FAIL" if r["failed"] else "ok")
            print(f"[{len(results):3d}/{len(mods)}] {flag:5s} {r['module']} ran={r['ran']} failed={r['failed']} ({r['seconds']}s)", flush=True)
    results.sort(key=lambda r: r["module"])
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    tot = {k: sum(r[k] for r in results) for k in ("ran", "failed", "skipped")}
    crashed = [r["module"] for r in results if r["crashed"]]
    print(f"\nmodules={len(results)} tests_ran={tot['ran']} failed={tot['failed']} skipped={tot['skipped']} crashed_modules={len(crashed)}")
    for c in crashed:
        print("  crashed:", c)


if __name__ == "__main__":
    main()
