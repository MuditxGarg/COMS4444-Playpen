#!/usr/bin/env python3
"""Test the Iteration 2 beam-search weights on the real simulator (report item 2).

Claim being tested: the "-0.7 x distance from start" term and the
"+0.045 x turn" term pushed the beam toward small shapes.

Needs iter2.py = the Iteration 2 player (commit 14d91c7), saved in this folder:
    cmd /c "git show 14d91c7:players/player8/player.py > players\\player8\\tests\\iter2.py"

Run (from the repo root; about 20 min, give it a free machine):
    python players/player8/tests/iter2_weights.py --jobs 4

It builds four variants in memory (base, nodist = distance term off,
noturn = turn term off, none = both off), runs each on each room and writes
iter2_weights.csv plus a table of enclosed area relative to the base player.
Finished runs are skipped if you start it again.
"""
import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_root():
    for d in [HERE, *HERE.parents]:
        if (d / "main.py").exists() and (d / "src").is_dir():
            return d
    sys.exit("Could not find the repo root.")


ROOT = find_root()
DIST = "-0.7 * math.hypot(x, y)"
TURN = "min(heading_turn, 135) * 0.045"
VARIANTS = {
    "base": [],
    "nodist": [(DIST, "-0.0 * math.hypot(x, y)")],
    "noturn": [(TURN, "min(heading_turn, 135) * 0.0")],
    "none": [(DIST, "-0.0 * math.hypot(x, y)"), (TURN, "min(heading_turn, 135) * 0.0")],
}


def find_scenario(name):
    hits = [p for p in ROOT.rglob(f"{name}.json") if ".git" not in p.parts]
    hits.sort(key=lambda p: len(p.parts))
    return hits[0] if hits else None


def worker(variant, scn_path):
    src = (HERE / "iter2.py").read_text(encoding="utf-8")
    for old, new in VARIANTS[variant]:
        if src.count(old) != 1:
            raise SystemExit(
                f"expected exactly one '{old}' in iter2.py, found {src.count(old)}")
        src = src.replace(old, new)
    path = HERE / f"_iter2_{variant}.py"
    path.write_text(src, encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(f"_iter2_{variant}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    from src.enclosure import score_construction, validate_construction
    from src.scenario import read_scenario

    sc = read_scenario(scn_path)
    t0 = time.process_time()
    p = mod.Player8(sc.room.copy(), sc.inventory.copy(), sc.weights)
    c = p.build_enclosure()
    cpu = time.process_time() - t0
    out = {"score": 0.0, "area": 0.0, "perimeter": 0.0, "cpu": cpu}
    if c is not None:
        r = validate_construction(c, sc.room, sc.inventory)
        if r.valid:
            out.update(score=float(score_construction(r, sc.weights)),
                       area=float(r.area), perimeter=float(r.perimeter))
        else:
            out["score"] = -1000.0
    print("RESULT " + json.dumps(out))


def run(job):
    room, variant, scn = job
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker", variant, str(scn)]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"score": float("nan"), "area": 0, "perimeter": 0, "cpu": 900}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    err = (proc.stderr.strip().splitlines() or ["?"])[-1]
    print("   crash:", err)
    return {"score": float("nan"), "area": 0, "perimeter": 0, "cpu": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2)
    ap.add_argument("--rooms", default="spiral,crown,corridor,curved")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="iter2_weights.csv")
    a = ap.parse_args()
    if a.worker:
        worker(*a.worker)
        return
    if not (HERE / "iter2.py").exists():
        sys.exit("iter2.py not found next to this script (see the top of this file).")

    fields = ["room", "variant", "score", "area", "perimeter", "cpu"]
    out = Path(a.out)
    done = {}
    if out.exists():
        with out.open(newline="") as f:
            for row in csv.reader(f):
                if row and row[0] != "room" and len(row) >= 6:
                    done[(row[0], row[1])] = dict(zip(fields, row))
    new = not out.exists() or out.stat().st_size == 0
    fh = out.open("a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(fields)
        fh.flush()
    rooms = [r for r in a.rooms.split(",") if r]
    scn = {r: find_scenario(r) for r in rooms}
    for r, s in scn.items():
        if s is None:
            print("!! scenario not found:", r)
    jobs = [(r, v, scn[r]) for r in rooms if scn[r]
            for v in VARIANTS if (r, v) not in done]
    print(len(jobs), "runs to do")

    def go(job):
        res = run(job)
        row = [job[0], job[1], f"{res['score']:.1f}", f"{res['area']:.1f}",
               f"{res['perimeter']:.1f}", f"{res['cpu']:.1f}"]
        w.writerow(row)
        fh.flush()
        print(f"{job[0]:12s} {job[1]:8s} score {res['score']:>10.1f}  area {res['area']:>10.1f}  cpu {res['cpu']:6.1f}s", flush=True)
        return job, row

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for job, row in ex.map(go, jobs):
            done[(job[0], job[1])] = dict(zip(fields, row))
    fh.close()

    print("\nEnclosed area as % of the base player's (score in brackets)")
    print(f"{'room':12s} " + " ".join(f"{v:>16s}" for v in VARIANTS))
    for r in rooms:
        base = done.get((r, "base"))
        if not base or float(base["area"]) == 0:
            print(f"{r:12s} (no base result)")
            continue
        cells = []
        for v in VARIANTS:
            x = done.get((r, v))
            if not x:
                cells.append(f"{'-':>16s}")
                continue
            pct = 100 * float(x["area"]) / float(base["area"])
            cells.append(f"{pct:5.0f}% ({float(x['score']):>7.0f})")
        print(f"{r:12s} " + " ".join(cells))


if __name__ == "__main__":
    main()
