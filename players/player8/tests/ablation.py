#!/usr/bin/env python3
"""Ablation and constant-sensitivity runs for the final player (Player 8).

Put this file in  players/player8/tests/  (any folder inside the repo works),
then run it from anywhere inside the repo:

    python players/player8/tests/ablation.py --groups phases --jobs 3
    python players/player8/tests/ablation.py --groups constants --jobs 3

What it does
  * For every (room, configuration) it starts a fresh Python process, switches
    one phase off or lowers one constant, runs the real player once and
    records the validated score and the CPU time.
  * Every phase of the player is bounded by work, not seconds, so scores do
    not depend on how busy the machine is (except on rooms that reach the
    290 s safety net: spiral, crown, 32_edge_concave_room, coms4444, 2a ...).
  * Results are appended to ablation_results.csv as they finish, so you can
    stop it (Ctrl-C) and start it again: finished runs are skipped.
  * At the end it prints, and writes to ablation_summary.md, a table of each
    configuration's score above 1,000 as a percentage of the full player's.

Use --jobs at most (number of cores - 1). Do not run other heavy work at the
same time.  Use --dry to test the script itself without running the player.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def find_root():
    p = Path(__file__).resolve()
    for d in [p.parent, *p.parents]:
        if (d / "main.py").exists() and (d / "src").is_dir():
            return d
    sys.exit("Could not find the repo root (a folder with main.py and src/).")


ROOT = find_root()

# --------------------------------------------------------------------------
# Rooms. Names are scenario file names (without .json); they are looked up
# anywhere under the repo, so the tournament rooms 1a, 2a, ... work too.
# --------------------------------------------------------------------------
DEFAULT_ROOMS = [
    "narrow", "hallway", "corridor", "u_corridor",   # corridors / thin rooms
    "i_room", "n_shaped_room",                        # axis-aligned
    "cool", "1a",                                     # diagonal / many-wall
    "3a",                                             # 100x100 square
]
# run to the 290 s net
HEAVY_ROOMS = ["spiral", "crown", "32_edge_concave_room"]

# --------------------------------------------------------------------------
# Configurations: "off" switches a phase off, "consts" overrides module
# constants of players/player8/player.py.
# --------------------------------------------------------------------------
PHASES = {
    "full": {},
    "no_grid": {"off": ["grid"]},
    "no_trace": {"off": ["trace"]},
    "no_bumps": {"off": ["bumps"]},
    "no_templates": {"off": ["templates"]},
    "no_restarts": {"off": ["restarts"]},
    "no_exact": {"consts": {"EXACT_LIMIT": 0}},          # sweep placement only
    "no_dominance": {"consts": {"DOMINANCE_OK": set()}},  # no family pruning
}

CONSTANTS = {
    "full": {},
    # iteration 3's value
    "seeds_4": {"consts": {"MAX_REFINE_SEEDS": 4}},
    "seeds_8": {"consts": {"MAX_REFINE_SEEDS": 8}},
    "pose_seeds_0": {"consts": {"POSE_SEEDS": 0}},
    "restarts_8k": {"consts": {"MAX_RESTARTS": 8_000, "IDLE_RESTARTS": 2_000}},
    "restarts_2k": {"consts": {"MAX_RESTARTS": 2_000, "IDLE_RESTARTS": 500}},
    "template_checks_500k": {"consts": {"TEMPLATE_CHECKS": 500_000}},
    "grids_15": {"consts": {"MAX_GRIDS": 15}},
    "exact_angles_8": {"consts": {"EXACT_MAX_ANGLES": 8}},
    "node_limit_400": {"consts": {"NODE_LIMIT": 400}},
    "relax_sides_12": {"consts": {"RELAX_MAX_SIDES": 12}},
}

GROUPS = {"phases": PHASES, "constants": CONSTANTS}


def find_scenario(name):
    hits = [
        p for p in ROOT.rglob(f"{name}.json")
        if ".git" not in p.parts and ".venv" not in p.parts
    ]
    if not hits:
        return None
    hits.sort(key=lambda p: len(p.parts))
    return hits[0]


# --------------------------------------------------------------------------
# Worker: one run in its own process
# --------------------------------------------------------------------------
def worker(scn_path, cfg):
    sys.path.insert(0, str(ROOT))
    import players.player8.player as P
    from src.enclosure import score_construction, validate_construction
    from src.scenario import read_scenario

    for k, v in cfg.get("consts", {}).items():
        if not hasattr(P, k):
            raise SystemExit(f"player.py has no constant {k}")
        setattr(P, k, v)
    off = {
        "grid": (P.GridEngine, "run", lambda self, *a, **k: None),
        "trace": (P.Player8, "_room_trace", lambda self: None),
        "bumps": (P.Player8, "_refine", lambda self: None),
        "templates": (P.Player8, "_templates", lambda self, work: None),
        "restarts": (P.Player8, "_explore", lambda self: None),
    }
    for what in cfg.get("off", []):
        cls, attr, stub = off[what]
        if not hasattr(cls, attr):
            raise SystemExit(f"player.py has no {cls.__name__}.{attr}")
        setattr(cls, attr, stub)

    sc = read_scenario(scn_path)
    t0 = time.process_time()
    p = P.Player8(sc.room.copy(), sc.inventory.copy(), sc.weights)
    c = p.build_enclosure()
    cpu = time.process_time() - t0
    if c is None:
        out = {"score": 0.0, "valid": True, "cpu": cpu, "note": "None"}
    else:
        r = validate_construction(c, sc.room, sc.inventory)
        out = {
            "score": score_construction(r, sc.weights) if r.valid else -1000.0,
            "valid": bool(r.valid),
            "cpu": cpu,
            "note": str(getattr(p, "note", ""))[:120],
        }
    print("RESULT " + json.dumps(out))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_one(job, dry):
    room, cfg_name, cfg, scn = job
    if dry:
        import random
        random.seed(hash((room, cfg_name)) & 0xFFFF)
        base = 1000 + 500 * (len(room) % 5 + 1)
        return {"score": base * (1 if cfg_name == "full" else random.choice([1, 1, 0.9, 0.5])),
                "valid": True, "cpu": 1.0, "note": "dry"}
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--scenario", str(scn), "--cfg", json.dumps(cfg, default=sorted)]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"score": float("nan"), "valid": False, "cpu": 900.0, "note": "timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"score": float("nan"), "valid": False, "cpu": 0.0,
            "note": "crash: " + (proc.stderr.strip().splitlines() or ["?"])[-1][:100]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="phases",
                    help="phases, constants or both: phases,constants")
    ap.add_argument("--rooms", default="",
                    help="comma list of scenario names (default: the 9 listed in the script)")
    ap.add_argument("--heavy", action="store_true",
                    help="also run spiral, crown, 32_edge_concave_room (each run can take ~290 s)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="ablation_results.csv")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--scenario")
    ap.add_argument("--cfg")
    a = ap.parse_args()

    if a.worker:
        worker(a.scenario, json.loads(a.cfg))
        return

    rooms = [r for r in a.rooms.split(",") if r] or list(DEFAULT_ROOMS)
    if a.heavy:
        rooms += HEAVY_ROOMS
    configs = {}
    for g in a.groups.split(","):
        for name, cfg in GROUPS[g].items():
            configs.setdefault(name, cfg)

    scn = {}
    for r in rooms:
        s = find_scenario(r)
        if s is None:
            print(f"!! scenario '{r}' not found, skipped")
        else:
            scn[r] = s
    rooms = list(scn)

    out = Path(a.out)
    done = {}
    if out.exists():
        with out.open() as f:
            for row in csv.DictReader(f):
                done[(row["room"], row["config"])] = row
    new = not out.exists()
    fh = out.open("a", newline="")
    w = csv.DictWriter(
        fh, fieldnames=["room", "config", "score", "valid", "cpu", "note"])
    if new:
        w.writeheader()

    jobs = [(r, n, c, scn[r]) for r in rooms for n, c in configs.items()
            if (r, n) not in done]
    print(f"{len(jobs)} runs to do, {len(done)} already in {out}")

    def go(job):
        res = run_one(job, a.dry)
        row = {"room": job[0], "config": job[1], "score": f"{res['score']:.1f}",
               "valid": res["valid"], "cpu": f"{res['cpu']:.1f}", "note": res["note"]}
        w.writerow(row)
        fh.flush()
        print(
            f"{job[0]:28s} {job[1]:22s} {res['score']:>10.1f}  {res['cpu']:6.1f}s  {res['note'][:60]}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for row in ex.map(go, jobs):
            done[(row["room"], row["config"])] = row
    fh.close()

    # summary: score above 1,000 as a percentage of the full player's
    names = list(configs)
    lines = ["| room | full score | " + " | ".join(n for n in names if n != "full") + " |",
             "|---|---|" + "---|" * (len(names) - 1)]
    changed = {n: 0 for n in names}
    tot = {n: [] for n in names}
    for r in rooms:
        full = done.get((r, "full"))
        if not full:
            continue
        fs = float(full["score"])
        cells = []
        for n in names:
            if n == "full":
                continue
            row = done.get((r, n))
            if not row or fs <= 1000:
                cells.append("-")
                continue
            s = float(row["score"])
            pct = 100 * (s - 1000) / (fs - 1000) if s == s else float("nan")
            cells.append(f"{pct:.0f}%")
            tot[n].append(pct)
            if abs(s - fs) > 0.5:
                changed[n] += 1
        lines.append(f"| {r} | {fs:.0f} | " + " | ".join(cells) + " |")
    lines.append("| **mean** | | " + " | ".join(
        f"{sum(v)/len(v):.0f}%" if v else "-" for n, v in tot.items() if n != "full") + " |")
    lines.append("| **rooms changed** | | " + " | ".join(
        str(changed[n]) for n in names if n != "full") + " |")
    text = "\n".join(lines)
    Path("ablation_summary.md").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
