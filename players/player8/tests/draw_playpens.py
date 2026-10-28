#!/usr/bin/env python3
"""Draw the enclosures our player builds (report items 5 and 6).

Put in  players/player8/tests/  and run from anywhere inside the repo.
Needs matplotlib (it is in a normal Anaconda "base"; otherwise: pip install matplotlib).

1) Final player on several rooms  ->  one picture, one panel per room
       python players/player8/tests/draw_playpens.py rooms --rooms narrow,hallway,i_room,1a --jobs 4

2) Iteration by iteration on one room  ->  one picture, one panel per iteration
   First save each iteration's player.py under its own name, for example
       git show <commit>:players/player8/player.py > iter1.py     (repeat for iter2, iter3)
   (the final player is players/player8/player.py), then
       python players/player8/tests/draw_playpens.py compare --room spiral ^
           --player "1: backtracking=iter1.py" --player "2: beam search=iter2.py" ^
           --player "3: phase pipeline=iter3.py" --player "4: final=players/player8/player.py" --jobs 4
   (in PowerShell write ` instead of ^ at the end of a line, or put it all on one line)

Each run prints the score and CPU time per panel, and writes
    playpens_rooms.png / playpens_compare.png   (300 dpi)
    plus one small PNG per panel in the folder  playpen_figs/
Drawing: the room is grey, the enclosure blue, gates are drawn thick in orange.
spiral and crown run to the 290 s safety net, so give them a free machine.
"""
import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def find_root():
    p = Path(__file__).resolve()
    for d in [p.parent, *p.parents]:
        if (d / "main.py").exists() and (d / "src").is_dir():
            return d
    sys.exit("Could not find the repo root (a folder with main.py and src/).")


ROOT = find_root()


def find_scenario(name):
    hits = [p for p in ROOT.rglob(f"{name}.json")
            if ".git" not in p.parts and ".venv" not in p.parts]
    hits.sort(key=lambda p: len(p.parts))
    return hits[0] if hits else None


def load_player_class(path):
    """Import a player file by path and return its Player class."""
    path = Path(path).resolve()
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        f"_pl_{path.stem}_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for name in ("Player8", "Player"):
        if hasattr(mod, name):
            return getattr(mod, name)
    for name in dir(mod):
        if name.startswith("Player") and name != "Player":
            return getattr(mod, name)
    raise SystemExit(f"no Player class found in {path}")


def run_task(task):
    """Run one player on one room. Returns what the drawing needs."""
    label, player_file, room_name = task
    sys.path.insert(0, str(ROOT))
    from src.enclosure import score_construction, validate_construction
    from src.scenario import read_scenario

    scn = find_scenario(room_name)
    if scn is None:
        return {"label": label, "room": room_name, "error": "scenario not found"}
    sc = read_scenario(str(scn))
    cls = load_player_class(player_file or ROOT /
                            "players" / "player8" / "player.py")
    t0 = time.process_time()
    p = cls(sc.room.copy(), sc.inventory.copy(), sc.weights)
    c = p.build_enclosure()
    cpu = time.process_time() - t0
    out = {"label": label, "room": room_name, "cpu": cpu,
           "room_pts": [tuple(map(float, q)) for q in sc.room.get_boundary_points()]}
    if c is None:
        out.update(score=0.0, verts=None, gates=[], note="no enclosure")
        return out
    r = validate_construction(c, sc.room, sc.inventory)
    if not r.valid:
        out.update(score=-1000.0, verts=None, gates=[],
                   note="invalid: " + r.reason)
        return out
    out.update(
        score=float(score_construction(r, sc.weights)),
        verts=[tuple(map(float, v)) for v in r.vertices],
        gates=[bool(pc.is_gate) for pc in c.pieces],
        area=float(r.area), perimeter=float(r.perimeter), note="",
    )
    return out


def draw_panel(ax, res, title=None):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or res["label"], fontsize=11)
    rp = res.get("room_pts")
    if rp:
        xs = [q[0] for q in rp] + [rp[0][0]]
        ys = [q[1] for q in rp] + [rp[0][1]]
        ax.fill(xs, ys, color="#e3e3e3", zorder=1)
        ax.plot(xs, ys, color="#555555", lw=1.2, zorder=2)
    v = res.get("verts")
    if not v:
        ax.text(0.5, 0.5, res.get("note") or res.get("error", ""), transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#aa0000")
        return
    n = len(v)
    xs = [q[0] for q in v] + [v[0][0]]
    ys = [q[1] for q in v] + [v[0][1]]
    ax.fill(xs, ys, color="#4a90e2", alpha=0.45, zorder=3)
    ax.plot(xs, ys, color="#1f5fae", lw=1.4, zorder=4)
    for i, is_gate in enumerate(res["gates"]):
        if is_gate:
            a, b = v[i], v[(i + 1) % n]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#f28e2b", lw=4, zorder=5,
                    solid_capstyle="butt")
    ax.text(0.5, -0.02, f"score {res['score']:,.0f}   area {res['area']:,.0f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)


def make_figure(results, ncols, out_png, panel_dir, title_fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        4.2 * ncols, 4.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    panel_dir.mkdir(exist_ok=True)
    for ax, res in zip(axes.ravel(), results):
        draw_panel(ax, res, title_fn(res))
        f2, a2 = plt.subplots(figsize=(5, 5))
        draw_panel(a2, res, title_fn(res))
        f2.savefig(panel_dir / f"{res['room']}_{res['label'].split(':')[0].strip().replace(' ', '_')}.png",
                   dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(f2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", out_png, "and per-panel files in", panel_dir)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("rooms")
    a.add_argument("--rooms", default="narrow,hallway,i_room,1a")
    a.add_argument("--player", default="",
                   help="player file (default: the final player)")
    a.add_argument("--jobs", type=int, default=2)
    a.add_argument("--cols", type=int, default=2)
    c = sub.add_parser(
        "redraw", help="redraw from a saved playpens_*.json without rerunning the player")
    c.add_argument("--json", required=True)
    c.add_argument("--cols", type=int, default=2)
    b = sub.add_parser("compare")
    b.add_argument("--room", default="spiral")
    b.add_argument("--player", action="append", required=True,
                   help='"label=path/to/player.py" (repeat)')
    b.add_argument("--jobs", type=int, default=2)
    b.add_argument("--cols", type=int, default=2)
    args = ap.parse_args()

    if args.mode == "redraw":
        import json
        saved = json.loads(Path(args.json).read_text())
        mode = saved["mode"]
        title = (lambda res: res["room"]) if mode == "rooms" else (
            lambda res: res["label"])
        make_figure(saved["results"], args.cols, Path(
            f"playpens_{mode}.png"), Path("playpen_figs"), title)
        return

    if args.mode == "rooms":
        tasks = [("final player", args.player or None, r)
                 for r in args.rooms.split(",") if r]

        def title(res): return res["room"]
        out_png = Path("playpens_rooms.png")
    else:
        tasks = []
        for spec in args.player:
            label, _, path = spec.partition("=")
            tasks.append((label.strip(), path.strip(), args.room))

        def title(res): return res["label"]
        out_png = Path("playpens_compare.png")

    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results = list(ex.map(run_task, tasks))
    for r in results:
        print(f"{r['room']:24s} {r['label']:24s} score {r.get('score', float('nan')):>10.1f}  "
              f"cpu {r.get('cpu', 0):6.1f}s  {r.get('note', r.get('error', ''))}")
    import json
    Path(f"playpens_{args.mode}.json").write_text(
        json.dumps({"mode": args.mode, "results": results}))
    print(
        f"saved playpens_{args.mode}.json (redraw later with:  draw_playpens.py redraw --json playpens_{args.mode}.json)")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("matplotlib is missing in THIS python. Run:  python -m pip install matplotlib   "
                 f"then:  python players/player8/tests/draw_playpens.py redraw --json playpens_{args.mode}.json")
    make_figure(results, args.cols, out_png, Path("playpen_figs"), title)


if __name__ == "__main__":
    main()
