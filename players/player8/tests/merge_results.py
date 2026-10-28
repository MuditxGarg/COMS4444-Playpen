"""Merge ablation CSVs into one clean file, all_results.csv.

    python players/player8/tests/merge_results.py ablation_results.csv phases_a.csv phases_b.csv heavy.csv

Wildcards also work (PowerShell does not expand them, so the script does):
    python players/player8/tests/merge_results.py "*.csv"

It drops the fake rows left by --dry runs (note == "dry"), copes with a CSV
whose header line is missing, keeps one row per (room, config), and prints how
many real rows each file contributed.
"""
import csv
import glob
import sys

FIELDS = ["room", "config", "score", "valid", "cpu", "note"]

paths = []
for arg in sys.argv[1:]:
    paths += sorted(glob.glob(arg)) or [arg]
paths = [p for p in paths if p != "all_results.csv"]

rows = {}
for path in paths:
    n = 0
    try:
        with open(path, newline="") as f:
            lines = list(csv.reader(f))
    except FileNotFoundError:
        print(f"  {path}: not found, skipped")
        continue
    if lines and lines[0] and lines[0][0] == "room":
        lines = lines[1:]  # header present
    for vals in lines:
        if len(vals) < 5:
            continue
        vals = (vals + [""] * 6)[:6]
        r = dict(zip(FIELDS, vals))
        if r["note"].strip() == "dry":
            continue
        rows[(r["room"], r["config"])] = r
        n += 1
    print(f"  {path}: {n} real rows")
bad = [k for k, r in rows.items()
       if r["score"] == "nan" or r["note"].startswith(("crash", "timeout"))]
with open("all_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows.values())
print(f"wrote all_results.csv with {len(rows)} rows")
if bad:
    print("rows that crashed or timed out (rerun their command):", sorted(bad))
