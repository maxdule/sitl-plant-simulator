"""Offline evaluation: many runs, random faults, the same Machine / Detector / Alarms code as the containers.

    python evaluate.py --seeds tune      # while checking thresholds (seeds 1000-1099)
    python evaluate.py --seeds report    # the reported numbers (seeds 2000-2099): run once, on the final commit

Writes results/<seeds>_<git commit>_*.csv, a sensitivity figure and an info file (commit, versions, seeds, criteria).
Everything printed or written here is computed from the simulation; nothing is typed in by hand.
"""
import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

import simulator
from edge import Alarms, Detector

SEED_SETS = {"tune": range(1000, 1100), "report": range(2000, 2100)}     # disjoint on purpose
FS = simulator.FS
ALARM_FOR = {"shock": "shock", "noise": "variance", "drift": "drift"}    # injected fault -> alarm class

# Experiment design, fixed before the reported run.
NOMINAL_RUNS, NOMINAL_HOURS = 2, 4                       # false alarms are counted on 8 h of fault-free signal
RUN_S, SLOT_S, FIRST_SLOT_S = 1500, 150, 100             # one run: 9 faults, one per 150 s slot, none during calibration
FAULTS_PER_TYPE = 3
MAGNITUDE = {"shock": (0.3, 8.0), "noise": (0.05, 1.5), "drift": (0.002, 0.5)}   # log-uniform: mm/s, extra sigma mm/s, C/s
DURATION_S = {"shock": 0.08, "noise": 15.0, "drift": 120.0}
D_MAX_S = {"shock": 0.5, "noise": 5.0, "drift": 100.0}   # an alarm of the right class must be raised within this time
ACK_S = 10.0                                             # simulated operator, as in edge.py

# Acceptance criteria, fixed before the reported run. "Design band" = faults at least this large.
DESIGN_BAND = {"shock": 3.0, "noise": 0.5, "drift": 0.02}
CRITERIA = {
    "false alarms per hour (nominal)": ("<=", 0.5),
    "shock detection rate (design band)": (">=", 0.95),
    "noise detection rate (design band)": (">=", 0.95),
    "drift detection rate (design band)": (">=", 0.95),
    "shock median delay [s]": ("<=", 0.1),
    "noise median delay [s]": ("<=", 3.0),
    "drift median delay [s]": ("<=", 30.0),
}


def run(machine, seconds):
    """Run Machine -> Detector -> Alarms for `seconds`; return the alarm raises as (class, sim_t)."""
    detector, alarms, raises, ack_due = Detector(), Alarms(), [], {}
    for k in range(1, int(seconds * FS)):
        p, _ = machine.sample(k)
        flags = detector.step(p["sim_t"], p["vibration_mm_s"], p["temperature_c"])
        if flags is None:
            continue
        for c, transition in alarms.update(flags):
            if transition == "raise":
                raises.append((c, p["sim_t"]))
                ack_due[c] = p["sim_t"] + ACK_S
        for c in [c for c, due in ack_due.items() if p["sim_t"] >= due]:
            del ack_due[c]
            alarms.acknowledge(c)
    return raises


def faulty_run(seed):
    """One run with 9 randomly placed faults. Returns one dict per fault."""
    rng = np.random.default_rng(seed)
    kinds = list(ALARM_FOR) * FAULTS_PER_TYPE
    rng.shuffle(kinds)
    machine = simulator.Machine(seed, scheduled=False)
    faults = []
    for i, kind in enumerate(kinds):
        onset = FIRST_SLOT_S + i * SLOT_S + rng.uniform(0, 20)
        lo, hi = MAGNITUDE[kind]
        mag = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        onset_k = round(onset * FS)                      # the fault starts on a sample: compare against exactly that time
        machine.inject(kind, onset_k, DURATION_S[kind], mag)
        faults.append({"seed": seed, "type": kind, "onset_s": onset_k / FS, "magnitude": mag})
    raises = run(machine, RUN_S)
    for f in faults:
        end = f["onset_s"] + DURATION_S[f["type"]]
        mine = [t for c, t in raises if c == ALARM_FOR[f["type"]] and f["onset_s"] <= t < f["onset_s"] + D_MAX_S[f["type"]]]
        f["detected"] = bool(mine)
        f["delay_s"] = round(min(mine) - f["onset_s"], 3) if mine else None
        f["wrong_class"] = any(c != ALARM_FOR[f["type"]] and f["onset_s"] <= t < end for c, t in raises)
    guard = [(f["onset_s"], f["onset_s"] + DURATION_S[f["type"]] + 30) for f in faults]
    unexplained = [r for r in raises if not any(a <= r[1] < b for a, b in guard)]
    return faults, len(unexplained)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def git_info():
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"commit": git("rev-parse", "HEAD") or "unknown", "dirty": bool(git("status", "--porcelain", "--", ":!results", ":!figures"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", choices=SEED_SETS, required=True)
    ap.add_argument("--runs", type=int, help="use only the first N seeds (quick check)")
    a = ap.parse_args()
    seeds = list(SEED_SETS[a.seeds])[: a.runs]
    git = git_info()
    tag = f"{a.seeds}_{git['commit'][:7]}{'+dirty' if git['dirty'] else ''}"
    out = Path("results")
    out.mkdir(exist_ok=True)

    print(f"nominal: {NOMINAL_RUNS} x {NOMINAL_HOURS} h ...", flush=True)
    nominal_raises = {c: 0 for c in ALARM_FOR.values()}
    for i in range(NOMINAL_RUNS):
        for c, _ in run(simulator.Machine(seeds[i] + 10000, scheduled=False), NOMINAL_HOURS * 3600):
            nominal_raises[c] += 1
    hours = NOMINAL_RUNS * NOMINAL_HOURS
    fa_per_hour = sum(nominal_raises.values()) / hours

    events, unexplained = [], 0
    for n, seed in enumerate(seeds, 1):
        f, u = faulty_run(seed)
        events += f
        unexplained += u
        print(f"\rfault runs: {n}/{len(seeds)}", end="", flush=True)
    print()

    with open(out / f"{tag}_events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(events[0]))
        w.writeheader()
        w.writerows(events)

    rows, stats = [], {}
    for kind in ALARM_FOR:
        ev = [e for e in events if e["type"] == kind]
        det = [e for e in ev if e["detected"]]
        band = [e for e in ev if e["magnitude"] >= DESIGN_BAND[kind]]
        delays = [e["delay_s"] for e in det]
        lo, hi = wilson(len(det), len(ev))
        stats[kind] = {
            "rate_band": sum(e["detected"] for e in band) / len(band) if band else float("nan"),
            "median_delay": float(np.median(delays)) if delays else float("nan")}
        rows.append({"fault": kind, "alarm_class": ALARM_FOR[kind], "events": len(ev), "detected": len(det),
                     "detection_rate": round(len(det) / len(ev), 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                     "design_band_min": DESIGN_BAND[kind], "design_band_events": len(band),
                     "design_band_rate": round(stats[kind]["rate_band"], 4),
                     "delay_median_s": round(stats[kind]["median_delay"], 3),
                     "delay_p95_s": round(float(np.percentile(delays, 95)), 3) if delays else "",
                     "delay_max_s": max(delays) if delays else "",
                     "wrong_class_alarm_rate": round(sum(e["wrong_class"] for e in ev) / len(ev), 4)})
    with open(out / f"{tag}_detection.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / f"{tag}_false_alarms.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["alarm_class", "raises_in_nominal_run", "nominal_hours", "per_hour"])
        for c, n in nominal_raises.items():
            w.writerow([c, n, hours, round(n / hours, 4)])
        w.writerow(["total", sum(nominal_raises.values()), hours, round(fa_per_hour, 4)])

    measured = {"false alarms per hour (nominal)": fa_per_hour}
    for kind in ALARM_FOR:
        measured[f"{kind} detection rate (design band)"] = stats[kind]["rate_band"]
        measured[f"{kind} median delay [s]"] = stats[kind]["median_delay"]
    verdicts = {}
    print(f"\n{'criterion':<40}{'measured':>10}{'required':>12}  verdict")
    for name, (op, limit) in CRITERIA.items():
        v = measured[name]
        ok = (v <= limit) if op == "<=" else (v >= limit)
        verdicts[name] = {"measured": v, "requirement": f"{op} {limit}", "pass": bool(ok)}
        print(f"{name:<40}{v:>10.3f}{op + ' ' + str(limit):>12}  {'PASS' if ok else 'FAIL'}")
    print(f"\nwrong-class alarm during a fault, per type: " + ", ".join(f"{r['fault']} {r['wrong_class_alarm_rate']:.2f}" for r in rows))
    print(f"alarm raises outside every fault window in the fault runs: {unexplained}")

    info = {"seeds": a.seeds, "seed_range": [seeds[0], seeds[-1]], "git": git, "python": sys.version.split()[0],
            "numpy": np.__version__, "platform": platform.platform(),
            "design": {"nominal_hours": hours, "run_s": RUN_S, "slot_s": SLOT_S, "faults_per_type": FAULTS_PER_TYPE,
                       "magnitude": MAGNITUDE, "duration_s": DURATION_S, "d_max_s": D_MAX_S, "design_band": DESIGN_BAND},
            "criteria": verdicts, "unexplained_raises": unexplained}
    (out / f"{tag}_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")
    if git["dirty"]:
        print("warning: uncommitted changes: commit before the reported run so the numbers belong to a commit")

    try:
        for lang in LABELS:                                   # Serbian and English versions
            sensitivity_figure(events, out / f"{tag}_sensitivity_{lang}", lang)
    except ImportError:
        print("matplotlib not installed: tables written, figure skipped (pip install matplotlib)")
    print(f"written to {out}/{tag}_*")


LABELS = {
    "sr": {"title": {"shock": "Udar", "noise": "Nalet šuma", "drift": "Drift"},
           "x": {"shock": "Visina udara B (mm/s)", "noise": "Dodatna sigma šuma (mm/s)", "drift": "Brzina drifta (°C/s)"},
           "y": "Verovatnoća detekcije", "band": "projektni opseg"},
    "en": {"title": {"shock": "Shock", "noise": "Noise burst", "drift": "Drift"},
           "x": {"shock": "Shock magnitude (mm/s)", "noise": "Extra noise sigma (mm/s)", "drift": "Drift rate (deg C/s)"},
           "y": "Detection probability", "band": "design band"},
}
TICKS = {"shock": [0.3, 0.5, 1, 2, 4, 8], "noise": [0.05, 0.1, 0.2, 0.5, 1, 1.5], "drift": [0.002, 0.005, 0.01, 0.05, 0.1, 0.5]}


def sensitivity_figure(events, stem, lang="sr"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    colour = {"shock": "#dc2626", "noise": "#d97706", "drift": "#2563eb"}
    T = LABELS[lang]
    plt.rcParams.update({"font.family": "serif"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, kind in zip(axes, ALARM_FOR):
        ev = [e for e in events if e["type"] == kind]
        mag = np.array([e["magnitude"] for e in ev])
        hit = np.array([e["detected"] for e in ev])
        edges = np.geomspace(mag.min() * 0.999, mag.max() * 1.001, 9)
        idx = np.clip(np.digitize(mag, edges) - 1, 0, 7)
        x, y, lo, hi = [], [], [], []
        for b in range(8):
            sel = idx == b
            if sel.any():
                p, (l, h) = hit[sel].mean(), wilson(int(hit[sel].sum()), int(sel.sum()))
                x.append(math.sqrt(edges[b] * edges[b + 1])); y.append(p); lo.append(max(0.0, p - l)); hi.append(max(0.0, h - p))
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=2, color=colour[kind])
        ax.axvline(DESIGN_BAND[kind], color="gray", ls=":", label=T["band"])
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.05); ax.grid(True, ls=":")
        ax.set_xticks(TICKS[kind]); ax.set_xticklabels([str(v) for v in TICKS[kind]])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(T["title"][kind]); ax.set_xlabel(T["x"][kind])
    axes[0].set_ylabel(T["y"]); axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
