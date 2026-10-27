"""Figures from the data stored in InfluxDB. Runs on the host, not in a container.

    pip install matplotlib
    python figures.py --minutes 10            # writes every figure in Serbian (_sr) and English (_en)

Writes figures/<window>_<sr|en>_*.png and .pdf plus a delay table and a latency summary (CSV, language independent), where <window> is the UTC time of the
first data point in the selected range, so every file says which data it was made from.
Nothing is typed in by hand: every plotted value and every number comes from InfluxDB.
"""
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("INFLUX_URL", "http://localhost:8086")
import report  # noqa: E402  (reads INFLUX_URL on import)

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True, "grid.linestyle": ":", "lines.linewidth": 1.1})
COLOUR = {"shock": "#dc2626", "noise": "#d97706", "variance": "#d97706", "drift": "#2563eb"}
TEXT = {
    "en": {"time": "Time since start of range (s)", "vib": "Vibration (mm/s)", "temp": "Temperature (°C)",
           "lat": "Latency (ms)", "state": "Alarm raises", "shock": "Shock", "noise": "Noise burst", "drift": "Drift",
           "variance": "Variance", "rms": "RMS", "amp": "Estimated amplitude (2 Hz)", "peak": "Peak", "mean": "Mean",
           "min": "Min", "max": "Max", "p99": "P99", "proc": "Processing (mean)", "fault": "Injected fault",
           "t1": "Vibration, temperature and detection", "t2": "Waveform around alarm raises", "t3": "Latency generator to edge result (transit + processing), 5 s windows",
           "snap_x": "Time around the raise (s)"},
    "sr": {"time": "Vreme od početka opsega (s)", "vib": "Vibracije (mm/s)", "temp": "Temperatura (°C)",
           "lat": "Latencija (ms)", "state": "Podizanje alarma", "shock": "Udar", "noise": "Nalet šuma", "drift": "Drift",
           "variance": "Varijansa", "rms": "RMS", "amp": "Procenjena amplituda (2 Hz)", "peak": "Vršna vrednost", "mean": "Srednja vrednost",
           "min": "Min", "max": "Max", "p99": "P99", "proc": "Obrada (srednja)", "fault": "Injektovani otkaz",
           "t1": "Vibracije, temperatura i detekcija", "t2": "Talasni oblik oko podizanja alarma", "t3": "Latencija od generatora do rezultata na edge čvoru (transport + obrada), prozori od 5 s",
           "snap_x": "Vreme oko podizanja alarma (s)"},
}


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def series(rows, t0, *fields):
    """(x seconds since t0, {field: values}) for the rows that have all the fields."""
    ok = [r for r in rows if all(r.get(f) not in (None, "") for f in fields)]
    return [(parse_time(r["_time"]) - t0).total_seconds() for r in ok], {f: [float(r[f]) for r in ok] for f in fields}


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def timeline(data, t0, L, stem):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.3]})
    x, v = series(data["telemetry"], t0, "vib_rms", "amp_hat", "vib_peak")
    a1.plot(x, v["vib_peak"], color="#9ca3af", label=L["peak"])
    a1.plot(x, v["vib_rms"], color="#111827", label=L["rms"])
    a1.plot(x, v["amp_hat"], color="#16a34a", label=L["amp"])
    a1.set_ylabel(L["vib"]); a1.set_title(L["t1"], loc="left"); a1.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=3, frameon=False)
    x, v = series(data["telemetry"], t0, "temp_mean", "temp_min", "temp_max")
    a2.fill_between(x, v["temp_min"], v["temp_max"], color="#bfdbfe", label=f'{L["min"]} / {L["max"]}')
    a2.plot(x, v["temp_mean"], color="#1d4ed8", label=L["mean"])
    a2.set_ylabel(L["temp"]); a2.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False)
    # ground-truth spans (from the simulator's own start/end events)
    open_ = {}
    for r in data["truth"]:
        t = (parse_time(r["_time"]) - t0).total_seconds()
        if r["event"] == "start":
            open_[r["fault"]] = t
        elif r["fault"] in open_:
            for ax in (a1, a2, a3):
                ax.axvspan(open_[r["fault"]], max(t, open_[r["fault"]] + 0.8), color=COLOUR[r["fault"]], alpha=0.18, lw=0)
            del open_[r["fault"]]
    for i, cls in enumerate(("shock", "variance", "drift")):
        xs = [(parse_time(r["_time"]) - t0).total_seconds() for r in data["alarms"]
              if r["alarm_class"] == cls and r["transition"] == "raise"]
        a3.scatter(xs, [i] * len(xs), marker="v", color=COLOUR[cls], s=40)
    a3.set_yticks([0, 1, 2]); a3.set_yticklabels([L["shock"], L["variance"], L["drift"]]); a3.set_ylim(-0.7, 2.7)
    a3.set_ylabel(L["state"]); a3.set_xlabel(L["time"])
    fig.tight_layout()
    save(fig, stem)


def snapshots(data, L, stem):
    """Waveforms are stored one point per sample; split them into separate snapshots at gaps over 1 s."""
    pts = sorted(((parse_time(r["_time"]), r["alarm_class"], float(r["vibration_mm_s"])) for r in data["snapshot"] if r.get("vibration_mm_s")))
    groups = []
    for cls in sorted({p[1] for p in pts}):               # per class, because snapshots of different classes can overlap in time
        cur = []
        for p in (p for p in pts if p[1] == cls):
            if cur and (p[0] - cur[-1][0]).total_seconds() > 1.0:
                groups.append(cur); cur = []
            cur.append(p)
        groups.append(cur)
    # Show a snapshot that belongs to a real injected fault of that kind (not, for example, the shock alarm that a
    # noise burst raises at its start). Drift is left out: it is a temperature fault and the snapshot is vibration.
    starts = [(parse_time(r["_time"]), r["fault"]) for r in data["truth"] if r["event"] == "start"]

    def belongs(g, fault):
        mid = g[len(g) // 2][0]
        return any(f == fault and -1.0 <= (mid - t).total_seconds() <= 3.0 for t, f in starts)
    chosen = {}
    for cls, fault in (("shock", "shock"), ("variance", "noise")):
        found = [g for g in sorted(groups, key=lambda g: g[0][0]) if g[0][1] == cls and belongs(g, fault)]
        if found:
            chosen[cls] = found[-1]
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2.6 * len(chosen)), squeeze=False)
    for ax, (cls, g) in zip(axes[:, 0], sorted(chosen.items())):
        mid = g[len(g) // 2][0]
        ax.plot([(p[0] - mid).total_seconds() for p in g], [p[2] for p in g], color=COLOUR[cls])
        ax.set_title(f'{L.get(cls, cls)}  ({g[0][0]:%H:%M:%S} UTC)', loc="left", fontsize=9)
        ax.set_ylabel(L["vib"])
    axes[-1, 0].set_xlabel(L["snap_x"])
    fig.suptitle(L["t2"]); fig.tight_layout()
    save(fig, stem)


def latency(data, t0, L, stem):
    x, v = series(data["telemetry"], t0, "latency_mean_ms", "latency_p99_ms", "latency_max_ms", "processing_mean_ms")
    if not x:
        return
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(x, v["latency_max_ms"], color="#9ca3af", label=L["max"])
    ax.plot(x, v["latency_p99_ms"], color="#d97706", label=L["p99"])
    ax.plot(x, v["latency_mean_ms"], color="#111827", label=L["mean"])
    ax.plot(x, v["processing_mean_ms"], color="#16a34a", label=L["proc"])
    ax.axhline(20.0, color="darkred", ls=":", label="20 ms")
    ax.set_yscale("log"); ax.set_xlabel(L["time"]); ax.set_ylabel(L["lat"]); ax.set_title(L["t3"], fontsize=10); ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, stem)


def latency_csv(data, t0, stem):
    """The numbers behind the latency figure."""
    x, v = series(data["telemetry"], t0, "latency_mean_ms", "latency_p99_ms", "latency_max_ms", "processing_mean_ms")
    if not x:
        return
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["windows", "mean_of_window_means_ms", "highest_window_p99_ms", "max_ms", "windows_with_p99_over_20ms", "processing_mean_ms"])
        w.writerow([len(x), f'{sum(v["latency_mean_ms"]) / len(x):.3f}', f'{max(v["latency_p99_ms"]):.3f}', f'{max(v["latency_max_ms"]):.3f}',
                    sum(p > 20 for p in v["latency_p99_ms"]), f'{sum(v["processing_mean_ms"]) / len(x):.3f}'])
    print(f"wrote {stem}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=int, default=10, help="how far back to read from InfluxDB")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    data = {m: report.rows(m, a.minutes) for m in ("telemetry", "alarms", "truth", "snapshot")}
    if not data["telemetry"]:
        raise SystemExit("no telemetry in the selected range: is the stack running, and is --minutes large enough?")
    t0 = min(parse_time(r["_time"]) for r in data["telemetry"])
    Path(a.out).mkdir(exist_ok=True)
    base = f"{a.out}/{t0:%Y%m%dT%H%M%SZ}"
    for lang, L in TEXT.items():
        timeline(data, t0, L, f"{base}_{lang}_timeline")
        snapshots(data, L, f"{base}_{lang}_snapshots")
        latency(data, t0, L, f"{base}_{lang}_latency")
    latency_csv(data, t0, f"{base}_latency")
    stem = base
    found, extra = report.detections(a.minutes)
    with open(f"{stem}_delays.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fault", "start_sim_t_s", "alarm_class", "delay_s"])
        w.writerows([(f, f"{t:.2f}", c or "", "" if d is None else f"{d:.3f}") for f, t, c, d in found])
    print(f"wrote {stem}_delays.csv ({len(found)} faults, {len(extra)} unmatched alarm raises)")


if __name__ == "__main__":
    main()
