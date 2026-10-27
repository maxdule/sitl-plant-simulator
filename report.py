"""Detection delay per injected fault, computed from what is stored in InfluxDB (ground truth vs alarm raises).

    docker compose exec edge python report.py          # from the host (uses the container's network)
    INFLUX_URL=http://localhost:8086 python report.py  # or directly, standard library only

Delays are in simulated seconds (sim_t of the alarm raise minus sim_t of the fault start).
"""
import csv
import io
import os
import urllib.request

URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
TOKEN = "sitl-token"
ALARM_FOR = {"shock": "shock", "noise": "variance", "drift": "drift"}     # injected fault -> alarm class
WINDOW_S = 60                                                            # an alarm counts if it comes within this time


def rows(measurement, minutes=360):
    """All points of one measurement as a list of dicts (one per timestamp), oldest first."""
    flux = (f'from(bucket: "factory") |> range(start: -{int(minutes)}m) |> filter(fn: (r) => r._measurement == "{measurement}") '
            '|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value") |> group() |> sort(columns: ["_time"])')
    req = urllib.request.Request(f"{URL}/api/v2/query?org=sitl", data=flux.encode(), method="POST", headers={
        "Authorization": f"Token {TOKEN}", "Accept": "application/csv", "Content-Type": "application/vnd.flux"})
    text = urllib.request.urlopen(req, timeout=30).read().decode()
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    return [r for r in csv.DictReader(io.StringIO("\n".join(lines))) if r.get("_time") != "_time"]


def detections(minutes=360):
    """One entry per injected fault: (fault, start sim_t, alarm class or None, delay in s or None), plus unmatched raises."""
    starts = [(r["fault"], float(r["sim_t"])) for r in rows("truth", minutes) if r["event"] == "start"]
    raises = [(r["alarm_class"], float(r["sim_t"])) for r in rows("alarms", minutes) if r["transition"] == "raise"]
    result, used = [], set()
    for fault, t0 in starts:
        hit = next((a for a in raises if a[0] == ALARM_FOR[fault] and t0 <= a[1] < t0 + WINDOW_S), None)
        used.add(hit)
        result.append((fault, t0, ALARM_FOR[fault] if hit else None, hit[1] - t0 if hit else None))
    return result, [a for a in raises if a not in used]


if __name__ == "__main__":
    found, extra = detections()
    print(f"{'fault':<8}{'start [s]':>10}{'alarm':>10}{'delay [s]':>11}")
    for fault, t0, alarm, delay in found:
        print(f"{fault:<8}{t0:>10.1f}{(alarm or '-'):>10}{(f'{delay:.2f}' if delay is not None else 'missed'):>11}")
    print()
    for fault in ALARM_FOR:
        d = [x[3] for x in found if x[0] == fault and x[3] is not None]
        if d:
            print(f"{fault}: detected {len(d)}/{sum(1 for x in found if x[0] == fault)}, "
                  f"mean delay {sum(d) / len(d):.2f} s, max {max(d):.2f} s")
    print(f"alarm raises not matched to a fault of that class: {len(extra)}")
