"""Run the machine and the detector in-process, at full speed, and print what was detected. No Docker, no MQTT.

    python selftest.py            # 10 simulated minutes, same code as the containers
"""
import sys

import simulator
from edge import Alarms, Detector

DURATION_S = 600
MAP = {"shock": "shock", "noise": "variance", "drift": "drift"}      # fault type -> alarm class

machine, detector, alarms = simulator.Machine(), Detector(), Alarms()
fault_start, raised, previous, ack_due, missed = {}, [], set(), {}, 0
for k in range(1, DURATION_S * simulator.FS):
    payload, truth = machine.sample(k)
    for kind in truth - previous:
        fault_start[(kind, k)] = payload["sim_t"]
    previous = truth
    flags = detector.step(payload["sim_t"], payload["vibration_mm_s"], payload["temperature_c"])
    if flags:
        for c, transition in alarms.update(flags):
            if transition == "raise":
                raised.append((c, payload["sim_t"]))
                ack_due[c] = payload["sim_t"] + 10            # simulated operator, as in edge.py
        for c in [c for c, due in ack_due.items() if payload["sim_t"] >= due]:
            del ack_due[c]
            alarms.acknowledge(c)

print(f"{'fault':<8}{'starts at':>10}{'alarm class':>14}{'delay':>9}")
matched = set()
for (kind, _), t0 in sorted(fault_start.items(), key=lambda x: x[1]):
    hit = next(((c, t) for c, t in raised if c == MAP[kind] and t >= t0 and t - t0 < 60), None)
    matched.add(hit)
    print(f"{kind:<8}{t0:>9.1f}s{MAP[kind]:>14}" + (f"{hit[1] - t0:>8.2f}s" if hit else "   missed"))
    missed = missed + (hit is None)
extra = [r for r in raised if r not in matched]
by_class = {c: sum(1 for r in extra if r[0] == c) for c in simulator and ("shock", "variance", "drift")}
print(f"alarm raises with no matching fault of that class: {by_class}")
print("(shock raises at a noise onset: until 6 large residuals are seen, a noise burst looks like isolated shocks)")
ok = missed == 0 and by_class["variance"] == 0 and by_class["drift"] == 0
print("SELFTEST", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
