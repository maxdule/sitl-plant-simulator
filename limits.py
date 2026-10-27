"""Limits of the detector outside the simulator's own assumptions (fault-free signal, no injected faults).

    python limits.py          # prints the tables and writes them to results/limits_<git commit>.txt

A) the spindle speed changes after calibration (the detector assumes f = 2.0 Hz is known and constant)
B) the temperature changes slowly after calibration (ambient or load), while the machine is healthy
Every run uses the same Detector / Alarms code as edge.py and counts the alarm raises
(alarms are acknowledged 10 s after the raise, as by the simulated operator in demo mode).
"""
import contextlib
import io
import math
from pathlib import Path

import numpy as np

import simulator
from edge import Alarms, Detector

FS = simulator.FS
CAL_END_S = 60                     # the change starts after the detector has calibrated (45 s) and settled


class VariableMachine(simulator.Machine):
    """Healthy machine whose speed and temperature may change after CAL_END_S."""

    def __init__(self, seed, speed_change=0.0, temp_rate_c_per_h=0.0, temp_sine_amp=0.0, temp_sine_period_h=24.0):
        super().__init__(seed, scheduled=False)
        self.speed_change, self.rate = speed_change, temp_rate_c_per_h / 3600.0
        self.sine_amp, self.sine_period = temp_sine_amp, temp_sine_period_h * 3600.0
        self.phase = 0.0

    def sample(self, k):
        t = k / FS
        after = max(0.0, t - CAL_END_S)
        f = simulator.FREQ_HZ * (1 + (self.speed_change if t >= CAL_END_S else 0.0))
        self.phase += 2 * math.pi * f / FS
        z_vib, z_temp = self.rng.standard_normal(2)
        drift = self.rate * after + self.sine_amp * math.sin(2 * math.pi * after / self.sine_period)
        return {"sim_t": t,
                "vibration_mm_s": simulator.AMPLITUDE * math.sin(self.phase) + simulator.NOISE_SIGMA * z_vib,
                "temperature_c": simulator.T_STEADY + drift + simulator.T_SIGMA * z_temp}, set()


def raises(machine, seconds):
    """Alarm raises as (class, sim_t), and the share of time after CAL_END_S with the variance condition on."""
    detector, alarms, out, ack_due = Detector(), Alarms(), [], {}
    variance_on = after = 0
    with contextlib.redirect_stdout(io.StringIO()):   # Detector prints its calibration result
        for k in range(1, int(seconds * FS)):
            p, _ = machine.sample(k)
            flags = detector.step(p["sim_t"], p["vibration_mm_s"], p["temperature_c"])
            if p["sim_t"] >= CAL_END_S:
                after += 1
                variance_on += detector.variance_on.state
            if flags is None:
                continue
            for c, transition in alarms.update(flags):
                if transition == "raise":
                    out.append((c, p["sim_t"]))
                    ack_due[c] = p["sim_t"] + 10.0
            for c in [c for c, due in ack_due.items() if p["sim_t"] >= due]:
                del ack_due[c]
                alarms.acknowledge(c)
    return out, variance_on / max(after, 1)


def main():
    seeds = range(3000, 3010)
    hours = len(seeds) * (600 - CAL_END_S) / 3600
    print(f"A) speed change after calibration, 10 min, healthy machine, {len(seeds)} seeds ({hours:.2f} h after the change)")
    print(f"{'speed change':>13}{'shock/h':>9}{'variance/h':>12}{'drift/h':>9}{'variance on':>13}  (share of time after the change)")
    for eps in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
        counts, share = {"shock": 0, "variance": 0, "drift": 0}, 0.0
        for s in seeds:
            r, on = raises(VariableMachine(s, speed_change=eps), 600)
            for c, _ in r:
                counts[c] += 1
            share += on / len(seeds)
        print(f"{eps * 100:>12.1f}%{counts['shock'] / hours:>9.1f}{counts['variance'] / hours:>12.1f}{counts['drift'] / hours:>9.1f}{share * 100:>12.0f}%")

    seeds = range(3000, 3005)
    print("\nB) slow temperature change after calibration, healthy machine, 5 seeds")
    print(f"{'change':>28}{'first drift alarm after':>26}{'deviation then':>16}")
    cases = [("ramp 0.1 C/h", dict(temp_rate_c_per_h=0.1), 3600),
             ("ramp 1 C/h", dict(temp_rate_c_per_h=1.0), 1800),
             ("daily sine, +-1 C", dict(temp_sine_amp=1.0), 3600)]
    for name, kw, dur in cases:
        delays, devs = [], []
        for s in seeds:
            m = VariableMachine(s, **kw)
            r, _ = raises(m, dur)
            first = next((t for c, t in r if c == "drift"), None)
            if first is not None:
                after = first - CAL_END_S
                delays.append(after)
                devs.append(m.rate * after + m.sine_amp * math.sin(2 * math.pi * after / m.sine_period))
        if delays:
            print(f"{name:>28}{np.median(delays):>20.0f} s ({len(delays)}/5){np.median(devs):>13.3f} C")
        else:
            print(f"{name:>28}{'none':>26}")
    print(f"\nsigma of the 1 Hz temperature mean: {simulator.T_SIGMA / math.sqrt(FS):.4f} C (simulated sensor)")


if __name__ == "__main__":
    from evaluate import git_info
    print("running (a few minutes) ...", flush=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    git = git_info()
    path = Path("results") / f"limits_{git['commit'][:7]}{'+dirty' if git['dirty'] else ''}.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue() + f"written to {path}")
