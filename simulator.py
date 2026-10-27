"""Virtual CNC spindle (the "machine" side of the SITL setup).

Publishes raw samples at 50 Hz on MQTT, as an emulated fieldbus read. Three rules keep it honest:
  * time is simulated: sim_t = k / 50 exactly, the wall clock only paces the loop;
  * all randomness comes from one seeded generator, so the same SEED gives the same signal;
  * ground truth never goes into the sample payload. It is published on a separate topic.
"""
import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import paho.mqtt.client as mqtt

BROKER = os.getenv("MQTT_BROKER", "mosquitto")
SEED = int(os.getenv("SEED", "42"))
ASSET = "legacy_cnc_01"
RAW, INJECT, TRUTH = f"sim/{ASSET}/raw", f"sim/{ASSET}/cmd/inject", f"sim/{ASSET}/truth"

FS = 50                                   # samples per second
FREQ_HZ, AMPLITUDE, NOISE_SIGMA = 2.0, 2.0, 0.15      # vibration: 2 Hz sine in mm/s plus Gaussian noise
T_STEADY, T_SIGMA = 75.0, 0.05            # the machine is already warm; sensor noise in deg C

# Fault schedule as data: (type, start in cycle [s], duration [s], magnitude).
#   shock : half-sine impulse of this height in mm/s, added to the signal (independent of the sine phase)
#   noise : extra Gaussian noise, standard deviation in mm/s
#   drift : temperature ramp in deg C per second
# One healthy start, then the cycle repeats forever. Faults can also be injected on demand.
HEALTHY_START_S, CYCLE_S = 60, 120
SCHEDULE = [("shock", 10, 0.08, 6.0), ("noise", 30, 15, 1.0), ("drift", 60, 30, 0.15)]
INJECTED_DEFAULTS = {"shock": (0.08, 6.0), "noise": (15, 1.0), "drift": (30, 0.15)}


class Machine:
    def __init__(self, seed=SEED, scheduled=True):
        self.rng = np.random.default_rng(seed)
        self.scheduled = scheduled        # False: only injected faults (used by evaluate.py)
        self.injected = []                # (type, onset sample, number of samples, magnitude)

    def inject(self, kind, k, seconds=None, magnitude=None):
        default_seconds, default_magnitude = INJECTED_DEFAULTS[kind]
        seconds = default_seconds if seconds is None else seconds
        magnitude = default_magnitude if magnitude is None else magnitude
        self.injected.append((kind, k, max(1, round(seconds * FS)), magnitude))

    def active_faults(self, k):
        """All faults acting on sample k as (type, samples since onset, number of samples, magnitude)."""
        found = [(kind, k - onset, n, mag) for kind, onset, n, mag in self.injected if onset <= k < onset + n]
        start = HEALTHY_START_S * FS
        if self.scheduled and k >= start:
            local = (k - start) % (CYCLE_S * FS)
            for kind, at_s, seconds, mag in SCHEDULE:
                onset, n = round(at_s * FS), round(seconds * FS)
                if onset <= local < onset + n:
                    found.append((kind, local - onset, n, mag))
        return found

    def sample(self, k):
        """Return (payload for the edge, ground truth) for sample number k."""
        t = k / FS
        # every sample starts from the nominal parameters, then all active faults are applied
        sigma, kick, offset = NOISE_SIGMA, 0.0, 0.0
        for kind, i, n, mag in self.active_faults(k):
            if kind == "shock":
                kick += mag * math.sin(math.pi * (i + 1) / (n + 1))
            elif kind == "noise":
                sigma = math.hypot(sigma, mag)
            elif kind == "drift":
                offset += mag * (i + 1) / FS
        z_vib, z_temp = self.rng.standard_normal(2)
        payload = {
            "seq": k,
            "asset_id": ASSET,
            "sim_t": t,
            "ts_iso": datetime.now(timezone.utc).isoformat(),
            "t_send_ns": time.time_ns(),
            "vibration_mm_s": round(AMPLITUDE * math.sin(2 * math.pi * FREQ_HZ * t) + kick + sigma * z_vib, 4),
            "temperature_c": round(T_STEADY + offset + T_SIGMA * z_temp, 4),
        }
        truth = {kind for kind, _, _, _ in self.active_faults(k)}
        return payload, truth


def main():
    manual = os.getenv("DEMO_MODE", "auto") == "manual"     # manual: faults only when injected; auto: the schedule above
    machine = Machine(scheduled=not manual)
    k_now = 0

    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe(INJECT)

    def on_message(client, userdata, msg):
        kind = msg.payload.decode().strip()
        if kind in INJECTED_DEFAULTS and k_now < HEALTHY_START_S * FS:
            print(f"ignored {kind}: the edge node is still learning, try again after {HEALTHY_START_S} s", flush=True)
        elif kind in INJECTED_DEFAULTS:
            machine.inject(kind, k_now + 1)           # starts at the next sample
            print(f"injected {kind}", flush=True)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="simulator")
    client.on_connect, client.on_message = on_connect, on_message
    client.reconnect_delay_set(1, 10)
    client.connect_async(BROKER, 1883)
    client.loop_start()                               # network runs in its own thread, publish() never blocks
    time.sleep(3)                                     # give the edge node time to subscribe

    previous, t0 = set(), time.monotonic()
    print(f"simulator running, mode: {'manual' if manual else 'auto'}", flush=True)
    while True:
        k_now += 1
        k = k_now
        deadline = t0 + k / FS                        # absolute deadline: sleep jitter does not accumulate
        time.sleep(max(0.0, deadline - time.monotonic()))
        payload, truth = machine.sample(k)
        client.publish(RAW, json.dumps(payload), qos=0)
        for kind in truth ^ previous:                 # report ground-truth changes, not every sample
            event = "start" if kind in truth else "end"
            client.publish(TRUTH, json.dumps({"fault": kind, "event": event, "seq": k, "sim_t": payload["sim_t"]}), qos=1)
        previous = truth


if __name__ == "__main__":
    main()
