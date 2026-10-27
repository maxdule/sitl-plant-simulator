"""Edge node: residual-based fault detection, alarm latching and report-by-exception publishing.

Reads the raw 50 Hz stream, decides, and publishes only what matters:
  telemetry  5 s aggregates                      (QoS 0)
  alarms     one message per alarm transition    (QoS 1)
  state      displayed machine state, on change  (QoS 1, retained)
  snapshot   waveform 2 s before / 2 s after an alarm raise, one message per alarm (QoS 1)

Detection idea (model-based fault detection + SPC on the residual):
  vibration    the machine frequency is known, so over N = 50 samples (exactly 2 periods) a sine fit
               v ~ a sin + b cos + c is one DFT bin. Residual = v - fit; amplitude and noise are separate.
    shock      |residual| of the next sample, predicted from the previous window, against sigma0
    variance   chi-square test on the window residual: SSE / sigma0^2 ~ chi2(N - 3) when healthy
  temperature  CUSUM on the deviation from the baseline learned during calibration
Reference values (sigma0, temperature baseline) are learned once during the first CAL_S seconds and then frozen.
Cost per sample is O(N), N = 50.
"""
import collections
import json
import math
import os
import time

import numpy as np
import paho.mqtt.client as mqtt

BROKER = os.getenv("MQTT_BROKER", "mosquitto")
AUTO_ACK_S = 0.0 if os.getenv("DEMO_MODE", "auto") == "manual" else 10.0     # auto: simulated operator acknowledges after 10 s
ASSET = "legacy_cnc_01"
RAW = f"sim/{ASSET}/raw"
TELEMETRY, STATE, ALARMS, SNAPSHOT, STATUS, CMD_ACK = (
    f"factory/{ASSET}/{name}" for name in ("telemetry", "state", "alarms", "snapshot", "status", "cmd/ack"))

FS, N, OMEGA = 50, 50, 2 * math.pi * 2.0        # sample rate, window length (2 periods of 2 Hz), machine frequency
CAL_S = 45                                       # healthy calibration time in simulated seconds
K_SHOCK = 5.49          # z limit: per-sample false-alarm probability 1e-7 for a Gaussian residual, times sqrt(1 + 3/N)
MAX_ISOLATED = 5        # up to this many shock-sized residuals in a window are shocks; more means a noise regime
CHI2_LIMIT = 91.84      # scipy.stats.chi2.isf(1e-4, N - 3): false-alarm probability 1e-4 per window
CUSUM_K, CUSUM_H = 1.0, 6.0                      # in sigma of the 1 Hz temperature mean; a K of 0.5 gave false alarms because the
                                                 # baseline learned from 45 s is itself uncertain (chosen on the tune seeds, see evaluate.py)
AGG_SAMPLES, PRE, POST = 5 * FS, 2 * FS, 2 * FS

PRIORITY = ["shock", "variance", "drift"]        # display priority, highest first
LABEL = {"shock": "Acute Shock", "variance": "Variance Expansion", "drift": "Progressive Drift"}


class OnOff:
    """Debounce: switch on after `on` consecutive true inputs, off after `off` consecutive false ones."""

    def __init__(self, on, off):
        self.on, self.off, self.run, self.state = on, off, 0, False

    def update(self, raw):
        self.run = self.run + 1 if raw == (not self.state) else 0
        if self.run >= (self.off if self.state else self.on):
            self.state, self.run = not self.state, 0
        return self.state


class Detector:
    def __init__(self):
        self.s, self.c, self.v = np.zeros(N), np.zeros(N), np.zeros(N)   # sin, cos and value of the last N samples
        self.seen, self.fit = 0, None
        self.outliers = collections.deque(maxlen=N)          # which of the last N samples had a shock-sized residual
        self.cal_var, self.cal_temp = [], []
        self.sigma0 = None                       # set at the end of calibration, then frozen
        self.temp_sum, self.cusum, self.drift_flag = 0.0, 0.0, False
        self.variance_on, self.drift_on = OnOff(25, 50), OnOff(2, 5)
        self.feature = {"z": None, "amp_hat": None, "s_r": None, "c_plus": None}

    def step(self, t, vib, temp):
        """Process one sample. Returns {"shock": bool, "variance": bool, "drift": bool} or None while calibrating."""
        # 1) shock: predict this sample from the PREVIOUS window, so the spike cannot hide in its own statistics
        z = pred = None
        if self.fit is not None and self.sigma0:
            a, b, c0 = self.fit
            pred = a * math.sin(OMEGA * t) + b * math.cos(OMEGA * t) + c0
            z = abs(vib - pred) / (self.sigma0 * math.sqrt(1 + 3 / N))
        n_out = sum(self.outliers)
        shock = z is not None and z > K_SHOCK and n_out <= MAX_ISOLATED
        self.outliers.append(z is not None and z > K_SHOCK)

        # 2) append to the window and refit. A shock enters the window as its prediction, otherwise
        #    it would distort the fit and read as variance expansion for the next 50 samples.
        i = self.seen % N
        self.s[i], self.c[i], self.v[i] = math.sin(OMEGA * t), math.cos(OMEGA * t), pred if shock else vib
        self.seen += 1
        sse = amp = None
        if self.seen >= N:
            a, b = 2 / N * self.v @ self.s, 2 / N * self.v @ self.c      # orthogonal over whole periods
            c0 = self.v.mean()
            sse = float(np.sum((self.v - (a * self.s + b * self.c + c0)) ** 2))
            self.fit, amp = (a, b, c0), math.hypot(a, b)

        # 3) temperature at 1 Hz: block mean
        self.temp_sum += temp
        tick = self.seen % FS == 0
        temp_mean = None
        if tick:
            temp_mean, self.temp_sum = self.temp_sum / FS, 0.0

        # 4) still calibrating: learn the healthy reference values
        if self.sigma0 is None:
            if sse is not None:
                self.cal_var.append(sse / (N - 3))
            if temp_mean is not None:
                self.cal_temp.append(temp_mean)
            if self.seen >= CAL_S * FS:
                self.sigma0 = math.sqrt(np.mean(self.cal_var))
                self.t_base = float(np.mean(self.cal_temp))
                self.t_sigma = float(np.std(self.cal_temp))
                print(f"calibrated: sigma0={self.sigma0:.3f} mm/s, temperature baseline={self.t_base:.2f} C", flush=True)
            return None

        # 5) variance: chi-square on the window residual
        variance = self.variance_on.update(sse is not None and sse / self.sigma0 ** 2 > CHI2_LIMIT)

        # 6) drift: one-sided CUSUM on the 1 Hz temperature deviation, in units of the sensor sigma
        if tick:
            e = (temp_mean - self.t_base) / self.t_sigma
            self.cusum = min(CUSUM_H + 5, max(0.0, self.cusum + e - CUSUM_K))     # capped, so it can clear after a long drift
            self.drift_flag = self.drift_on.update(self.cusum > CUSUM_H)
        self.feature = {"z": z, "amp_hat": amp, "s_r": math.sqrt(sse / (N - 3)) if sse is not None else None,
                        "c_plus": self.cusum}
        return {"shock": shock, "variance": variance, "drift": self.drift_flag}


class Alarms:
    """ISA-18.2 states per alarm class: normal -> active -> acknowledged -> normal, or active -> returned -> normal."""

    def __init__(self):
        self.state = {c: "normal" for c in PRIORITY}
        self.n = 0
        self.ids = {}

    def update(self, flags):
        """Feed the detector flags; returns the list of transitions as (alarm_class, transition)."""
        out = []
        for c in PRIORITY:
            s, active = self.state[c], flags[c]
            if s == "normal" and active:
                self.n += 1
                self.ids[c] = f"{c}-{self.n}"
                self.state[c] = "active"                 # unacknowledged and active
                out.append((c, "raise"))
            elif s == "returned" and active:
                self.state[c] = "active"                 # came back before anyone acknowledged: same alarm, no new message
            elif s == "active" and not active:
                self.state[c] = "returned"               # gone, but the operator has not seen it yet
                out.append((c, "return"))
            elif s == "acked" and not active:
                self.state[c] = "normal"
                out.append((c, "clear"))
        return out

    def acknowledge(self, target):
        """Acknowledging never clears an alarm whose condition is still present."""
        out = []
        for c in PRIORITY:
            if target in ("all", c):
                if self.state[c] == "active":
                    self.state[c] = "acked"
                    out.append((c, "ack"))
                elif self.state[c] == "returned":
                    self.state[c] = "normal"
                    out.append((c, "clear"))
        return out

    def display(self):
        """Highest-priority alarm that is active or not yet acknowledged."""
        for c in PRIORITY:
            if self.state[c] != "normal":
                return LABEL[c]
        return "Normal"


class Edge:
    def __init__(self):
        self.detector, self.alarms = Detector(), Alarms()
        self.shown = None
        self.ring = collections.deque(maxlen=PRE)         # last 2 s of raw samples
        self.pending = []                                  # snapshots waiting for their 2 s after the alarm
        self.agg = {"v": [], "t": [], "amp": [], "s_r": [], "transit": [], "proc": []}
        self.ack_due = {}                                  # alarm_class -> sim_t of the simulated operator's ack
        self.last_seq, self.last_t = 0, 0.0
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edge")
        self.client.will_set(STATUS, "offline", qos=1, retain=True)
        self.client.reconnect_delay_set(1, 10)
        self.client.on_connect, self.client.on_message = self.on_connect, self.on_message

    def on_connect(self, client, userdata, flags, reason_code, properties):
        client.subscribe([(RAW, 0), (CMD_ACK, 1)])
        client.publish(STATUS, "online", qos=1, retain=True)

    def publish(self, topic, payload, qos, retain=False):
        self.client.publish(topic, payload if isinstance(payload, str) else json.dumps(payload), qos=qos, retain=retain)

    def on_message(self, client, userdata, msg):
        if msg.topic == CMD_ACK:                           # operator acknowledgment
            self.report(self.alarms.acknowledge(msg.payload.decode().strip() or "all"), self.last_seq, self.last_t, {})
            return
        t1 = time.time_ns()
        p = json.loads(msg.payload)
        t, vib, temp = p["sim_t"], p["vibration_mm_s"], p["temperature_c"]
        self.last_seq, self.last_t = p["seq"], t
        flags = self.detector.step(t, vib, temp)
        t2 = time.time_ns()
        self.ring.append((p["seq"], t, vib, temp))
        self.collect(p, t1, t2)
        for done in [s for s in self.pending if len(s["seq"]) >= s["want"]]:       # snapshot complete
            self.pending.remove(done)
            self.publish(SNAPSHOT, {k: done[k] for k in ("alarm_id", "alarm_class", "seq", "sim_t", "vibration_mm_s", "temperature_c")}, 1)
        for s in self.pending:
            s["seq"].append(p["seq"]); s["sim_t"].append(t); s["vibration_mm_s"].append(vib); s["temperature_c"].append(temp)
        if flags is None:
            return
        events = self.alarms.update(flags)
        for c in [c for c, due in self.ack_due.items() if t >= due]:              # simulated operator
            del self.ack_due[c]
            events += self.alarms.acknowledge(c)
        self.report(events, p["seq"], t, self.detector.feature)

    def report(self, events, seq, t, feature):
        """Publish alarm transitions and, if it changed, the displayed state. Nothing else leaves the edge per sample."""
        for c, transition in events:
            body = {"alarm_id": self.alarms.ids[c], "alarm_class": c, "transition": transition, "seq": seq, "sim_t": t}
            body.update({k: v for k, v in feature.items() if v is not None})
            self.publish(ALARMS, body, 1)
            if transition == "raise":
                rows = list(self.ring)                    # the 2 s before, including the triggering sample
                self.pending.append({"alarm_id": self.alarms.ids[c], "alarm_class": c, "want": len(rows) + POST,
                                     "seq": [r[0] for r in rows], "sim_t": [r[1] for r in rows],
                                     "vibration_mm_s": [r[2] for r in rows], "temperature_c": [r[3] for r in rows]})
                if AUTO_ACK_S > 0:
                    self.ack_due[c] = t + AUTO_ACK_S
        shown = self.alarms.display()
        if shown != self.shown:
            self.shown = shown
            self.publish(STATE, shown, 1, retain=True)

    def collect(self, p, t1, t2):
        """Window aggregates (5 s) incl. latency: transit = edge receive - generator send, processing = step() time."""
        a, f = self.agg, self.detector.feature
        a["v"].append(p["vibration_mm_s"]); a["t"].append(p["temperature_c"])
        a["transit"].append((t1 - p["t_send_ns"]) / 1e6); a["proc"].append((t2 - t1) / 1e6)
        for key, name in (("amp", "amp_hat"), ("s_r", "s_r")):
            if f[name] is not None:
                a[key].append(f[name])
        if len(a["v"]) >= AGG_SAMPLES:
            v, lat = np.array(a["v"]), np.array(a["transit"]) + np.array(a["proc"])
            self.publish(TELEMETRY, {
                "seq": p["seq"], "sim_t": p["sim_t"], "vib_rms": float(np.sqrt(np.mean(v ** 2))),
                "vib_peak": float(np.abs(v).max()), "amp_hat": float(np.mean(a["amp"])) if a["amp"] else 0.0,
                "s_r": float(np.mean(a["s_r"])) if a["s_r"] else 0.0,
                "temp_mean": float(np.mean(a["t"])), "temp_min": float(min(a["t"])), "temp_max": float(max(a["t"])),
                "latency_mean_ms": float(lat.mean()), "latency_p99_ms": float(np.percentile(lat, 99)),
                "latency_max_ms": float(lat.max()), "processing_mean_ms": float(np.mean(a["proc"])),
                "calibrated": 1.0 if self.detector.sigma0 else 0.0}, 0)
            for key in a:
                a[key] = []

    def run(self):
        self.client.connect_async(BROKER, 1883)
        self.client.loop_forever()                         # network loop; message handling stays in on_message


if __name__ == "__main__":
    Edge().run()
