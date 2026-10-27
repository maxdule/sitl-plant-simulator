# SITL demo: legacy CNC spindle with an edge fault detector

`simulator.py` (virtual machine, 50 Hz) → MQTT → `edge.py` (detection, alarms, buffering) → MQTT → Node-RED → InfluxDB → Grafana.
Only these two Python files contain logic (about 400 lines). Everything else is configuration.

## 1. Launch (one command, needs only Docker)

```bash
docker compose up -d --build        # background, keeps the terminal free
docker compose logs -f edge simulator   # optional: watch the two Python services (Ctrl+C leaves them running)
```

First start downloads the images (a few minutes). The machine learns "normal" for the first 45 s and stays healthy until 60 s.
After that, what happens depends on the one setting in `demo.env`:

| `DEMO_MODE` | Faults | Acknowledging alarms |
|---|---|---|
| `manual` (for presenting) | only when you inject them (section 2b) | you, with the Node-RED button |
| `auto` (default; for figures and unattended runs) | every 120 s: **shock** (+10 s), **noise burst** (+30 s, 15 s long), **thermal drift** (+60 s, 30 s long) | a simulated operator, 10 s after each alarm |

To switch: edit `demo.env`, then `docker compose up -d --force-recreate simulator edge`.

## 2. Verify the data

| Where | What you should see |
|---|---|
| Grafana <http://localhost:3000> (admin / admin), dashboard *SITL demo* | Machine state tile switches Normal → Acute Shock → Variance Expansion → Progressive Drift; vibration and temperature aggregates every 5 s; snapshot waveform around each alarm; alarm and ground-truth tables |
| InfluxDB <http://localhost:8086> (admin / adminpassword), bucket `factory` | measurements `telemetry`, `alarms`, `state`, `snapshot`, `truth` |
| Terminal | commands below |

```bash
# raw edge output (one line per message; alarms appear only on transitions)
docker compose exec mosquitto mosquitto_sub -t "factory/#" -v

# alarms stored in InfluxDB
docker compose exec influxdb influx query --org sitl --token sitl-token \
  'from(bucket:"factory") |> range(start:-10m) |> filter(fn:(r) => r._measurement=="alarms" and r._field=="transition")'

# detection delay of every injected fault (ground truth vs alarm raise), computed from InfluxDB
docker compose exec edge python report.py
```

Expected from `report.py`: shock delay 0.00 s, noise about 0.7 s, drift about 2 s. One extra shock alarm per noise burst is a known limitation (see below).

### 2b. Injecting faults and acknowledging by hand (`DEMO_MODE=manual`)

Easiest: open Node-RED at <http://localhost:1880> and click the button at the left of **Inject shock**, **Inject noise expansion**, **Inject thermal drift** or **Acknowledge all alarms**.
The same from a terminal:

```bash
docker compose exec mosquitto mosquitto_pub -t sim/legacy_cnc_01/cmd/inject -m shock    # or noise, drift
docker compose exec mosquitto mosquitto_pub -t factory/legacy_cnc_01/cmd/ack -m all
```

Injections before the 60 s mark are ignored (the simulator log says so). A fault is injected for a fixed duration (shock 0.08 s, noise 15 s, drift 30 s; see `INJECTED_DEFAULTS` in `simulator.py`). The alarm stays on the dashboard until you acknowledge it and the condition has cleared.

## 3. Shut down

```bash
docker compose down -v      # stops everything and deletes the stored data
```

If Grafana shows "No data" or InfluxDB logs `authorization not found`, an old InfluxDB volume with another token is being reused: run `docker compose down -v` (also in any older checkout of this project) and start again.

## Figures

```bash
pip install matplotlib
python figures.py --minutes 10      # --minutes = how far back to read
```

Reads InfluxDB (<http://localhost:8086>) and writes every figure in Serbian (`_sr`) and English (`_en`) to `figures/` (git-ignored): a timeline (vibration, temperature, injected faults, alarm raises), the waveform around alarm raises, edge latency, and a CSV with the detection delay of every fault. Each filename starts with the UTC time of the first data point it was made from. Let the demo run for at least one full cycle (about 4 minutes) first.

## Final evaluation numbers

Detection rates, delays and false alarms come from many randomized runs of the same code, not from the live demo:

```bash
python selftest.py                    # quick check, exit code 0 = ok
python evaluate.py --seeds tune       # while adjusting thresholds (seeds 1000-1099)
git commit -am "final detector"       # the report run must belong to a commit
python evaluate.py --seeds report     # the reported numbers (seeds 2000-2099): run once, takes a few minutes
python limits.py                      # behaviour outside the simulator's assumptions (speed change, slow temperature change)
git tag -a 1.0-release -m "..."       # cite this tag in the report
```

`evaluate.py` writes `results/` (git-ignored): detection rate with 95 % interval, delay percentiles, wrong-class alarms, false alarms per hour on 8 h of fault-free signal, a sensitivity figure, and an info file with the git commit, versions and seeds. The experiment design and the pass/fail criteria are constants at the top of the file, fixed before the report run. `limits.py` writes its two tables to `results/limits_<commit>.txt`. Latency comes from the live demo: `python figures.py` also writes a latency summary CSV.

## Reproducing every result from a clean checkout

Windows with Docker Engine in WSL2 (Ubuntu); the Python steps run on Windows, Docker in WSL. Any file whose name carries `+dirty` was made from uncommitted code and must not be cited.

```powershell
git clone https://github.com/maxdule/sitl-plant-simulator; cd sitl-plant-simulator; git checkout 1.0-release
py -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt matplotlib
python selftest.py                    # 1. sanity check
python evaluate.py --seeds report     # 2. results/report_<commit>_*  (tables, sensitivity figure sr/en, info.json)
python limits.py                      # 3. results/limits_<commit>.txt
wsl -d Ubuntu-24.04 -- docker compose up -d --build     # 4. live stack (DEMO_MODE=auto), run from this folder
# wait at least 10 minutes (calibration, then four 120 s fault cycles)
python figures.py --minutes 10        # 5. figures/<UTC time>_{sr,en}_* plus latency and delay CSVs
wsl -d Ubuntu-24.04 -- docker compose down -v           # 6. stop and delete the demo data
```

## Without Docker

`pip install -r requirements.txt && python selftest.py` runs the same machine and detector in-process for 10 simulated minutes and prints the detection table.

## What to know

* The same seed gives the same signal (`SEED`, default 42). Ground truth is published on `sim/.../truth` and never in the sample payload.
* Detector: sine fit over whole periods (one DFT bin) gives amplitude and a residual; shock = residual of the next sample, variance = chi-square on the window residual, drift = CUSUM on temperature. Reference values are learned once in the calibration window.
* Simplifications for the demo: the machine starts warm (no warm-up model), no broker authentication or TLS, demo credentials in the compose file.
* Known limitation: at the start of a strong noise burst the first large residuals look like isolated shocks, so one short shock alarm is raised before the variance alarm.
