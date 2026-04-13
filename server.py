"""
Predictive Maintenance Simulator — FastAPI Server

Endpoints (matching problem statement):
  GET  /stream/{machine_id}         SSE: 1 reading/sec
  GET  /history/{machine_id}        Last 7 days of history (JSON)
  POST /alert                       Raise a maintenance alert
  POST /schedule-maintenance        Book a maintenance slot (bonus)

Control endpoints (for the GUI panel):
  GET  /status                      All machines current state
  POST /inject-failure              Inject a failure mode into a machine
  POST /clear-failure               Clear failure from a machine
  GET  /alerts                      All raised alerts
  GET  /scheduled                   All scheduled maintenance slots
  GET  /control-panel               Serve the HTML control panel
"""

import os
import json
import asyncio
import random
import math
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from physics import MACHINE_PROFILES, MACHINE_IDS, apply_physics, FAILURE_MODES

# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(title="Predictive Maintenance Simulator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ────────────────────────────────────────────────────────────

# Per-machine failure state
# { machine_id: { "mode": str, "progress": float 0–1, "start_time": float } }
failure_state: dict = {m: None for m in MACHINE_IDS}

# Alert log
alert_log: list = []

# Scheduled maintenance log
scheduled_log: list = []

# Stream disconnect tracking
stream_paused: dict = {m: False for m in MACHINE_IDS}


# ─── Sensor generation ───────────────────────────────────────────────────────

def day_load_factor(ts: datetime) -> float:
    h = ts.hour + ts.minute / 60.0
    if 6 <= h < 14:
        base, ripple = 0.92, 0.06 * math.sin(2 * math.pi * (h - 6) / 8)
    elif 14 <= h < 22:
        base, ripple = 0.78, 0.05 * math.sin(2 * math.pi * (h - 14) / 8)
    else:
        base, ripple = 0.38, 0.04 * math.sin(2 * math.pi * ((h - 22) % 24) / 8)
    return max(0.1, base + ripple + random.gauss(0, 0.01))


def get_failure_offsets(machine_id: str) -> tuple[dict, str]:
    """
    Returns (sensor_offsets, status_string) based on current failure progress.
    Progress 0→0.3: status=running (pre-warning), 0.3→0.7: warning, 0.7→1.0: fault
    """
    fs = failure_state[machine_id]
    if fs is None:
        return {}, "running"

    mode = FAILURE_MODES[fs["mode"]]
    progress = fs["progress"]

    # Ramp: offset = full_offset * progress * fault_multiplier_if_fault
    if progress < 0.3:
        scale = progress / 0.3
        status = "running"
    elif progress < 0.7:
        scale = 0.3 + (progress - 0.3) / 0.4 * 0.7
        status = mode["threshold_status"]  # "warning"
    else:
        scale = 1.0 + (progress - 0.7) / 0.3 * (mode["fault_multiplier"] - 1.0)
        status = mode["fault_status"]  # "fault"

    offsets = {k: v * scale for k, v in mode["sensor_offsets"].items()}
    return offsets, status


def generate_live_reading(machine_id: str) -> dict:
    profile = MACHINE_PROFILES[machine_id]
    ts = datetime.now(timezone.utc)
    load = day_load_factor(ts)

    offsets, status = get_failure_offsets(machine_id)

    rpm     = profile["rpm_base"]     * load + random.gauss(0, profile["rpm_std"])
    current = profile["current_base"] * load + random.gauss(0, profile["current_std"])
    vib     = abs(profile["vib_base"] + random.gauss(0, profile["vib_std"]))
    temp    = profile["temp_base"]    + random.gauss(0, profile["temp_std"])

    raw = {
        "machine_id":     machine_id,
        "timestamp":      ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temperature_C":  temp    + offsets.get("temperature_C",  0),
        "vibration_mm_s": vib     + offsets.get("vibration_mm_s", 0),
        "rpm":            abs(rpm + offsets.get("rpm",            0)),
        "current_A":      abs(current + offsets.get("current_A", 0)),
        "status":         status,
    }

    corrected = apply_physics(raw, profile)
    corrected["status"] = status  # physics doesn't override status
    return corrected


# ─── Failure progress ticker ─────────────────────────────────────────────────

async def failure_tick_loop():
    """Background task: advance failure progress every second."""
    while True:
        for machine_id in MACHINE_IDS:
            fs = failure_state[machine_id]
            if fs is not None:
                mode = FAILURE_MODES[fs["mode"]]
                # progress from 0→1 over ramp_seconds
                increment = 1.0 / mode["ramp_seconds"]
                fs["progress"] = min(1.0, fs["progress"] + increment)
        await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(failure_tick_loop())


# ─── SSE Stream ──────────────────────────────────────────────────────────────

async def sensor_event_generator(machine_id: str):
    if machine_id not in MACHINE_IDS:
        yield f"data: {json.dumps({'error': 'unknown machine'})}\n\n"
        return

    while True:
        try:
            reading = generate_live_reading(machine_id)
            yield f"data: {json.dumps(reading)}\n\n"
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            break
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(2)


@app.get("/stream/{machine_id}")
async def stream_machine(machine_id: str):
    return StreamingResponse(
        sensor_event_generator(machine_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ─── History endpoint ─────────────────────────────────────────────────────────

@app.get("/history/{machine_id}")
async def get_history(machine_id: str):
    if machine_id not in MACHINE_IDS:
        raise HTTPException(status_code=404, detail="Unknown machine")

    history_path = os.path.join(
        os.path.dirname(__file__), "history", f"{machine_id}.csv"
    )

    if not os.path.exists(history_path):
        raise HTTPException(
            status_code=404,
            detail=f"History not generated yet. Run: python generate_history.py"
        )

    import csv
    readings = []
    with open(history_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            readings.append({
                "machine_id":     row["machine_id"],
                "timestamp":      row["timestamp"],
                "temperature_C":  float(row["temperature_C"]),
                "vibration_mm_s": float(row["vibration_mm_s"]),
                "rpm":            float(row["rpm"]),
                "current_A":      float(row["current_A"]),
                "status":         row["status"],
            })

    return {
        "machine_id": machine_id,
        "label":      MACHINE_PROFILES[machine_id]["label"],
        "count":      len(readings),
        "readings":   readings,
    }


# ─── Alert endpoint ───────────────────────────────────────────────────────────

class AlertPayload(BaseModel):
    machine_id: str
    reason: str
    risk_score: Optional[float] = None
    sensor_values: Optional[dict] = None


@app.post("/alert")
async def raise_alert(payload: AlertPayload):
    entry = {
        "id":           len(alert_log) + 1,
        "machine_id":   payload.machine_id,
        "reason":       payload.reason,
        "risk_score":   payload.risk_score,
        "sensor_values":payload.sensor_values,
        "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "acknowledged": False,
    }
    alert_log.append(entry)
    print(f"[ALERT #{entry['id']}] {payload.machine_id}: {payload.reason}")
    return {"success": True, "alert_id": entry["id"], "entry": entry}


# ─── Schedule maintenance endpoint ───────────────────────────────────────────

class SchedulePayload(BaseModel):
    machine_id: str
    reason: Optional[str] = None
    priority: Optional[str] = "normal"   # low / normal / high / critical


@app.post("/schedule-maintenance")
async def schedule_maintenance(payload: SchedulePayload):
    # Simulate booking: next available slot (30-min buckets, 08:00–18:00)
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    # Find next 30-min slot at least 2 hours from now
    slot_start = now + timedelta(hours=2)
    mins = slot_start.minute
    slot_start = slot_start.replace(
        minute=30 if mins < 30 else 0, second=0, microsecond=0
    )
    if mins >= 30:
        slot_start += timedelta(hours=1)

    entry = {
        "id":          len(scheduled_log) + 1,
        "machine_id":  payload.machine_id,
        "reason":      payload.reason,
        "priority":    payload.priority,
        "slot_start":  slot_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slot_end":    (slot_start + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "booked_at":   now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status":      "scheduled",
    }
    scheduled_log.append(entry)
    print(f"[SCHEDULE #{entry['id']}] {payload.machine_id} → {entry['slot_start']}")
    return {"success": True, "schedule_id": entry["id"], "entry": entry}


# ─── Control endpoints (for GUI) ──────────────────────────────────────────────

@app.get("/status")
async def get_status():
    result = {}
    for m in MACHINE_IDS:
        fs = failure_state[m]
        reading = generate_live_reading(m)
        result[m] = {
            "label":        MACHINE_PROFILES[m]["label"],
            "live":         reading,
            "failure_mode": fs["mode"]     if fs else None,
            "failure_label":FAILURE_MODES[fs["mode"]]["label"] if fs else None,
            "progress":     round(fs["progress"], 3) if fs else 0.0,
        }
    return result


class InjectPayload(BaseModel):
    machine_id: str
    failure_mode: str


@app.post("/inject-failure")
async def inject_failure(payload: InjectPayload):
    if payload.machine_id not in MACHINE_IDS:
        raise HTTPException(status_code=404, detail="Unknown machine")
    if payload.failure_mode not in FAILURE_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown failure mode. Valid: {list(FAILURE_MODES.keys())}")

    failure_state[payload.machine_id] = {
        "mode":     payload.failure_mode,
        "progress": 0.0,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    mode_info = FAILURE_MODES[payload.failure_mode]
    print(f"[INJECT] {payload.machine_id} → {mode_info['label']}")
    return {
        "success":     True,
        "machine_id":  payload.machine_id,
        "mode":        payload.failure_mode,
        "label":       mode_info["label"],
        "description": mode_info["description"],
        "ramp_seconds":mode_info["ramp_seconds"],
    }


class ClearPayload(BaseModel):
    machine_id: str


@app.post("/clear-failure")
async def clear_failure(payload: ClearPayload):
    if payload.machine_id not in MACHINE_IDS:
        raise HTTPException(status_code=404, detail="Unknown machine")

    prev = failure_state[payload.machine_id]
    failure_state[payload.machine_id] = None
    return {
        "success":    True,
        "machine_id": payload.machine_id,
        "cleared":    prev["mode"] if prev else None,
    }


@app.get("/alerts")
async def get_alerts():
    return {"count": len(alert_log), "alerts": alert_log}


@app.get("/scheduled")
async def get_scheduled():
    return {"count": len(scheduled_log), "scheduled": scheduled_log}


@app.get("/failure-modes")
async def get_failure_modes():
    return {
        k: {
            "label":       v["label"],
            "description": v["description"],
            "ramp_seconds":v["ramp_seconds"],
        }
        for k, v in FAILURE_MODES.items()
    }


# ─── Control Panel HTML ───────────────────────────────────────────────────────

@app.get("/control-panel", response_class=HTMLResponse)
async def control_panel():
    html_path = os.path.join(os.path.dirname(__file__), "control_panel.html")
    with open(html_path) as f:
        return HTMLResponse(content=f.read())


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Predictive Maintenance Simulator",
        "machines": MACHINE_IDS,
        "endpoints": {
            "stream":       "GET  /stream/{machine_id}",
            "history":      "GET  /history/{machine_id}",
            "alert":        "POST /alert",
            "schedule":     "POST /schedule-maintenance",
            "status":       "GET  /status",
            "inject":       "POST /inject-failure",
            "clear":        "POST /clear-failure",
            "alerts_log":   "GET  /alerts",
            "scheduled_log":"GET  /scheduled",
            "failure_modes":"GET  /failure-modes",
            "control_panel":"GET  /control-panel",
        }
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
