"""
Physics model: realistic sensor correlations for industrial machines.

Correlation rules (all bidirectional cause->effect chains):
  RPM up    -> temperature up (friction), current up (load), vibration slightly up
  current up -> temperature up (I²R heating)
  vibration up -> temperature up (bearing friction), rpm slightly unstable
  temperature up -> current up slightly (motor resistivity increases)
  status fault -> all sensors degrade
"""

import random
import math

# Per-machine baseline profiles (each machine has different "personality")
MACHINE_PROFILES = {
    "CNC_01": {
        "temp_base": 72.0,   "temp_std": 3.0,
        "vib_base": 1.40,    "vib_std": 0.15,
        "rpm_base": 1480.0,  "rpm_std": 20.0,
        "current_base": 12.5,"current_std": 0.8,
        "label": "CNC Lathe",
    },
    "HYD_02": {
        "temp_base": 58.0,   "temp_std": 4.0,
        "vib_base": 0.85,    "vib_std": 0.10,
        "rpm_base": 960.0,   "rpm_std": 15.0,
        "current_base": 9.8, "current_std": 0.6,
        "label": "Hydraulic Press",
    },
    "COMP_03": {
        "temp_base": 88.0,   "temp_std": 5.0,
        "vib_base": 2.10,    "vib_std": 0.20,
        "rpm_base": 3000.0,  "rpm_std": 30.0,
        "current_base": 18.2,"current_std": 1.2,
        "label": "Air Compressor",
    },
    "CONV_04": {
        "temp_base": 45.0,   "temp_std": 2.5,
        "vib_base": 0.60,    "vib_std": 0.08,
        "rpm_base": 480.0,   "rpm_std": 10.0,
        "current_base": 6.4, "current_std": 0.4,
        "label": "Conveyor Belt",
    },
}

MACHINE_IDS = list(MACHINE_PROFILES.keys())

def apply_physics(reading: dict, profile: dict) -> dict:
    """
    Given raw sensor values (possibly with a failure offset),
    apply cross-sensor physics so correlations stay realistic.
    Returns adjusted reading.
    """
    r = dict(reading)

    # RPM deviation from baseline drives temperature and current
    rpm_delta = (r["rpm"] - profile["rpm_base"]) / profile["rpm_base"]  # fractional
    # +10% RPM -> +2°C temp, +0.5A current
    r["temperature_C"] += rpm_delta * 20.0
    r["current_A"]     += rpm_delta * 5.0

    # Current deviation drives temperature (I²R)
    cur_delta = (r["current_A"] - profile["current_base"]) / profile["current_base"]
    # +10% current -> +1.5°C
    r["temperature_C"] += cur_delta * 15.0

    # High vibration -> bearing friction -> temperature rise + rpm jitter
    vib_delta = (r["vibration_mm_s"] - profile["vib_base"]) / max(profile["vib_base"], 0.1)
    r["temperature_C"]   += vib_delta * 4.0
    r["rpm"]             += vib_delta * (-5.0) * random.uniform(0.5, 1.5)  # unstable

    # Temperature rise -> motor resistance -> tiny current increase
    temp_delta = (r["temperature_C"] - profile["temp_base"]) / profile["temp_base"]
    r["current_A"] += temp_delta * 2.0

    # Add noise on top
    r["temperature_C"]  += random.gauss(0, profile["temp_std"] * 0.2)
    r["vibration_mm_s"] += random.gauss(0, profile["vib_std"] * 0.2)
    r["rpm"]            += random.gauss(0, profile["rpm_std"] * 0.15)
    r["current_A"]      += random.gauss(0, profile["current_std"] * 0.2)

    # Clamp to physical limits
    r["temperature_C"]  = max(20.0,  round(r["temperature_C"],  2))
    r["vibration_mm_s"] = max(0.01,  round(r["vibration_mm_s"], 3))
    r["rpm"]            = max(0.0,   round(r["rpm"],            1))
    r["current_A"]      = max(0.0,   round(r["current_A"],      2))

    return r


# ─── Failure injection models ───────────────────────────────────────────────

FAILURE_MODES = {
    "bearing_wear": {
        "label": "Bearing Wear",
        "description": "Gradual vibration increase → temperature rise → rpm instability",
        "sensor_offsets": {
            "vibration_mm_s": 3.5,   # primary
            "temperature_C": 8.0,    # secondary via physics
            "rpm": -30.0,            # slight slowdown
            "current_A": 1.5,        # slight overload
        },
        "ramp_seconds": 60,          # seconds to reach full offset
        "threshold_status": "warning",
        "fault_status": "fault",
        "fault_multiplier": 2.5,
    },
    "overheating": {
        "label": "Overheating",
        "description": "Thermal runaway: cooling failure → temp spikes → current spike",
        "sensor_offsets": {
            "temperature_C": 22.0,   # primary
            "current_A": 3.5,        # secondary via physics
            "vibration_mm_s": 0.4,
            "rpm": -20.0,
        },
        "ramp_seconds": 45,
        "threshold_status": "warning",
        "fault_status": "fault",
        "fault_multiplier": 2.0,
    },
    "electrical_fault": {
        "label": "Electrical Fault",
        "description": "Current surge → insulation breakdown → temp + vibration",
        "sensor_offsets": {
            "current_A": 6.0,        # primary
            "temperature_C": 12.0,   # secondary
            "vibration_mm_s": 1.0,
            "rpm": -50.0,            # motor struggles
        },
        "ramp_seconds": 20,          # faster onset
        "threshold_status": "warning",
        "fault_status": "fault",
        "fault_multiplier": 2.2,
    },
    "mechanical_imbalance": {
        "label": "Mechanical Imbalance",
        "description": "Mass imbalance → vibration spike → bearing stress → heat",
        "sensor_offsets": {
            "vibration_mm_s": 5.0,   # primary
            "temperature_C": 10.0,
            "current_A": 2.0,
            "rpm": 40.0,             # runaway tendency
        },
        "ramp_seconds": 30,
        "threshold_status": "warning",
        "fault_status": "fault",
        "fault_multiplier": 3.0,
    },
    "rpm_runaway": {
        "label": "RPM Runaway",
        "description": "Speed controller failure → RPM spikes → thermal + current overload",
        "sensor_offsets": {
            "rpm": 350.0,            # primary
            "temperature_C": 15.0,
            "current_A": 4.0,
            "vibration_mm_s": 1.8,
        },
        "ramp_seconds": 25,
        "threshold_status": "warning",
        "fault_status": "fault",
        "fault_multiplier": 2.8,
    },
}
