"""
Generate 7-day error-free historical data for all 4 machines.
Output: history/{machine_id}.csv
"""

import os
import csv
import math
import random
from datetime import datetime, timedelta, timezone
from physics import MACHINE_PROFILES, MACHINE_IDS, apply_physics

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "history")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_INTERVAL_SECONDS = 60   # one reading every 10 seconds → 60480 rows per machine/7days


def day_load_factor(ts: datetime) -> float:
    """
    Simulates a realistic shift pattern:
      - 06:00–14:00  day shift:   load 0.85–1.0
      - 14:00–22:00  evening shift: load 0.7–0.9
      - 22:00–06:00  night (low):  load 0.3–0.5
    Returns a multiplier applied to RPM/current (temp follows via physics).
    """
    h = ts.hour + ts.minute / 60.0
    if 6 <= h < 14:
        base = 0.92
        ripple = 0.06 * math.sin(2 * math.pi * (h - 6) / 8)
    elif 14 <= h < 22:
        base = 0.78
        ripple = 0.05 * math.sin(2 * math.pi * (h - 14) / 8)
    else:
        base = 0.38
        ripple = 0.04 * math.sin(2 * math.pi * ((h - 22) % 24) / 8)
    return max(0.1, base + ripple + random.gauss(0, 0.01))


def generate_clean_reading(ts: datetime, profile: dict, machine_id: str) -> dict:
    load = day_load_factor(ts)

    # Base values scaled by load
    rpm     = profile["rpm_base"]     * load + random.gauss(0, profile["rpm_std"])
    current = profile["current_base"] * load + random.gauss(0, profile["current_std"])
    vib     = profile["vib_base"]     + random.gauss(0, profile["vib_std"])
    temp    = profile["temp_base"]    + random.gauss(0, profile["temp_std"])

    raw = {
        "machine_id":       machine_id,
        "timestamp":        ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temperature_C":    round(temp,    2),
        "vibration_mm_s":   round(abs(vib), 3),
        "rpm":              round(abs(rpm), 1),
        "current_A":        round(abs(current), 2),
        "status":           "running",
    }

    # Apply cross-sensor physics (small deviations only since no fault offset)
    raw["temperature_C"]  = temp    # let physics recalculate from rpm/current
    raw["vibration_mm_s"] = abs(vib)

    corrected = apply_physics(raw, profile)
    corrected["status"] = "running"
    return corrected


def generate_history():
    end_time   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_time = end_time - timedelta(days=7)

    for machine_id in MACHINE_IDS:
        profile = MACHINE_PROFILES[machine_id]
        filepath = os.path.join(OUTPUT_DIR, f"{machine_id}.csv")

        fieldnames = [
            "machine_id", "timestamp", "temperature_C",
            "vibration_mm_s", "rpm", "current_A", "status"
        ]

        ts = start_time
        rows_written = 0

        print(f"Generating history for {machine_id} ({profile['label']})...")

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while ts <= end_time:
                row = generate_clean_reading(ts, profile, machine_id)
                writer.writerow(row)
                ts += timedelta(seconds=SAMPLE_INTERVAL_SECONDS)
                rows_written += 1

        print(f"  → {rows_written} rows written to {filepath}")

    print("\nHistory generation complete.")


if __name__ == "__main__":
    generate_history()
