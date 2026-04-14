import requests
import joblib
import json
from sseclient import SSEClient
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = "http://localhost:8000"
machine_id = "CNC_01"

# Load trained model
model = joblib.load(f"model_{machine_id}.pkl")

# Threshold (tune this later)
ERROR_THRESHOLD = 1.0

# -----------------------------
# ALERT FUNCTION
# -----------------------------
def send_alert(reason, risk_score, data):
    payload = {
        "machine_id": machine_id,
        "reason": reason,
        "risk_score": risk_score,
        "sensor_values": data
    }

    try:
        res = requests.post(f"{BASE_URL}/alert", json=payload)
        print("🚨 ALERT SENT:", res.json())
    except Exception as e:
        print("Alert failed:", e)

# -----------------------------
# STREAM PROCESSING
# -----------------------------
def run_agent():
    print(f"Connecting to stream for {machine_id}...")

    url = f"{BASE_URL}/stream/{machine_id}"
    client = SSEClient(url)

    for event in client:
        try:
            if not event.data:
                continue

            data = json.loads(event.data)

            rpm = data["rpm"]
            temp = data["temperature_C"]
            vib = data["vibration_mm_s"]
            actual_current = data["current_A"]

            # ✅ FIX HERE
            X = pd.DataFrame([{
                "rpm": rpm,
                "temperature_C": temp,
                "vibration_mm_s": vib
            }])

            predicted_current = model.predict(X)[0]
            error = abs(actual_current - predicted_current)

            print(f"Actual: {actual_current:.2f}, Predicted: {predicted_current:.2f}, Error: {error:.2f}")

            if error > ERROR_THRESHOLD:
                print("⚠️ Anomaly detected!")

        except Exception as e:
            print("Error:", e)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_agent()