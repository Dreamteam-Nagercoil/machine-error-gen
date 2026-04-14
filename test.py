import json

import joblib
import pandas as pd
import requests

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = "http://localhost:8000"
machine_id = "CNC_01"

# Load trained model
try:
    model = joblib.load(f"model_{machine_id}.pkl")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

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
        "sensor_values": data,
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

    try:
        # We use stream=True to keep the connection open indefinitely
        # headers={'Accept': 'text/event-stream'} tells the server we want SSE
        with requests.get(
            url, stream=True, headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                print(f"Failed to connect: {response.status_code}")
                return

            # iter_lines handles the byte-to-string decoding automatically
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                # SSE lines start with "data: "
                if line.startswith("data:"):
                    try:
                        # Extract the JSON part after "data: "
                        payload_str = line[5:].strip()
                        if not payload_str:
                            continue

                        data = json.loads(payload_str)

                        rpm = data["rpm"]
                        temp = data["temperature_C"]
                        vib = data["vibration_mm_s"]
                        actual_current = data["current_A"]

                        # Prepare data for prediction
                        X = pd.DataFrame(
                            [{"rpm": rpm, "temperature_C": temp, "vibration_mm_s": vib}]
                        )

                        predicted_current = model.predict(X)[0]
                        error = abs(actual_current - predicted_current)

                        print(
                            f"Actual: {actual_current:.2f}, Predicted: {predicted_current:.2f}, Error: {error:.2f}"
                        )

                        if error > ERROR_THRESHOLD:
                            print("⚠️ Anomaly detected!")
                            send_alert("High Current Deviation", error, data)

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Processing error: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_agent()
