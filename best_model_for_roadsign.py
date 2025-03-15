import multiprocessing
import numpy as np
import cv2, cvzone
import requests
import time
from flask import Flask, jsonify
from ctypes import c_int
import re  # For optimized speed limit extraction

# Shared value for speed limit with a lock
shared_value = multiprocessing.Value(c_int, 0)
lock = multiprocessing.Lock()

app = Flask(__name__)

# Function to fetch token
def get_token():
    try:
        auth_url = "http://wazigate.local/auth/token"
        credentials = {"username": "admin", "password": "loragateway"}
        response = requests.post(auth_url, json=credentials, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException as e:
        print("Token Error:", e)
        return None

# Function to fetch sensor data
def get_sensor_data(token):
    try:
        sensor_url = "http://wazigate.local/devices/b827ebae5f25058c/sensors"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        response = requests.get(sensor_url, headers=headers, timeout=5)
        if response.status_code == 200:
            Data = response.json()
            Data = Data[0].get("value")
            return Data
        return {"error": "Failed to fetch sensor data"}
    except requests.RequestException as e:
        print("Sensor Data Error:", e)
        return {"error": str(e)}

# API to send vehicle data
@app.route('/api/vehicle-data')
def send_data():
    token = get_token()
    if not token:
        return jsonify({"error": "Authentication failed"}), 404

    sensor_data = get_sensor_data(token)

    with lock:
        current_speed_limit = shared_value.value  # Safely read shared value

    data = {
        "vehicle_id": "TZ 322 ABC",
        "location": "Kijitonyama",
        "vts_status": "Active",
        "driving_performance": "Good",
        "driver_name": "Simon sosola Sylas",
        "driver_license": "D123456789",
        "speed_limit": current_speed_limit,
        "current_speed": sensor_data,
        "is_overspeed": False,
        "registration_number": "Asds-sd23-ds",
        "vehicle_type": "Truck",
        "contact_number": "+255629110284",
        "road_name": "Bagamoyo Road",
        "coordinates": "-6.7714281, 39.2399597"
    }
    return jsonify(data)

# Function to run Flask server in a separate process
def run_server():
    app.run(debug=False, use_reloader=False, threaded=True, host="0.0.0.0", port=5000)

# Main vision system function
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    import inference
    model = inference.get_model("road_sign-detector/3")

    frame_counter = 0  # To process every 3rd frame only

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Frame capture failed.")
            continue

        frame_counter += 1
        if frame_counter % 3 != 0:  # Skip frames for efficiency
            time.sleep(0.01)  # Reduce CPU usage
            continue

        try:
            img1 = cv2.resize(img, (640, 640))
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img1 = cv2.equalizeHist(img1).reshape(640, 640, 1)

            result = model.infer(image=img1)
            result = list(result)[0].predictions

            if result:
                class_name = result[0].class_name
                conf = result[0].confidence
                x1, y1, x2, y2 = map(int, (result[0].x, result[0].y, result[0].width, result[0].height))

                cv2.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), (255, 0, 0), 3)
                label_str = f"{conf:.2f} : {class_name}"

                if conf >= 0.70:
                    cvzone.putTextRect(img, label_str, (x1 - 25, y1), 1, 2)

                    # Optimized way to extract speed limit from class_name
                    match = re.search(r"t-(\d+)k", class_name)
                    if match:
                        speed_limit = int(match.group(1))  # Extract number
                        with lock:
                            shared_value.value = speed_limit

        except (IndexError, ValueError) as e:
            print("Processing Error:", e)

        # Display the image output
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    # Run Flask in a separate process
    flask_process = multiprocessing.Process(target=run_server)
    flask_process.start()

    # Run main vision processing
    main()

    # Ensure Flask process terminates cleanly
    flask_process.terminate()
    flask_process.join()
