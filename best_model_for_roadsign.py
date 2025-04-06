import multiprocessing
import numpy as np
import cv2, cvzone
import requests
import time
from flask import Flask, jsonify
from ctypes import c_int
import re  # For optimized speed limit extraction
import socket
import threading

HOST = "192.168.254.99"  # Replace with ESP32's IP
PORT = 8080

shared_value = multiprocessing.Value(c_int, 80)  # Speed limit
shared_sensor_data = multiprocessing.Value('d', 0)  # Sensor data
lock = multiprocessing.Lock()

app = Flask(__name__)

def send_data_esp32(sensor_data, speed_limit):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect((HOST, PORT))
            data = f"SENSOR:{sensor_data}\nLIMIT:{speed_limit}\n"
            s.sendall(data.encode())
    except Exception as e:
        print("ESP32 Socket Error:", e)

def get_token():
    try:
        auth_url = "http://wazigate.local/auth/token"
        credentials = {"username": "admin", "password": "loragateway"}
        response = requests.post(auth_url, json=credentials, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        print("Token Error:", e)
    return None

def get_sensor_data(token):
    try:
        sensor_url = "http://wazigate.local/devices/b827ebae5f25058c/sensors"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        response = requests.get(sensor_url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data[0].get("value")
    except requests.RequestException as e:
        print("Sensor Data Error:", e)
    return None

def fetch_sensor_data_periodically():
    while True:
        token = get_token()
        if token:
            sensor_data = get_sensor_data(token)
            if isinstance(sensor_data, (int, float)):
                sensor_data *= 7.5
                with lock:
                    shared_sensor_data.value = sensor_data
        time.sleep(5)

@app.route('/api/vehicle-data')
def send_data():
    with lock:
        current_speed_limit = shared_value.value
        current_sensor_data = shared_sensor_data.value

    is_overspeed = current_sensor_data > current_speed_limit

    data = {
        "vehicle_id": "TZ 322 ABC",
        "location": "Kijitonyama",
        "vts_status": "Active",
        "driving_performance": "Good",
        "driver_name": "Simon sosola Sylas",
        "driver_license": "D123456789",
        "speed_limit": current_speed_limit,
        "current_speed": current_sensor_data,
        "is_overspeed": is_overspeed,
        "registration_number": "Asds-sd23-ds",
        "vehicle_type": "Truck",
        "contact_number": "+255629110284",
        "road_name": "Bagamoyo Road",
        "coordinates": "-6.7714281, 39.2399597"
    }
    return jsonify(data)

def run_server():
    app.run(debug=False, use_reloader=False, threaded=True, host="0.0.0.0", port=5000)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    import inference
    model = inference.get_model("road_sign-detector/3")

    frame_counter = 0
    last_sent_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Frame capture failed.")
            continue

        frame_counter += 1

        # Send to ESP32 every second
        if time.time() - last_sent_time > 1:
            with lock:
                send_data_esp32(shared_sensor_data.value, shared_value.value)
            last_sent_time = time.time()

        if frame_counter % 5 != 0:
            continue

        try:
            img_resized = cv2.resize(frame, (640, 640))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_input = img_gray.reshape(640, 640, 1)

            result = list(model.infer(image=img_input))[0].predictions

            if result:
                pred = result[0]
                x, y, w, h = map(int, (pred.x, pred.y, pred.width, pred.height))
                label = f"{pred.confidence:.2f} : {pred.class_name}"

                if pred.confidence >= 0.70:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cvzone.putTextRect(frame, label, (x - 25, y), 1, 2)

                    match = re.search(r"t-(\d+)k", pred.class_name)
                    if match:
                        with lock:
                            shared_value.value = int(match.group(1))

        except Exception as e:
            print("Inference error:", e)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    flask_process = multiprocessing.Process(target=run_server)
    flask_process.start()

    sensor_thread = threading.Thread(target=fetch_sensor_data_periodically, daemon=True)
    sensor_thread.start()

    main()

    flask_process.terminate()
    flask_process.join()