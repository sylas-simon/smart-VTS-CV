import multiprocessing
import threading
import socket
import requests
import time
import cv2
import cvzone
import numpy as np
import re
from flask import Flask, jsonify
from ctypes import c_int

# ========== Helper Functions ==========

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    return ip_address

import socket

def send_data_to_esp32(sensor_data, speed_limit):
    try:
        # Create a UDP socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # Define server address and port
            server_address = (ESP32_IP, ESP32_PORT)

            # Prepare the payload
            payload = f"SENSOR:{sensor_data} LIMIT:{speed_limit}\n"

            # Send the data over UDP (no connection setup needed)
            sock.sendto(payload.encode(), server_address)

            print(f"Data sent to ESP32 (UDP): {payload.strip()}")

    except Exception as e:
        print(f"[ESP32 UDP Send Error] {e}")


def get_token():
    try:
        response = requests.post(
            "http://wazigate.local/auth/token",
            json={"username": "admin", "password": "loragateway"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("access_token")
    except requests.RequestException as e:
        print(f"[Token Error] {e}")
    return None

def get_sensor_data(token):
    try:
        response = requests.get(
            "http://wazigate.local/devices/b827ebae5f25058c/sensors",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                return data[0].get("value")
    except requests.RequestException as e:
        print(f"[Sensor Data Error] {e}")
    return None

# ========== Globals ==========

ip_address = get_ip_address()
ip_parts = ip_address.split('.')
ip_parts[-1] = '168'  # Adjust last octet for ESP32
ESP32_IP = '.'.join(ip_parts)
ESP32_PORT = 8080

shared_speed_limit = multiprocessing.Value(c_int, 80)   # Default speed limit
shared_sensor_data = multiprocessing.Value('d', 0.0)    # Default sensor value
lock = multiprocessing.Lock()

app = Flask(__name__)

# ========== Flask API Server ==========

@app.route('/api/vehicle-data')
def api_vehicle_data():
    with lock:
        speed_limit = shared_speed_limit.value
        sensor_data = shared_sensor_data.value

    is_overspeed = sensor_data > speed_limit

    data = {
        "vehicle_id": "TZ 322 ABC",
        "location": "Kijitonyama",
        "vts_status": "Active",
        "driving_performance": "Good",
        "driver_name": "Simon sosola Sylas",
        "driver_license": "D123456789",
        "speed_limit": speed_limit,
        "current_speed": sensor_data,
        "is_overspeed": is_overspeed,
        "registration_number": "Asds-sd23-ds",
        "vehicle_type": "Truck",
        "contact_number": "+255629110284",
        "road_name": "Bagamoyo Road",
        "coordinates": "-6.7714281, 39.2399597"
    }
    return jsonify(data)

def run_flask_server():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)

# ========== Threads ==========

def sensor_fetch_thread():
    while True:
        token = get_token()
        if token:
            sensor_value = get_sensor_data(token)
            if isinstance(sensor_value, (int, float)):
                with lock:
                    shared_sensor_data.value = sensor_value * 7.5  # Scaling factor
                    print(f"[Sensor] Updated sensor data: {shared_sensor_data.value}")
        else:
            print("[Sensor] Token not retrieved.")
        time.sleep(5)

def esp32_sender_thread():
    while True:
        with lock:
            speed_limit = shared_speed_limit.value
            sensor_data = 130
        send_data_to_esp32(sensor_data, speed_limit)
        time.sleep(2)

# ========== Main Camera Processing ==========

def main_camera_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[Camera Error] Could not open camera.")
        return

    try:
        import inference
        model = inference.get_model("road_sign-detector/3")
    except Exception as e:
        print(f"[Model Load Error] {e}")
        return

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("[Camera Frame Error]")
            continue

        frame_count += 1

        if frame_count % 5 == 0:
            try:
                resized_frame = cv2.resize(frame, (640, 640))
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                input_frame = gray_frame.reshape(640, 640, 1)

                prediction = list(model.infer(image=input_frame))[0].predictions

                if prediction:
                    pred = prediction[0]
                    x, y, w, h = map(int, (pred.x, pred.y, pred.width, pred.height))
                    label = f"{pred.confidence:.2f} : {pred.class_name}"

                    if pred.confidence >= 0.70:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                        cvzone.putTextRect(frame, label, (x-25, y), 1, 2)

                        match = re.search(r"t-(\d+)k", pred.class_name)
                        if match:
                            new_speed_limit = int(match.group(1))
                            with lock:
                                shared_speed_limit.value = new_speed_limit
                                print(f"[Camera] Detected new speed limit: {new_speed_limit}")

            except Exception as e:
                print(f"[Inference Error] {e}")

        # cv2.imshow("Vehicle Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========== Entrypoint ==========

if __name__ == "__main__":
    print(f"Local IP Address: {ip_address}")

    flask_process = multiprocessing.Process(target=run_flask_server)
    flask_process.start()

    threading.Thread(target=sensor_fetch_thread, daemon=True).start()
    threading.Thread(target=esp32_sender_thread, daemon=True).start()

    main_camera_loop()

    flask_process.terminate()
    flask_process.join()
