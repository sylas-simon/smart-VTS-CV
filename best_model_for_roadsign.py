import multiprocessing.process
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import cv2,cvzone
from ctypes import c_char_p
from multiprocessing import Value
import time
import requests

#export ROBOFLOW_API_KEY="3kK4ZOLcFMNNpFdAadhw"
from flask import Flask, jsonify
class_names = None
intSpeedLimit = int()
shared_value = multiprocessing.Value('i',intSpeedLimit)

app = Flask(__name__)
# Shared variable for speed limit (from your vision system)
shared_value = multiprocessing.Value('i', 0)

# Function to fetch token
def get_token():
    try:
        auth_url = "http://wazigate.local/auth/token"
        credentials = {"username": "admin", "password": "loragateway"}
        response = requests.post(auth_url, json=credentials)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print("Token Error:", e)
        return None

# Function to fetch sensor data
def get_sensor_data(token):
    try:
        sensor_url = "http://wazigate.local/devices/b827ebae5f25058c/sensors"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        response = requests.get(sensor_url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Failed to fetch sensor data"}
    except Exception as e:
        print("Sensor Data Error:", e)
        return {"error": str(e)}

# API to send both sensor data and speed limit to frontend
@app.route('/data')
def send_data():
    token = get_token()
    if not token:
        return jsonify({"error": "Authentication failed"}), 404

    sensor_data = get_sensor_data(token)
    
    # Combine sensor data + speed limit
    data = {
        "sensor_data": sensor_data,
        "speed_limit": shared_value.value
    }
    return jsonify(data)

def runningServer():
   app.run(debug=False, use_reloader=False, threaded=True)
   data =input.value


  
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    import inference
    model = inference.get_model("road_sign-detector/3")
    while True:
        try:
            success, img = cap.read()
            img1 = np.asarray(img)
            img1 = cv2.resize(img1, (640, 640))
            img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
            img1 = cv2.equalizeHist(img1)
            img1 = img1.reshape(640,640,1)
            result = model.infer(image= img1)

            #img = cv2.imread("/home/sylas/COMPUTER_VISION/archive (1)/traffic_Data (copy)/DATA/1/001_0010.png")
           ## print(result)
            result = list(result)
            result = result[0].predictions
            class_name = result[0].class_name
            conf = result[0].confidence
            x1,y1,x2,y2 = int(result[0].x),int(result[0].y), int(result[0].width),int(result[0].height)

            cv2.rectangle(img,(int(x1+x2/2),int(y1+y2/2)),(int(x1-x2/2),int(y1-y2/2)),(255,0,0),3,)

            label_str = f"{conf:.2f} : {class_name}"

            #print(label_str)
            if conf >= 0.70:
                cvzone.putTextRect(img,label_str,(x1-25,y1),1,2)
                global class_names,shared_value,substring,intSpeedLimit
                class_names = str(class_name)
                startIndex = class_names.find("t-")
                endIndex = class_names.find("k")
                e =slice(startIndex+2,endIndex)
                substring = class_names[e]
                intSpeedLimit = int(substring)
                # Synchronize access
                shared_value.value = intSpeedLimit
                ##print(intSpeedLimit)
                

                
            
        except IndexError:
            #print("attempted to access")
            pass

        cv2.imshow("image",img)
        
        cv2.waitKey(1)


    #print(result[0])
    """

    model1 = YOLO('yolov8n.pt')
    #result = model("/home/sylas/COMPUTER_VISION/images.jpeg",show = True,)
    #dic_class = model1.model.names
    #for i,j in dic_class.items():
    #   print(j)
    #cv2.waitKey(0) 
    #print(len(dic_class))

    result = model1.train(data='/home/sylas/COMPUTER_VISION/road_sign detector.v3i.yolov8/data.yaml',
                        epochs = 1,
                        batch = 10,
                        imgsz= 640,
                        name = "best")

    """

if __name__ == "__main__":  

  flaskmultiprocessing = multiprocessing.Process(target=runningServer )
  flaskmultiprocessing.start()

  main()