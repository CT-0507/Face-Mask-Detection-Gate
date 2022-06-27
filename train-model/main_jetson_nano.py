import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import numpy as np
import imutils
import multiprocessing
import cv2
import json
import serial
import threading
import socket
import time
import requests
from flask import Flask, render_template, Response


enterlogApi = 'http://172.31.250.62:3000/enterlogs/upload'
updateDeviceApi = 'http://172.31.250.62:3000/update/device'
refreshDeviceApi = 'http://172.31.250.62:3000/refresh'
deviceIP = None
deviceID = None
refreshDelay = 180 # 3 phut

def get_current_ip():
    # hostname = socket.gethostname()
    # IPAddr = socket.gethostbyname(hostname)
    # return IPAddr
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
            # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# Khởi động môt thread
def start_thread(func, args):
    p = threading.Thread(target= func, args = args, daemon=True)
    p.start()

def register_device(api):
    r = requests.post(url = api, json = {'name': 'Jetson Nano', 'ip':deviceIP})
    print(f'Register result: {r}')
    return r.text

def refresh_device(url, deviceId, delay):
    last_refresh = time.time()
    while True:
        if time.time() - last_refresh > delay:
            print(f'Refresh device after {delay} seconds')
            r = requests.patch(url= url, json = {'_id': deviceId})
            if r.status_code == 200:
                print('Refresh successful.')
            last_refresh = time.time()


def create_packet(img, info, withMask):
    img_encoded = cv2.imencode('.jpg', img)[1]
    name_img = 'img_' + str(round(time.time()))
    mask = 'true' if withMask else 'false' # JS lỗi parsing True, False
    multipart_form_data = {
        'image': (name_img, img_encoded.tobytes()),
        'name':(None, 'Jetson Nano'),
        'ip':(None, info['ip']), # Viết code tìm ip thiết bị
        'withMask':(None, mask),
        'attachTo': (None, info['id']), # Viết 
    }
    return multipart_form_data

# Gửi request
def send_img_request(url, data):
    print('Send img post request')
    r = requests.post(url = url, files = data)
    print(r)

def mask_or_no_mask(results):
    print(f'Total sample: {len(results)}' )
    positive = results.count('yes') / len(results)
    if positive >= 0.6:
        return True
    else:
        return False

def generate_arduino_input(results):
    if mask_or_no_mask(results):
        return json.dumps({'servo': True, 'leds': [True, False, False], 'horn':True})
    else:
        return json.dumps({'servo': False, 'leds': [False, True, False], 'horn': False})
    



def gstreamer_pipline(
    capture_width = 852,
    capture_height = 480,
    display_width = 852,
    display_height = 480,
    framerate = 30,
    flip_method = 2,):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def generate_frame(frameQueue, arduinoInput, streamFrames, records):
    colorRed = (0,0,255)
    colorGreen = (0,255,0)
    results = []

    def detect_and_predict_mask_lite(frame, faceNet, interpreper):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = [] # danh sách face
        locs = [] # danh sách location face
        preds = [] # danh sách predict

        for i in range(0, detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence > 0.5:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX] # cắt khuôn mặt

                try:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                except:
                    print('Loi cvtColor')
                    break
                faces.append(face)  
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            # predict từng face.
            for face in faces:
                x = np.expand_dims(face, axis = 0)
                interpreper.set_tensor(input_index, x)
                interpreper.invoke()
                result = interpreper.get_tensor(output_index)
                preds.append(result[0])
                # print(result[0]) 
        # trả về location của các face và predict của chúng
        return (locs, preds)

    protocolPath = 'deploy.protocol.txt'
    weightPath = 'res10_300x300_ssd_iter_140000.caffemodel'
    tflite_model_file = 'trained_models/model.tflite'

    print('Loading dnn')
    faceNet = cv2.dnn.readNet(protocolPath, weightPath)
    faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    print('Loading model lite.')
    interpreter = tf.lite.Interpreter(model_path = tflite_model_file)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    print('Start gstreamer')
    cap = cv2.VideoCapture(gstreamer_pipline(), cv2.CAP_GSTREAMER)
    #cap = cv2.VideoCapture(0)
    print('Starting capture...')
    time.sleep(2)
    

    lastScan = time.time()
    delayScan = 5
    noCapture = False
    delayCapture = 5
    lastCapture = 60
   
    while True:
        isTrue, frame = cap.read()
        if not isTrue:
            continue

        frame = cv2.flip(frame, 1)
        
        if not noCapture:  
            (locs, preds) = detect_and_predict_mask_lite(frame, faceNet, interpreter)
            for (box, pred) in zip(locs, preds):
                # Lấy tọa độ để vẽ khung
                (startX, startY, endX, endY) = box
                # lấy kết quả từ perdicts
                (mask, withoutMask) = pred
                #determine the label and color
                
                if mask > withoutMask:
                    label = "Mask"
                    color  = colorGreen
                    results.append('yes')
                else:
                    color = colorRed
                    label = "No Mask"
                    results.append('no')

                # color = colorGreen if label == "Mask" else colorRed
                # Gắn nhãn kết quả 
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                # Hiển thị text nhãn kết quả
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                # vẽ box
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                if mask > withoutMask:
                    records.update({'mask': frame})
                else:
                    records.update({'noMask': frame})

        if time.time() - lastCapture > delayCapture and noCapture:
            noCapture = False
            lastScan = time.time()

        if time.time() - lastScan > delayScan and not noCapture:
            if len(results) > 15:
                arduinoInput.put_nowait(results) # This is for arduino
                print('Scan complete')
                noCapture = True
                lastCapture = time.time()
            results = []
            lastScan = time.time()
        
        if not frameQueue.full():
            frameQueue.put_nowait(frame) # Thêm frame vào queue
        if not streamFrames.full():
            streamFrames.put_nowait(frame)


def video_capture(frames):
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        if not frames.empty():
            frame = frames.get()
            
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            cv2.putText(frame, fps, (7, 70),cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Live', frame)
        
        key = cv2.waitKey(1) & 0xFF
	    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
def send_data(ser, arduinoInput):
    while True:
        if not arduinoInput.empty(): # Nếu arduinoInput khác empty
            input = generate_arduino_input(arduinoInput.get()) # tạo input gửi đến arduino
            print(input)
            ser.write(str.encode(input))

def read_data(ser, arduinoOutput, records, info):
    while True:
        if ser.inWaiting(): # Nếu có dữ liệu 
            data = ser.readline().decode('utf-8') # Biến đổi dữ liệu từ binary -> string
            data = json.loads(data) # Biến đổi từ string -> json
            time.sleep(0.05)
            print(f'Arduino Output Data: {data}')
            arduinoOutput.put(data)  # Đưa json data vào hàng đợi

        if not arduinoOutput.empty(): # Nếu hàng đợi có dữ liệu
            jsonData = arduinoOutput.get() # Lấy dữ liệu 
            if jsonData['isOpen'] == '1': # Nếu cửa mở
                print('Send img mask')
                data = create_packet(records['mask'], info, withMask=True)
                start_thread(send_img_request, (enterlogApi, data))
            else: # Nếu cửa đóng
                print('Send img no mask')  
                data = create_packet(records['noMask'],info ,withMask=False)# gửi hình ảnh người ko đeo khẩu trang
                start_thread(send_img_request, (enterlogApi, data))
           
def get_frame(streamFrames):
    while True:
        if not streamFrames.empty():
            img = streamFrames.get()
            imgencode=cv2.imencode('.jpg',img)[1]
            stringData=imgencode.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
                


if __name__ == '__main__':
    #Frame ảnh cho livestream và display
    frameQueue = multiprocessing.Queue(5)  # cùng share các frame ảnh 
    streamFrames  = multiprocessing.Queue(5)
    arduinoOutput = multiprocessing.Queue() # read this from arduino
    arduinoInput = multiprocessing.Queue() # send this to arduino
    
    manager = multiprocessing.Manager()
    records = manager.dict({'mask':'', 'noMask':''})

    
    deviceIP = get_current_ip()
    print(f'Device IP is {deviceIP}')
    deviceID = register_device(updateDeviceApi)
    print(f'Device ID is {deviceID}')
    start_thread(refresh_device, (refreshDeviceApi, deviceID, refreshDelay))
    info = {'ip': deviceIP, 'id': deviceID}

    print('Kết nối Arduino')
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout = 1)
    time.sleep(1)
    
    print('Kết nối Arduino')
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout = 1)
    time.sleep(1)
    
    
    t1 = threading.Thread(target= send_data, args = [ser, arduinoInput], daemon= True)
    t1.start()
    t2 = threading.Thread(target= read_data, args = [ser, arduinoOutput, records, info], daemon= True)
    t2.start()

    p2 = multiprocessing.Process(target = video_capture, args = (frameQueue, ))
    p2.start()
    p1 = multiprocessing.Process(target = generate_frame, args= (frameQueue, arduinoInput, streamFrames, records))
    p1.start()
    
    app = Flask(__name__)
    @app.route('/vid')
    def vid():
        return Response(get_frame(streamFrames),mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host=deviceIP,port=5000, debug=False, threaded=True)
    
    p2.join()
    p1.terminate()
    

    cv2.waitKey(0)



    

