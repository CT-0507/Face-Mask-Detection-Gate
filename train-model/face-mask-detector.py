import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
import imutils
import time
import cv2
import pathlib 


isMasked = False
end = 0
colorRed = (0,0,255)
colorGreen = (0,255,0)


def detect_and_predict_mask_lite(frame, faceNet, interpreper):
    # Lấy chiều cao và rộng của ảnh
    (h, w) = frame.shape[:2]
    # Đưa ảnh vào dnn để phân tích
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()


    faces = [] # danh sách face
    locs = [] # danh sách location face
    preds = [] # danh sách predict

    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2] # Lấy các confidence

        if confidence > 0.5: # Nếu confidence > 50%
            # Cắt khuôn mặt trong ảnh
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]

            # handle loi !_src.empty() 
            try:
                # Chuyển đổi định dạng cho ảnh 
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
			# add the face and bounding boxes to their respective
			# lists
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
            #print(result[0])
    # trả về location của các face và predict của chúng
    return (locs, preds)


# Tạo đường dẫn đến ResNet model và tensorflow lite model
protocolPath = 'deploy.protocol.txt'
weightPath = 'res10_300x300_ssd_iter_140000.caffemodel'
tflite_model_file = 'trained_models/model.tflite'

# loading dnn model có sẵn
print('Loading dnn')
faceNet = cv2.dnn.readNet(protocolPath, weightPath)

# Loading model tensorflow lite
print('Loading model lite.')
interpreter = tf.lite.Interpreter(model_path = 'model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]



# initialize the video stream and allow the camera sensor to warm up
print("starting video stream...")
vs = VideoStream(src=0, framerate = 30).start()
time.sleep(2.0)

while True:
    # lấy frame
    frame = vs.read()
    
    # kiểm tra frame
    if frame is None:
        continue
    
    frame = cv2.flip(frame, 1)
    # resize lại frame thành 4000
    frame = imutils.resize(frame, width=800)

    
    # Đừa vào detect và predict
    (locs, preds) = detect_and_predict_mask_lite(frame, faceNet, interpreter)

    for (box, pred) in zip(locs, preds):
		# Lấy tọa độ để vẽ khung
        (startX, startY, endX, endY) = box

        # lấy kết quả từ perdicts
        (mask, withoutMask) = pred
		
        #determine the label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = colorGreen if label == "Mask" else colorRed
		
        # Gắn nhãn kết quả 
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
        # Hiển thị text nhãn kết quả
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        # vẽ box
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # show ảnh
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()