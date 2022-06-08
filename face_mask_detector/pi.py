# https://pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

import numpy as np
import imutils
import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera


print('Loading dnn model...')
caffe_model = 'res10_300x300_ssd_iter_140000.caffemodel'
protocol_txt = 'deploy.protocol.txt'

net = cv2.dnn.readNetFromCaffe(protocol_txt, caffe_model)
print('Loading dnn successful.')

print('Starting camera ')

camera = PiCamera()
camera.resoluiton = (320, 320) # specifiy the resolution
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = (320, 320))

time.sleep(2.0)

for frame in camera.capture_continuous(rawCapture, format = 'bgr', use_video_port = True):
    image = frame.array
    image = imutils.resize(image, width= 400)
    (h,w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

    net.setInput(image)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence < 0.5:
            continue
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        text = "{:.2f}%".format(confidence * 100)
        
        
        y = startY - 10 if startY - 10 > 10 else startY + 10
        
        # vẽ box
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        
        # vẽ text
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()

