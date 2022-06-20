import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import pathlib

if not os.path.isdir('trained_models'):
    os.mkdir('trained_models/')

model_dir = 'trained_models/face-mask-detector/'
# Load lại model từ đường dẫn
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
# Chuyển đổi model tensorflow thành tensorflow lite
tflite_model = converter.convert()
# Lưu lại file tensorflow.lite
tflite_model_file = pathlib.Path('trained_models/model.tflite')
tflite_model_file.write_bytes(tflite_model)