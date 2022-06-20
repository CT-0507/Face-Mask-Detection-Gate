import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from tensorflow.keras.applications import MobileNetV2 # load MobileNetV2 model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths

# Thiết lập learning Rate, epochs, và batch size
LR = 1e-4
EPOCHS = 20
BS = 32

dir = 'dataset' # Đường dẫn đến dataset
imgPaths = paths.list_images(dir) # Lấy đường dẫn các ảnh
data = []
labels = []

for imgPath in imgPaths:
  label = imgPath.split(os.path.sep)[-2] # with_mask, witthout_mask
  image = load_img(imgPath, target_size =(224, 224)) # tải ảnh vào bộ đệm
  image = img_to_array(image) # Biến đổi ảnh thành array
  image = preprocess_input(image) # Chuyển đổi các giá trị của ảnh cho phù hợp mobilenet

  data.append(image) # thêm ảnh vào mảng
  labels.append(label) # thêm label ứng với ảnh

data = np.array(data, dtype = 'float32')
labels = np.array(labels)

lb = LabelBinarizer() 
labels = lb.fit_transform(labels)
labels = to_categorical(labels) # Biến đổi label từ with_mask, without_mask thành 0, 1

# Ta tách dữ liệu thành 80% để train, 20% để test
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Dùng ImageDataGnerator để giúp ảnh trở nên đa dạng hơn
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Load MobileNetV2 và load weigth, và bỏ
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Thêm vào FNN của ta vào
headModel = baseModel.output
headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(128, activation="relu")(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = models.Model(inputs=baseModel.input, outputs=headModel)

# Ta phải freeze lại để model ko học nữa
for layer in baseModel.layers:
	layer.trainable = False

opt = Adam(learning_rate=LR, decay=LR / EPOCHS)
# Lựa chọn hoàm optimizer và loss cho model
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Bắt đầu quá trình train.
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Sau khi train xong ta sẽ tiến hành phân tích kết quả tranin
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

if not os.path.isdir('trained_models'):
    os.mkdir('trained_modesl/')
# save thành một folder
model.save('trained_models/face-mask-detector')
# Save định dạng .h
model.save('trained_models/face-mask-detector.h5')