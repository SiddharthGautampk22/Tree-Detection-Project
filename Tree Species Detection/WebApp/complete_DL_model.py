from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import pandas as pd
import urllib.request
from PIL import Image
from keras.models import load_model
import io
import numpy as np
import matplotlib.pyplot as plt
train_data_dir = 'D:\Office_DL_model\small_dataset\Train'
validation_data_dir = 'D:\Office_DL_model\small_dataset\Validation'
img_width, img_height = 224, 224
batch_size = 10
epochs = 32
train_datagen = ImageDataGenerator(
    rescale = 1. / 225,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1. / 225)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')
# Specifies a neural network, create a neural network from input to output layer completely.
model = Sequential()
# Conv2D performs convolution to the input image to extract features it has 32 filters applied of size 3x3 and relu as activation function.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
# Maxpooling layer is used to downsample the feature maps produced by the previous layer.
model.add(MaxPooling2D((2, 2)))

# Second convolutional layer with 64 filters of 3x3 size and relu activation function.
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Third convolutional layer or final layer with 128 filtes and all of the things the same as the above two layers.
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output of the convolutional layers from a 2D array to a single dimensional array.
model.add(Flatten())

# Dense layers performs linear transformations of input data with 64 neurons and softmax activtion function. 
model.add(Dense(512, activation='relu'))
# Dropout layer is applied to prevent the model from overfitting.
model.add(Dropout(0.5))
# softmax activation function is used for multi-class classification.
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the generators
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size)
model_path = "small_model.h5"
model = model.save(model_path)
common_names = pd.read_excel(r'D:\Office_DL_model\small_dataset(10).xlsx')
common_names = common_names.values.tolist()
model = load_model("small_model.h5")
img_width, img_height = 224, 224
def predict_class(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((img_width, img_height))
    img = np.array(img) / 255.
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    return class_idx, class_names[class_idx]

class_names = common_names
url_or_path = input("Enter image URL or local file path: ")

if url_or_path.startswith(('http', 'https', 'ftp')):
    with urllib.request.urlopen(url_or_path) as url:
        img = url.read()
        class_idx, class_name = predict_class(img)
    image = Image.open(io.BytesIO(img))
else:
    class_idx, class_name = predict_class(open(url_or_path, 'rb').read())
    image = Image.open(url_or_path)

plt.imshow(image)
plt.axis('off')
# plt.title(f"Predicted class : {class_name}")
# # plt.title(f"Predicted common name: {comm_name}")
# plt.show()
# plt.title(f"Common name for predicted class : {comm_name}")
# plt.show()
plt.suptitle("Prediction made", fontsize = 14, fontweight = 'bold')
plt.title(f"Class name and common name : {class_name}", fontsize = 12)
plt.show()