import os
import urllib.request
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd


def load_image_classifier_model(model_path):
    model = load_model(model_path)
    return model


# def get_class_names(class_name):
#     folders = []
#     for dirpath, dirnames, filenames in os.walk(folder_path):
#         for dirname in dirnames:
#             folders.append(dirname)
#     return folders


def predict_class(model, img, class_names, img_width=224, img_height=224):
    img = Image.open(io.BytesIO(img))
    img = img.resize((img_width, img_height))
    img = np.array(img) / 255.
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    return class_idx, class_names[class_idx]


def show_image_prediction(url_or_path, model, class_names):
    if url_or_path.startswith(('http', 'https', 'ftp')):
        with urllib.request.urlopen(url_or_path) as url:
            img = url.read()
            class_idx, class_name = predict_class(model, img, class_names)
        image = Image.open(io.BytesIO(img))
        image.save("test.png","png")
        return class_name
    else:
        class_idx, class_name = predict_class(model, open(url_or_path, 'rb').read(), class_names)
        image = Image.open(url_or_path)
        image.save("test.png", "png")

        
    plt.imshow(image)
    plt.axis('off')
    plt.suptitle("Prediction made", fontsize = 14, fontweight = 'bold')
    plt.title(f"Class name and common name : {class_name}", fontsize = 12)
    plt.show()



# Example usage:
model_path = 'final_model.h5'
model = load_image_classifier_model(model_path)

names = pd.read_excel(r'C:\Users\siddh\Downloads\common_name.xlsx')
class_names = names.values.tolist()

# url_or_path = input("Enter image URL or local file path: ")
# show_image_prediction(url_or_path, model, class_names)
