from keras.models import load_model
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
class_names = ["Not a Tree" , "Tree"]
model = load_model("Tree_detection_model.h5")
img_width, img_height = 224, 224
def tree_detection(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((img_width, img_height))
    img = np.array(img) / 255.
    img = np.expand_dims(img, axis = 0)
    tree_detect  = model.predict(img)
    tree_idx = np.argmax(tree_detect)
    return tree_idx, class_names[tree_idx]
url_or_path = input("Enter web URL or local file path : ")
if url_or_path.startswith(('http', 'https', 'ftp')):
    with urllib.request.urlopen(url_or_path) as url:
        img = url.read()
        tree_idx, class_names = tree_detection(img)
    image = Image.open(io.BytesIO(img))
else:
    tree_idx, class_names = tree_detection(open(url_or_path, 'rb').read())
    image = Image.open(url_or_path)
plt.imshow(image)
plt.axis('off')
plt.suptitle("Prediction made ", fontsize = 14, fontweight = 'bold')
plt.title(f"Tree or not : {class_names}", fontsize= 12)
plt.show()
