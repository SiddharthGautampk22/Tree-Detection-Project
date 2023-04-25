from flask import Flask, request, jsonify, render_template
import urllib.request
from PIL import Image
import pandas as pd
import numpy as np
import io
from keras.models import load_model
app = Flask(__name__)
img_width, img_height = 224, 224
model = None
class_names = None
tree_detect_model = load_model("Tree_detection_model_new_2.h5")
tree_or_not = ["Not a Tree" , " Tree "]
def tree_detect_file(img):
    img_np = img
    img_np = img_np.resize((img_width, img_height))
    img_np = np.array(img_np) / 255.
    img_np = np.expand_dims(img_np, axis = 0)
    tree_pred = tree_detect_model.predict(img_np)
    tree_idx = np.argmax(tree_pred)
    if tree_idx == 1:
        print("Uploaded image is that of a tree.")
        return predict_class(img)
    else:
        print("Uploaded image is not of a tree.")
def tree_detect_url(img):
    img_np = img
    img_np = img_np.resize((img_width, img_height))
    img_np = np.array(img_np) / 255.
    img_np = np.expand_dims(img_np, axis = 0)
    tree_pred = tree_detect_model.predict(img_np)
    tree_idx = np.argmax(tree_pred)
    if tree_idx == 1:
        print("Uploaded image is that of a tree.")
        return predict_class(img)
    else:
        print("Uploaded image is not of a tree.")
def predict_class(img, n = 3):
    global model
    global class_names
    checked_img = img
    # checked_img = Image.open(io.BytesIO(img))
    checked_img = checked_img.resize((img_width, img_height))
    checked_img = np.array(checked_img) / 255.
    checked_img = np.expand_dims(checked_img, axis = 0)
    pred = model.predict(checked_img)[0]
    top_n_indices = pred.argsort()[-n:][::-1]
    #Selects the top n predicted probabilities using the indices.
    top_n_probs = pred[top_n_indices]
    #Maps the top n indices to their corresponding class names.
    top_n_classes = [class_names[i] for i in top_n_indices]
    return pred_result(top_n_classes, top_n_probs)
def pred_result(top_n_classes, top_n_probs):
    top_n_classes = top_n_classes
    top_n_probs = top_n_probs
    print(top_n_classes)
    print(top_n_probs)
    return top_n_classes, top_n_probs
@app.route('/' , methods = ['GET' , 'POST'])
def index():
    # flash('This is an alert message', 'alert-success')
    return render_template('index.html')
@app.route('/model_selection' , methods = ['POST'])
def model_selection():
    global model
    global class_names
    select = request.form['model-info']
    model = load_model(select)
    if select == "small_model.h5":
        web_name= "Small Model"
        common_names = pd.read_excel(r'D:\Office_DL_model\small_dataset(10).xlsx')
        class_names = common_names.values.tolist()
    elif select == "smaller_model.h5":
        web_name= "Smaller Model"
        common_names = pd.read_excel(r'D:\Office_DL_model\WebApp\smaller_dataset.xlsx')
        class_names = common_names.values.tolist()
    elif select == "super_final_model.h5":
        web_name= "Super Final Model"
        common_names = pd.read_excel(r'D:\Office_DL_model\WebApp\super_final_dataset.xlsx')
        class_names = common_names.values.tolist()
    elif select == "new_model.h5":
        web_name = "New Model"
        common_names = pd.read_excel(r'D:\Office_DL_model\WebApp\new_dataset.xlsx')
        class_names = common_names.values.tolist()
    else:
        web_name = "New Model 1"
        common_names = pd.read_excel(r'D:\Office_DL_model\WebApp\new_dataset.xlsx')
        class_names = common_names.values.tolist()
    return render_template('index.html' , web_name=web_name)
@app.route('/predict', methods=['POST'])
def predict():
    if request.files.get('file'):
        # If user uploaded a file, read it and call predict_class function
        req = request.files['file'].read()
        print(type(req))
        img = Image.open(io.BytesIO(req))
        top_n_classes, top_n_probs = tree_detect_file(img) 

        # image = Image.open(io.BytesIO(img))
        # image.save("static/test.png", "png")
    else:
        # If user provided a URL, read it and call predict_class function
        url_or_path = request.form['url']
        if url_or_path.startswith(('http' , 'https' , 'ftp')):
            req = urllib.request.urlopen(url_or_path)
            img = req.read()
            # print("img before BytesIO", img)
            img = Image.open(io.BytesIO(img))    
        top_n_classes, top_n_probs = tree_detect_url(img)


            
        
        # with urllib.request.urlopen(url) as url:
        #     img = url.read()
        #     top_n_classes, top_n_probs = predict_class(img)
        # image = Image.open(io.BytesIO(img))
        # image.save("static/test.png" , "png")
    results = []
    for class_name, prob in zip(top_n_classes, top_n_probs):
        result = {'class' : class_name, 'probability' : float(prob*100)}
        results.append(result)
    return jsonify(results=results)


if __name__ == '__main__':
    app.run(debug=True)


