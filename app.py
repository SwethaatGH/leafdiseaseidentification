from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os as os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from flask import jsonify
app = Flask(__name__)
model = load_model('/Users/swetha/Desktop/leafdetection/my_model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Call your model for prediction
        pred_class = model_predict(file_path, model)
        return jsonify(result=pred_class)
    return render_template('index.html')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    pred_class = preds.argmax(axis=-1)  # Get the index of the class with maximum probability
    labels = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
              'Blueberry___healthy','Cherry_(including_sour)___healthy','Cherry_(including_sour)___Powdery_mildew',
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
              'Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight','Grape___Black_rot','Grape___Esca_(Black_Measles)',
              'Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
              'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
              'Potato___healthy','Potato___Late_blight','Raspberry___healthy','Soybean___healthy',
              'Squash___Powdery_mildew','Strawberry___healthy','Strawberry___Leaf_scorch','Tomato___Bacterial_spot',
              'Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold',
              'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
              'Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']

    pred_label = labels[pred_class[0]]  # Get the label of the predicted class
    return pred_label

if __name__ == '__main__':
    app.run(port=5000, debug=True)
