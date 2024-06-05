from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from keras import layers
import numpy as np
from custom_patches import Patches
from custom_patchencoder import PatchEncoder
from keras.layers import Input

app = Flask(__name__)

input_shape = [240, 320, 3]
dic = {0: 'Eosinofil', 1: 'Limfosit', 2: 'Monosit', 3: 'Neutrofil'}

model = load_model('vit_rev_new.h5')

def predict_label(img_path):
    input = image.load_img(img_path, target_size=input_shape)
    input = image.img_to_array(input)
    input = np.expand_dims(input, axis=0)
    resized_img = tf.image.resize(input, size=(240,320))
    prediction = model.predict(resized_img)[0]
    predicted_class = np.argmax(prediction)
    predicted_label = dic[predicted_class]
    predicted_proba = int(round(prediction[predicted_class]))
    other_probas = {label: abs(int(round(prob))) for label, prob in zip(dic.values(), prediction) if label != predicted_label}
    return predicted_label, predicted_proba, other_probas
    # prediction = model.predict(resized_img)
    # predicted_class = np.argmax(prediction, axis=1)[0]
    # return dic[predicted_class]


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def get_output():
    if request.method == 'POST':
         img = request.files['my_image']

         img_path = "./static/" + img.filename
         img.save(img_path)

         p, proba, other_probas = predict_label(img_path)
        
    return render_template("index.html", prediction = p, probability=proba, other_probas=other_probas, img_path = img_path)

if __name__ == '__main__':
    app.run(debug=False)
