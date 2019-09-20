# load dependencies
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import requests
import base64
import json
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import cv2

# TF Serving Assets
HEADERS = {'content-type': 'application/json'}
MODEL2_API_URL = 'http://localhost:8501/v1/models/fashion_model_serving/versions/2:predict'
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Instantiate Flask App
app = Flask(__name__)
CORS(app)

# Image resizing utils
def resize_image_array(img, img_size_dims):
    img = cv2.resize(img, dsize=img_size_dims, 
                     interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)
    return img

# Model warmup function
def warmup_model2_serve(warmup_data, img_dims=(32, 32)):
    warmup_data_processed = (np.array([resize_image_array(img, 
                                                          img_size_dims=img_dims) 
                                            for img in np.stack([warmup_data]*3, 
                                                                axis=-1)])) / 255.
    data = json.dumps({"signature_name": "serving_default", 
                       "instances": warmup_data_processed.tolist()})

    json_response = requests.post(MODEL2_API_URL, data=data, headers=HEADERS)
    predictions = json.loads(json_response.text)['predictions']
    print('Model 2 warmup complete') # log this in actual production

# TF lazy loads so we warmup model with sample data
# This runs as soon as we setup our web service to run
warmup_data = np.load('serve_warmup_data.npy')
warmup_model2_serve(warmup_data) 

# Liveness test
@app.route('/apparel_classifier/api/v1/liveness', methods=['GET', 'POST'])
def liveness():
    return 'API Live!'

# Model 2 inference endpoint
@app.route('/apparel_classifier/api/v1/model2_predict', methods=['POST'])
def image_classifier_model2():
    img = np.array([keras.preprocessing.image.img_to_array(
            keras.preprocessing.image.load_img(BytesIO(base64.b64decode(request.form['b64_img'])),
                                               target_size=(32, 32))) / 255.])

    data = json.dumps({"signature_name": "serving_default", 
                       "instances": img.tolist()})
    
    json_response = requests.post(MODEL2_API_URL, data=data, headers=HEADERS)
    prediction = json.loads(json_response.text)['predictions']
    prediction = np.argmax(np.array(prediction), axis=1)[0]
    prediction = CLASS_NAMES[prediction]

    return jsonify({'apparel_type': prediction})


# running REST interface, port=5000 for direct test
# use debug=True when debugging
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)