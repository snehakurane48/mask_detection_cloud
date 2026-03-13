from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import time
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load only CLOUD model
cloud_model = tf.keras.models.load_model("results/mask_model.keras")

import json
import os

# Global variables to store last edge data
EDGE_DATA_FILE = "last_edge_data.json"

def load_edge_data():
    if os.path.exists(EDGE_DATA_FILE):
        with open(EDGE_DATA_FILE, 'r') as f:
            data = json.load(f)
            return data.get('prediction', 'Not received'), data.get('latency', 0)
    return 'Not received', 0

def save_edge_data(prediction, latency):
    with open(EDGE_DATA_FILE, 'w') as f:
        json.dump({'prediction': prediction, 'latency': latency}, f)

last_edge_prediction, last_edge_latency = load_edge_data()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    global last_edge_prediction, last_edge_latency

    # ----- EDGE VALUES FROM EDGE DEVICE -----
    edge_prediction = request.form.get("edge_prediction")
    edge_latency = request.form.get("edge_latency")

    if edge_prediction is not None:
        last_edge_prediction = edge_prediction
        if edge_latency is not None:
            last_edge_latency = round(float(edge_latency), 2)
        save_edge_data(last_edge_prediction, last_edge_latency)
        print(f"Received edge data: {last_edge_prediction}, {last_edge_latency} ms")
    else:
        edge_prediction = last_edge_prediction
        edge_latency = last_edge_latency

    # ----- RECEIVE IMAGE -----
    if "image" in request.files and request.files["image"].filename != "":
        file = request.files["image"]

        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

    else:
        image_data = request.form.get("image")

        if not image_data or "," not in image_data:
            return "Camera image not captured properly"

        encoded = image_data.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # preprocess
    img = cv2.resize(img,(224,224))
    img = img / 255.0
    img = np.expand_dims(img,axis=0)

    # ----- CLOUD INFERENCE -----
    start = time.time()

    prediction = cloud_model.predict(img)

    cloud_latency = (time.time() - start) * 1000

    cloud_label = "Mask" if prediction[0][0] < 0.5 else "No Mask"

    # ----- SEND RESULT TO FRONTEND -----
    return render_template(
        "result.html",
        edge_prediction=edge_prediction,
        edge_latency=edge_latency,
        cloud_prediction=cloud_label,
        cloud_latency=round(cloud_latency,2)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)