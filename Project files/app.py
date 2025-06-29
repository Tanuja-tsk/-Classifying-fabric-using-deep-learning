from flask import Flask, render_template, request, send_from_directory, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model/fabric_pattern_model.h5")

# Load class labels
with open("model/labels.txt", "r") as f:
    labels = f.read().splitlines()

# Upload folder path
UPLOAD_FOLDER = os.path.join("static", "uploaded")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    filename = file.filename
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    try:
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        label = labels[index] if index < len(labels) else "Unknown class"
    except Exception as e:
        label = f"Prediction error: {e}"

    return render_template("result.html", label=label, filename=filename)

@app.route("/uploaded/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
