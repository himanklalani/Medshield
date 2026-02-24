from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image


app = Flask(__name__)
CORS(app)


model = tf.keras.models.load_model("models/federated_brain_tumor_model.h5")
CLASSES = ["Brain_Glioma", "Brain_Menin", "Brain_Tumor"]


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400

    # Load and preprocess
    img = Image.open(file.stream).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")  # Removei dvision by 255 her
    arr = arr[None, ...]  # add batch dimension

    pred = model.predict(arr)
    idx = int(np.argmax(pred[0]))
    return jsonify({
        "class": CLASSES[idx],
        "confidence": float(pred[0][idx]) * 100
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
