# save as diagnostics_replay_training_files.py
import tensorflow as tf, numpy as np, os, cv2

MODEL_PATH = r"D:\himank model\models\federated_brain_tumor_model.h5"
IMG_SIZE = (224,224)
CLASS_NAMES = ["Brain_Glioma","Brain_Menin","Brain_Tumor"]  # must match flow_from_directory order

def preprocess(img_path):
    img = cv2.imread(img_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    x = (img.astype("float32")/255.0)[None,...]
    return x

def top1_confidence(probs):
    cls = int(np.argmax(probs, axis=1)[0])
    conf = float(probs[0,cls])
    return cls, conf

if __name__=="__main__":
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # IMPORTANT: switch to eval (Keras inference) – Dropout off, BN in inference
    # compile only for evaluation if needed
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    sample_files = [
        r"D:\himank model\Brain_Glioma\brain_glioma_0001.jpg",
        r"D:\himank model\Brain_Menin\brain_menin_0001.jpg",
        r"D:\himank model\Brain_Tumor\brain_tumor_0001.jpg",
    ]

    for p in sample_files:
        x = preprocess(p)
        probs = model.predict(x, verbose=0)
        cls, conf = top1_confidence(probs)
        print(os.path.basename(p), "->", CLASS_NAMES[cls], "conf:", round(conf,4))
