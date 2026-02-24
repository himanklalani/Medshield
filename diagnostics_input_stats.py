# save as diagnostics_input_stats.py
import numpy as np, tensorflow as tf, os, cv2, json

IMG_SIZE = (224, 224)
MEAN_STD_TRAIN = {"mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]}  # if you used ImageNet normalization after Rescaling(1/255)

def load_and_preprocess(path):
    # replicate Streamlit/frontend pipeline exactly; adjust if you use PIL etc.
    img = cv2.imread(path, cv2.IMREAD_COLOR)              # BGR 0–255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            # RGB
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    x = img.astype("float32")/255.0                       # Rescaling(1./255)
    # If you added extra normalization in the app, apply it here too
    return x

def batch_stats(filepaths, batch_size=32):
    xs = []
    for p in filepaths[:batch_size]:
        xs.append(load_and_preprocess(p))
    x = np.stack(xs, axis=0)                              # (B,224,224,3)
    mean = x.mean(axis=(0,1,2)).tolist()
    std  = x.std(axis=(0,1,2)).tolist()
    return mean, std

if __name__ == "__main__":
    # sample a mini‑batch from each class folder you serve to the app
    roots = [
        r"D:\himank model\Brain_Glioma",
        r"D:\himank model\Brain_Menin",
        r"D:\himank model\Brain_Tumor"
    ]
    files = []
    for r in roots:
        for f in os.listdir(r):
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                files.append(os.path.join(r,f))
        # sample up to ~12 per class
        files = files[:12] if len(files)>12 else files

    mean, std = batch_stats(files, batch_size=min(32, len(files)))
    print("Frontend batch mean:", np.round(mean,4))
    print("Frontend batch std :", np.round(std,4))
    print("Training pipeline mean (after 1/255):", MEAN_STD_TRAIN["mean"])
    print("Training pipeline std  (after 1/255):", MEAN_STD_TRAIN["std"])
    # If large deviation (>~0.05 per channel), you have preprocessing drift
