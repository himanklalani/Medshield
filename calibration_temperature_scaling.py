# save as calibration_temperature_scaling.py
import tensorflow as tf, numpy as np, json, os, matplotlib.pyplot as plt

MODEL_PATH = r"D:\himank model\models\federated_brain_tumor_model.h5"
VAL_DIR    = r"D:\himank model\federated_data\hospital_1\val"   # any held‑out split
IMG_SIZE   = (224,224); BATCH=32

def load_val_generator():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    gen = datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="categorical", shuffle=False
    )
    return gen

def logits_from_model(model, gen):
    # Get pre‑softmax activations by wrapping model without final softmax
    # Your model ends with Dense(num_classes, activation='softmax'), so:
    # create a new model from inputs to the Dense pre-activation
    penultimate = model.layers[-1]                # Dense softmax
    logits_model = tf.keras.Model(inputs=model.input,
                                  outputs=penultimate.input)  # pre-softmax tensor
    zs, ys = [], []
    steps = int(np.ceil(gen.n/gen.batch_size))
    for _ in range(steps):
        xb, yb = next(gen)
        z = logits_model.predict(xb, verbose=0)
        zs.append(z); ys.append(yb)
    return np.vstack(zs), np.vstack(ys)

def nll_with_temperature(z, y, T):
    zT = z / T
    log_probs = zT - tf.reduce_logsumexp(zT, axis=1, keepdims=True)
    nll = -tf.reduce_sum(y * log_probs, axis=1)
    return float(tf.reduce_mean(nll).numpy())

def fit_temperature(z, y, init_T=1.0, lr=0.01, iters=500):
    T = tf.Variable(init_T, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(lr)
    for _ in range(iters):
        with tf.GradientTape() as tape:
            loss = nll_with_temperature(z, y, tf.nn.softplus(T)+1e-6)  # keep T>0
        grads = tape.gradient(loss, [T]); opt.apply_gradients(zip(grads, [T]))
    return float(tf.nn.softplus(T).numpy())

def apply_temperature_softmax(z, T):
    zT = z / T
    e = np.exp(zT - np.max(zT, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def reliability_diagram(probs, labels, bins=10, title="Reliability Diagram"):
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    true = labels.argmax(axis=1)
    accs, confs = [], []
    edges = np.linspace(0,1,bins+1)
    for i in range(bins):
        mask = (conf>=edges[i]) & (conf<edges[i+1])
        if mask.sum()==0: continue
        bin_acc = (preds[mask]==true[mask]).mean()
        bin_conf = conf[mask].mean()
        accs.append(bin_acc); confs.append(bin_conf)
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],'k--',label="Perfect")
    plt.bar(confs, np.array(accs)-np.array(confs), width=0.09, bottom=confs, alpha=0.4, label="Gap")
    plt.scatter(confs, accs, c='b', label="Bins")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

if __name__=="__main__":
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    gen = load_val_generator()
    z, y = logits_from_model(model, gen)
    # Fit temperature
    T_opt = fit_temperature(z, y, init_T=1.0, lr=0.05, iters=300)
    print("Optimal temperature:", round(T_opt,3))
    # Before/after calibration ECE via reliability diagram
    probs_raw = apply_temperature_softmax(z, T=1.0)
    probs_cal = apply_temperature_softmax(z, T=T_opt)
    reliability_diagram(probs_raw, y, title="Before calibration")
    reliability_diagram(probs_cal, y, title=f"After temperature scaling (T={T_opt:.2f})")
    # Save T for frontend use
    with open(r"D:\himank model\models\temperature.json","w") as f:
        json.dump({"T":T_opt}, f, indent=2)
