# train.py (patched for TF 2.20 / Keras 3)
import os, glob, numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_process import load_dataset, MAX_LEN, VOCAB_SIZE
from model import build_crispr_bert_model
import utils

def get_adaptive_settings(imbalance_ratio):
    if imbalance_ratio < 250:
        return (7,3)
    elif imbalance_ratio < 2000:
        return (3,2)
    else:
        return (1,1)

def balanced_batch_generator(X_tokens, X_onehot, y, batch_size, pos_neg_sampling):
    pos_idx = np.where(y==1)[0]; neg_idx = np.where(y==0)[0]
    pos_ratio, neg_ratio = pos_neg_sampling
    pos_n = int(batch_size*pos_ratio/(pos_ratio+neg_ratio)); neg_n = batch_size-pos_n
    while True:
        pos_choice = np.random.choice(pos_idx, pos_n, replace=(len(pos_idx)<pos_n))
        neg_choice = np.random.choice(neg_idx, neg_n, replace=(len(neg_idx)<neg_n))
        idx = np.concatenate([pos_choice, neg_choice]); np.random.shuffle(idx)
        yield (X_tokens[idx], X_onehot[idx]), y[idx]

def train_on_dataset(path, epochs=30, batch=256, small_debug=True):
    print(f"\n=== Training on {path} ===")
    X_tokens, X_onehot, y = load_dataset(path)
    pos, neg = np.sum(y==1), np.sum(y==0)
    imb = neg/pos if pos>0 else float("inf")
    print(f"Samples: {len(y)}, pos={pos}, neg={neg}, imbalance={imb:.2f}")

    sampling = get_adaptive_settings(imb)
    Xt, Xv, Ot, Ov, yt, yv = train_test_split(X_tokens, X_onehot, y, test_size=0.1, stratify=y, random_state=42)

    model = build_crispr_bert_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, small_debug=small_debug)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ModelCheckpoint(f"best_model_{os.path.basename(path)}.h5", save_best_only=True)
    ]

    # Create a tf.data.Dataset from the generator
    gen = lambda: balanced_batch_generator(Xt, Ot, yt, batch_size=batch, pos_neg_sampling=sampling)
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32),
             tf.TensorSpec(shape=(None, MAX_LEN, 7), dtype=tf.float32)),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    steps = 30  # Reduced from max(1, len(yt)//batch) to limit steps per epoch

    model.fit(dataset, epochs=epochs, steps_per_epoch=steps,
              validation_data=([Xv, Ov], yv),
              callbacks=callbacks,
              verbose=1)

    y_prob = model.predict([Xv, Ov], batch_size=batch)
    metrics = utils.compute_metrics(yv, y_prob)
    print(f"Validation metrics for {path}:", metrics)
    return metrics

if __name__=="__main__":
    dataset_files = ["datasets/c.txt"]
    if not dataset_files:
        raise FileNotFoundError("No datasets found in datasets/ folder")
    results = {}
    for path in dataset_files:
        try:
            metrics = train_on_dataset(path, epochs=30, batch=128, small_debug=True)
            results[os.path.basename(path)] = metrics
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")

    print("\n=== Summary ===")
    for name, m in results.items():
        print(name, ":", m)
