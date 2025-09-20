# detailed_train.py - Enhanced training script with comprehensive print statements
import os, glob, numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_process import load_dataset, MAX_LEN, VOCAB_SIZE
from detailed_model import build_crispr_bert_model, predict_single_example
import utils

print("=" * 100)
print("CRISPR-Cas9 Off-Target Prediction - DETAILED TRAINING PROCESS")
print("=" * 100)

def get_adaptive_settings(imbalance_ratio):
    print(f"\n⚖️  [Class Balancing] Analyzing class imbalance...")
    print(f"    Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio < 250:
        settings = (7, 3)
        print(f"    Settings: Positive ratio=7, Negative ratio=3 (High imbalance)")
    elif imbalance_ratio < 2000:
        settings = (3, 2)
        print(f"    Settings: Positive ratio=3, Negative ratio=2 (Medium imbalance)")
    else:
        settings = (1, 1)
        print(f"    Settings: Positive ratio=1, Negative ratio=1 (Balanced)")
    
    return settings

def balanced_batch_generator(X_tokens, X_onehot, y, batch_size, pos_neg_sampling):
    print(f"\n🔄 [Batch Generator] Creating balanced batch generator...")
    print(f"    Total samples: {len(y)}")
    print(f"    Batch size: {batch_size}")
    print(f"    Sampling ratios: {pos_neg_sampling}")
    
    pos_idx = np.where(y==1)[0]
    neg_idx = np.where(y==0)[0]
    pos_ratio, neg_ratio = pos_neg_sampling
    pos_n = int(batch_size*pos_ratio/(pos_ratio+neg_ratio))
    neg_n = batch_size - pos_n
    
    print(f"    Positive samples available: {len(pos_idx)}")
    print(f"    Negative samples available: {len(neg_idx)}")
    print(f"    Positive samples per batch: {pos_n}")
    print(f"    Negative samples per batch: {neg_n}")
    
    batch_count = 0
    while True:
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"    Generated {batch_count} batches...")
            
        pos_choice = np.random.choice(pos_idx, pos_n, replace=(len(pos_idx)<pos_n))
        neg_choice = np.random.choice(neg_idx, neg_n, replace=(len(neg_idx)<neg_n))
        idx = np.concatenate([pos_choice, neg_choice])
        np.random.shuffle(idx)
        
        yield (X_tokens[idx], X_onehot[idx]), y[idx]

def train_on_dataset(path, epochs=30, batch=256, small_debug=True):
    print(f"\n" + "="*100)
    print(f"🚀 [Training] Starting training on dataset: {path}")
    print(f"="*100)
    
    print(f"\n📁 [Data Loading] Loading dataset...")
    print(f"    Dataset path: {path}")
    X_tokens, X_onehot, y = load_dataset(path)
    print(f"    Dataset loaded successfully!")
    print(f"    Token sequences shape: {X_tokens.shape}")
    print(f"    One-hot matrices shape: {X_onehot.shape}")
    print(f"    Labels shape: {y.shape}")
    
    pos, neg = np.sum(y==1), np.sum(y==0)
    imb = neg/pos if pos>0 else float("inf")
    print(f"\n📊 [Dataset Statistics]")
    print(f"    Total samples: {len(y)}")
    print(f"    Positive samples (off-target): {pos}")
    print(f"    Negative samples (on-target): {neg}")
    print(f"    Imbalance ratio: {imb:.2f}")
    print(f"    Positive percentage: {(pos/len(y)*100):.2f}%")
    print(f"    Negative percentage: {(neg/len(y)*100):.2f}%")

    sampling = get_adaptive_settings(imb)
    
    print(f"\n✂️  [Data Splitting] Splitting data into train/validation...")
    print(f"    Test size: 10%")
    print(f"    Random state: 42 (for reproducibility)")
    Xt, Xv, Ot, Ov, yt, yv = train_test_split(X_tokens, X_onehot, y, test_size=0.1, stratify=y, random_state=42)
    print(f"    Training samples: {len(yt)}")
    print(f"    Validation samples: {len(yv)}")
    print(f"    Training positive: {np.sum(yt==1)}")
    print(f"    Training negative: {np.sum(yt==0)}")
    print(f"    Validation positive: {np.sum(yv==1)}")
    print(f"    Validation negative: {np.sum(yv==0)}")

    print(f"\n🏗️  [Model Building] Creating CRISPR-BERT model...")
    model = build_crispr_bert_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, small_debug=small_debug)
    
    print(f"\n⚙️  [Model Compilation] Compiling model...")
    print(f"    Optimizer: Adam")
    print(f"    Loss function: sparse_categorical_crossentropy")
    print(f"    Metrics: accuracy")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    print(f"\n📋 [Model Summary]")
    model.summary()

    print(f"\n🔧 [Callbacks Setup] Setting up training callbacks...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(f"best_model_{os.path.basename(path)}.h5", save_best_only=True, verbose=1)
    ]
    print(f"    Early stopping: patience=6, monitor='val_loss'")
    print(f"    Learning rate reduction: factor=0.5, patience=3")
    print(f"    Model checkpoint: best_model_{os.path.basename(path)}.h5")

    print(f"\n🔄 [Data Pipeline] Creating balanced batch generator...")
    gen = lambda: balanced_batch_generator(Xt, Ot, yt, batch_size=batch, pos_neg_sampling=sampling)
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32),
             tf.TensorSpec(shape=(None, MAX_LEN, 7), dtype=tf.float32)),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    steps = 30
    print(f"    Steps per epoch: {steps}")
    print(f"    Total epochs: {epochs}")

    print(f"\n🚀 [Training Start] Beginning model training...")
    print(f"    This may take several minutes...")
    
    history = model.fit(dataset, epochs=epochs, steps_per_epoch=steps,
              validation_data=([Xv, Ov], yv),
              callbacks=callbacks,
              verbose=1)

    print(f"\n✅ [Training Complete] Training finished!")
    print(f"    Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"    Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

    print(f"\n🔍 [Validation Prediction] Making predictions on validation set...")
    y_prob = model.predict([Xv, Ov], batch_size=batch, verbose=1)
    print(f"    Prediction probabilities shape: {y_prob.shape}")
    print(f"    Sample predictions: {y_prob[:5]}")
    
    print(f"\n📊 [Metrics Calculation] Computing evaluation metrics...")
    metrics = utils.compute_metrics(yv, y_prob)
    print(f"    Validation metrics for {path}:")
    for metric, value in metrics.items():
        print(f"        {metric}: {value:.4f}")
    
    return metrics, model

def demonstrate_single_prediction(model, X_tokens, X_onehot, y, sample_idx=0):
    """
    Demonstrate detailed prediction on a single example
    """
    print(f"\n" + "="*100)
    print(f"🎯 [SINGLE EXAMPLE DEMONSTRATION]")
    print(f"="*100)
    
    print(f"\n📋 [Example Selection]")
    print(f"    Selected sample index: {sample_idx}")
    print(f"    Total samples available: {len(y)}")
    
    # Get the sample
    token_sample = X_tokens[sample_idx:sample_idx+1]
    onehot_sample = X_onehot[sample_idx:sample_idx+1]
    label_sample = y[sample_idx]
    
    print(f"\n📊 [Sample Details]")
    print(f"    Token sequence: {token_sample[0]}")
    print(f"    One-hot matrix shape: {onehot_sample[0].shape}")
    print(f"    One-hot matrix (first 5 positions):")
    for i in range(min(5, onehot_sample[0].shape[0])):
        print(f"        Position {i}: {onehot_sample[0][i]}")
    print(f"    True label: {label_sample} ({'Off-target' if label_sample == 1 else 'On-target'})")
    
    # Make detailed prediction
    prediction = predict_single_example(model, token_sample, onehot_sample, label_sample)
    
    return prediction

if __name__=="__main__":
    print(f"\n🎯 [Main Execution] Starting CRISPR-BERT training and demonstration...")
    
    dataset_files = ["datasets/c.txt"]
    if not dataset_files:
        raise FileNotFoundError("No datasets found in datasets/ folder")
    
    print(f"    Dataset files to process: {dataset_files}")
    
    results = {}
    for path in dataset_files:
        try:
            print(f"\n" + "="*100)
            print(f"🔄 [Processing] {path}")
            print(f"="*100)
            
            metrics, model = train_on_dataset(path, epochs=30, batch=128, small_debug=True)
            results[os.path.basename(path)] = metrics
            
            # Demonstrate single prediction
            print(f"\n🎯 [Demonstration] Loading data for single example demonstration...")
            X_tokens, X_onehot, y = load_dataset(path)
            demonstrate_single_prediction(model, X_tokens, X_onehot, y, sample_idx=0)
            
        except Exception as e:
            print(f"❌ [Error] Skipping {path} due to error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n" + "="*100)
    print(f"📊 [FINAL SUMMARY]")
    print(f"="*100)
    for name, m in results.items():
        print(f"\n{name}:")
        for metric, value in m.items():
            print(f"    {metric}: {value:.4f}")
    
    print(f"\n✅ [Complete] All processing finished!")
    print(f"="*100)
