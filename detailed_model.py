# detailed_model.py - Enhanced version with comprehensive print statements
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

print("=" * 80)
print("CRISPR-Cas9 Off-Target Prediction Model - Detailed Processing")
print("=" * 80)

# ---------- Transformer building blocks ----------
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        print(f"    [MultiHeadSelfAttention] Initialized with embed_dim={embed_dim}, num_heads={num_heads}")

    def call(self, x, training=False, mask=None):
        print(f"    [MultiHeadSelfAttention] Input shape: {x.shape}")
        print(f"    [MultiHeadSelfAttention] Computing self-attention...")
        result = self.att(x, x, x, attention_mask=mask, training=training)
        print(f"    [MultiHeadSelfAttention] Output shape: {result.shape}")
        return result

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        print(f"    [TransformerBlock] Initializing with embed_dim={embed_dim}, num_heads={num_heads}, ff_dim={ff_dim}")
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)
        print(f"    [TransformerBlock] Feed-forward network: {ff_dim} -> {embed_dim}")

    def call(self, x, training=False, mask=None):
        print(f"    [TransformerBlock] Input shape: {x.shape}")
        
        # Self-attention
        print(f"    [TransformerBlock] Step 1: Computing self-attention...")
        attn = self.att(x, training=training, mask=mask)
        print(f"    [TransformerBlock] Step 2: Applying dropout and residual connection...")
        out1 = self.norm1(x + self.drop1(attn, training=training))
        print(f"    [TransformerBlock] Step 3: Feed-forward network...")
        ffn = self.ffn(out1)
        print(f"    [TransformerBlock] Step 4: Final dropout and residual connection...")
        out2 = self.norm2(out1 + self.drop2(ffn, training=training))
        print(f"    [TransformerBlock] Output shape: {out2.shape}")
        return out2

# ---------- Model parts ----------
def inception_cnn_branch(inp, max_len=26):
    print(f"\n🔬 [CNN Branch] Starting Inception-style CNN processing...")
    print(f"    [CNN Branch] Input shape: {inp.shape}")
    print(f"    [CNN Branch] Max sequence length: {max_len}")
    
    convs = []
    kernel_sizes = [5, 15, 25, 35]
    print(f"    [CNN Branch] Using kernel sizes: {kernel_sizes}")
    
    for i, k in enumerate(kernel_sizes):
        print(f"    [CNN Branch] Conv1D {i+1}/4: kernel_size={k}, filters=80, padding='same'")
        c = layers.Conv1D(filters=80, kernel_size=k, padding="same", activation="relu")(inp)
        print(f"    [CNN Branch] Conv1D {i+1}/4 output shape: {c.shape}")
        convs.append(c)
    
    print(f"    [CNN Branch] Concatenating {len(convs)} convolution outputs...")
    x = layers.Concatenate()(convs)
    print(f"    [CNN Branch] Concatenated shape: {x.shape}")
    
    print(f"    [CNN Branch] Dense layer: 80 units with ReLU activation...")
    x = layers.Dense(80, activation="relu")(x)
    print(f"    [CNN Branch] Final CNN output shape: {x.shape}")
    return x

def bert_branch(inp, vocab_size, max_len, small_debug=False):
    print(f"\n🤖 [BERT Branch] Starting Transformer-based BERT processing...")
    print(f"    [BERT Branch] Input shape: {inp.shape}")
    print(f"    [BERT Branch] Vocabulary size: {vocab_size}")
    print(f"    [BERT Branch] Max sequence length: {max_len}")
    print(f"    [BERT Branch] Small debug mode: {small_debug}")
    
    embed_dim = 768 if not small_debug else 128
    num_heads = 12 if not small_debug else 4
    ff_dim = 3072 if not small_debug else 256
    num_layers = 12 if not small_debug else 2
    
    print(f"    [BERT Branch] Configuration: embed_dim={embed_dim}, num_heads={num_heads}, ff_dim={ff_dim}, num_layers={num_layers}")
    
    print(f"    [BERT Branch] Step 1: Token embedding...")
    x = layers.Embedding(vocab_size, embed_dim)(inp)
    print(f"    [BERT Branch] Embedding output shape: {x.shape}")
    
    print(f"    [BERT Branch] Step 2: Processing through {num_layers} Transformer blocks...")
    for i in range(num_layers):
        print(f"    [BERT Branch] Transformer Block {i+1}/{num_layers}:")
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
    print(f"    [BERT Branch] Step 3: Final dense layer...")
    x = layers.Dense(80, activation="relu")(x)
    print(f"    [BERT Branch] Final BERT output shape: {x.shape}")
    return x

def build_crispr_bert_model(vocab_size, max_len, small_debug=False):
    print(f"\n🏗️  [Model Builder] Building CRISPR-BERT model...")
    print(f"    [Model Builder] Vocabulary size: {vocab_size}")
    print(f"    [Model Builder] Max sequence length: {max_len}")
    print(f"    [Model Builder] Small debug mode: {small_debug}")
    
    print(f"\n📥 [Input Layer] Creating input layers...")
    inp_tok = layers.Input(shape=(max_len,), dtype=tf.int32, name="token_input")
    inp_hot = layers.Input(shape=(max_len,7), dtype=tf.float32, name="onehot_input")
    print(f"    [Input Layer] Token input shape: {inp_tok.shape}")
    print(f"    [Input Layer] One-hot input shape: {inp_hot.shape}")

    print(f"\n🔄 [Branch Processing] Processing both branches...")
    x_cnn = inception_cnn_branch(inp_hot, max_len=max_len)
    x_bert = bert_branch(inp_tok, vocab_size, max_len, small_debug=small_debug)

    print(f"\n🔄 [GRU Processing] Bidirectional GRU layers...")
    print(f"    [GRU] CNN branch GRU: 40 units, bidirectional...")
    x_cnn = layers.Bidirectional(layers.GRU(40, return_sequences=False))(x_cnn)
    print(f"    [GRU] CNN branch GRU output shape: {x_cnn.shape}")
    
    print(f"    [GRU] BERT branch GRU: 40 units, bidirectional...")
    x_bert = layers.Bidirectional(layers.GRU(40, return_sequences=False))(x_bert)
    print(f"    [GRU] BERT branch GRU output shape: {x_bert.shape}")

    print(f"\n🔗 [Fusion Layer] Combining branches with weighted fusion...")
    print(f"    [Fusion] CNN weight: 0.2, BERT weight: 0.8")
    merged = layers.Add()([layers.Lambda(lambda z: 0.2*z)(x_cnn),
                           layers.Lambda(lambda z: 0.8*z)(x_bert)])
    print(f"    [Fusion] Merged output shape: {merged.shape}")

    print(f"\n🎯 [Classification Head] Building classification layers...")
    print(f"    [Classification] Dense layer 1: 128 units, ReLU activation...")
    x = layers.Dense(128, activation="relu")(merged)
    print(f"    [Classification] Dense layer 1 output shape: {x.shape}")
    
    print(f"    [Classification] Dense layer 2: 64 units, ReLU activation...")
    x = layers.Dense(64, activation="relu")(x)
    print(f"    [Classification] Dense layer 2 output shape: {x.shape}")
    
    print(f"    [Classification] Dropout layer: 0.35 rate...")
    x = layers.Dropout(0.35)(x)
    print(f"    [Classification] Dropout output shape: {x.shape}")
    
    print(f"    [Classification] Final output layer: 2 units, softmax activation...")
    out = layers.Dense(2, activation="softmax")(x)
    print(f"    [Classification] Final output shape: {out.shape}")

    print(f"\n✅ [Model Complete] Model built successfully!")
    return Model([inp_tok, inp_hot], out)

def predict_single_example(model, token_input, onehot_input, label=None):
    """
    Predict on a single example with detailed step-by-step output
    """
    print(f"\n" + "="*80)
    print(f"🔍 SINGLE EXAMPLE PREDICTION - DETAILED ANALYSIS")
    print(f"="*80)
    
    print(f"\n📊 [Input Analysis]")
    print(f"    Token input shape: {token_input.shape}")
    print(f"    One-hot input shape: {onehot_input.shape}")
    print(f"    Token sequence: {token_input[0]}")
    print(f"    One-hot matrix shape: {onehot_input[0].shape}")
    if label is not None:
        print(f"    True label: {label}")
    
    print(f"\n🚀 [Prediction Process]")
    print(f"    Running forward pass through the model...")
    
    # Get intermediate outputs for detailed analysis
    print(f"\n🔬 [CNN Branch Analysis]")
    try:
        cnn_branch = Model(model.inputs, model.get_layer('concatenate_1').output)
        cnn_output = cnn_branch([token_input, onehot_input])
        print(f"    CNN branch output shape: {cnn_output.shape}")
        print(f"    CNN branch sample values: {cnn_output[0][:5]}")
    except:
        print(f"    CNN branch analysis skipped (layer name issue)")
    
    print(f"\n🤖 [BERT Branch Analysis]")
    try:
        # We need to get the BERT output before GRU
        bert_embedding = model.get_layer('embedding_1').output
        bert_transformer = Model(model.inputs, bert_embedding)
        bert_emb = bert_transformer([token_input, onehot_input])
        print(f"    BERT embedding shape: {bert_emb.shape}")
        print(f"    BERT embedding sample values: {bert_emb[0][0][:5]}")
    except:
        print(f"    BERT branch analysis skipped (layer name issue)")
    
    print(f"\n🔄 [GRU Processing Analysis]")
    try:
        # Get outputs after GRU layers
        cnn_gru = Model(model.inputs, model.get_layer('bidirectional_2').output)
        cnn_gru_out = cnn_gru([token_input, onehot_input])
        print(f"    CNN GRU output shape: {cnn_gru_out.shape}")
        print(f"    CNN GRU sample values: {cnn_gru_out[0][:5]}")
        
        bert_gru = Model(model.inputs, model.get_layer('bidirectional_3').output)
        bert_gru_out = bert_gru([token_input, onehot_input])
        print(f"    BERT GRU output shape: {bert_gru_out.shape}")
        print(f"    BERT GRU sample values: {bert_gru_out[0][:5]}")
        
        print(f"\n🔗 [Fusion Analysis]")
        print(f"    CNN contribution (0.2x): {cnn_gru_out[0][:5] * 0.2}")
        print(f"    BERT contribution (0.8x): {bert_gru_out[0][:5] * 0.8}")
        fusion_layer = Model(model.inputs, model.get_layer('add_1').output)
        fusion_out = fusion_layer([token_input, onehot_input])
        print(f"    Fused output shape: {fusion_out.shape}")
        print(f"    Fused sample values: {fusion_out[0][:5]}")
    except:
        print(f"    GRU and fusion analysis skipped (layer name issue)")
    
    print(f"\n🎯 [Classification Analysis]")
    try:
        dense1_layer = Model(model.inputs, model.get_layer('dense_15').output)
        dense1_out = dense1_layer([token_input, onehot_input])
        print(f"    Dense layer 1 output shape: {dense1_out.shape}")
        print(f"    Dense layer 1 sample values: {dense1_out[0][:5]}")
        
        dense2_layer = Model(model.inputs, model.get_layer('dense_16').output)
        dense2_out = dense2_layer([token_input, onehot_input])
        print(f"    Dense layer 2 output shape: {dense2_out.shape}")
        print(f"    Dense layer 2 sample values: {dense2_out[0][:5]}")
    except:
        print(f"    Classification analysis skipped (layer name issue)")
    
    print(f"\n🎲 [Final Prediction]")
    prediction = model.predict([token_input, onehot_input], verbose=0)
    print(f"    Raw prediction probabilities: {prediction[0]}")
    print(f"    Predicted class: {np.argmax(prediction[0])}")
    print(f"    Confidence: {np.max(prediction[0]):.4f}")
    
    if label is not None:
        correct = "✅ CORRECT" if np.argmax(prediction[0]) == label else "❌ INCORRECT"
        print(f"    Prediction result: {correct}")
    
    print(f"\n" + "="*80)
    return prediction
