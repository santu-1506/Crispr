# model.py
import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------- Transformer building blocks ----------
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def call(self, x, training=False, mask=None):
        return self.att(x, x, x, attention_mask=mask, training=training)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        attn = self.att(x, training=training, mask=mask)
        out1 = self.norm1(x + self.drop1(attn, training=training))
        ffn = self.ffn(out1)
        out2 = self.norm2(out1 + self.drop2(ffn, training=training))
        return out2

# ---------- Model parts ----------
def CNN_branch(inputs, max_len=26):
    """
    Inception-like CNN branch using Conv2D over the one-hot input (max_len x 7).
    Steps:
    - Expand channel dimension to make input 4D.
    - Apply parallel Conv2D with kernel sizes (1,1), (2,2), (3,3), (5,5) and filter counts 5, 15, 25, 35.
    - Concatenate along channels to get 80 feature maps.
    - Collapse the width dimension (7) via max-reduction to produce (max_len, 80) for BiGRU input.
    """
    # inputs: (batch, max_len, 7)
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(inputs)  # (batch, max_len, 7, 1)

    conv1 = layers.Conv2D(5, (1, 1), padding='same', activation='relu')(x)
    conv2 = layers.Conv2D(15, (2, 2), padding='same', activation='relu')(x)
    conv3 = layers.Conv2D(25, (3, 3), padding='same', activation='relu')(x)
    conv4 = layers.Conv2D(35, (5, 5), padding='same', activation='relu')(x)

    merged = layers.Concatenate(axis=-1)([conv1, conv2, conv3, conv4])  # (batch, max_len, 7, 80)

    # Collapse width dimension (the 7 features) to match desired shape (max_len, 80)
    collapsed = layers.Lambda(lambda t: tf.reduce_max(t, axis=2))(merged)  # (batch, max_len, 80)

    # Optional explicit reshape for clarity (no-op if shapes already align)
    cnn_out = layers.Reshape((max_len, 80))(collapsed)
    return cnn_out

def bert_branch(inp, vocab_size, max_len, small_debug=False):
    embed_dim = 768 if not small_debug else 128
    num_heads = 12 if not small_debug else 4
    ff_dim = 3072 if not small_debug else 256
    num_layers = 12 if not small_debug else 2

    x = layers.Embedding(vocab_size, embed_dim)(inp)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.Dense(80, activation="relu")(x)
    return x

def build_crispr_bert_model(vocab_size, max_len, small_debug=False):
    inp_tok = layers.Input(shape=(max_len,), dtype=tf.int32)
    inp_hot = layers.Input(shape=(max_len,7), dtype=tf.float32)

    x_cnn = CNN_branch(inp_hot, max_len=max_len)
    x_bert = bert_branch(inp_tok, vocab_size, max_len, small_debug=small_debug)

    x_cnn = layers.Bidirectional(layers.GRU(40, return_sequences=False))(x_cnn)
    x_bert = layers.Bidirectional(layers.GRU(40, return_sequences=False))(x_bert)

    merged = layers.Add()([layers.Lambda(lambda z: 0.2*z)(x_cnn),
                           layers.Lambda(lambda z: 0.8*z)(x_bert)])

    x = layers.Dense(128, activation="relu")(merged)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(2, activation="softmax")(x)

    return Model([inp_tok, inp_hot], out)

