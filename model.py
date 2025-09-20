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
def inception_cnn_branch(inp, max_len=26):
    convs = []
    for k in [5,15,25,35]:
        c = layers.Conv1D(filters=80, kernel_size=k, padding="same", activation="relu")(inp)
        convs.append(c)
    x = layers.Concatenate()(convs)
    x = layers.Dense(80, activation="relu")(x)
    return x

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

    x_cnn = inception_cnn_branch(inp_hot, max_len=max_len)
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
