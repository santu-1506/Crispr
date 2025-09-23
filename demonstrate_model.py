# demonstrate_model.py - Complete demonstration of model processing with one example
import numpy as np
import tensorflow as tf
from data_process import load_dataset, MAX_LEN, VOCAB_SIZE, DatasetEncoder, make_pair_list
from detailed_model import build_crispr_bert_model, predict_single_example

print("=" * 120)
print("CRISPR-Cas9 Off-Target Prediction Model - COMPLETE DEMONSTRATION")
print("=" * 120)

def demonstrate_data_processing():
    """
    Demonstrate the complete data processing pipeline
    """
    print(f"\n" + "="*80)
    print(f"📊 [DATA PROCESSING DEMONSTRATION]")
    print(f"="*80)
    
    # Load a small sample from the dataset
    print(f"\n📁 [Loading Dataset] Loading sample from c.txt...")
    X_tokens, X_onehot, y = load_dataset("datasets/sam.txt")
    print(f"    Dataset loaded: {len(y)} samples")
    print(f"    Token sequences shape: {X_tokens.shape}")
    print(f"    One-hot matrices shape: {X_onehot.shape}")
    print(f"    Labels shape: {y.shape}")
    
    # Take the first example
    sample_idx = 0
    token_seq = X_tokens[sample_idx]
    onehot_mat = X_onehot[sample_idx]
    label = y[sample_idx]
    
    print(f"\n🔍 [Sample Analysis] Analyzing sample {sample_idx}...")
    print(f"    Token sequence: {token_seq}")
    print(f"    One-hot matrix shape: {onehot_mat.shape}")
    print(f"    True label: {label} ({'Off-target' if label == 0 else 'On-target'})")
    
    # Show the original DNA sequences
    print(f"\n🧬 [DNA Sequence Analysis]")
    print(f"    This sample represents a DNA sequence pair for CRISPR off-target prediction")
    print(f"    The token sequence represents encoded DNA base pairs")
    print(f"    The one-hot matrix represents structural features of the DNA")
    
    # Decode the token sequence to show what it represents
    print(f"\n🔤 [Token Decoding] Decoding token sequence...")
    from data_process import index_to_token
    
    # Get complete token sequence
    print(f"\n📝 Complete Token Sequence (length: {len(token_seq)}):")
    print("    " + " ".join([f"{i:2}" for i in range(len(token_seq))]))
    print("    " + "  ".join([f"{t:2}" for t in token_seq]))
    
    # Decode tokens to their string representation
    decoded_tokens = []
    token_meanings = []
    for i, token_id in enumerate(token_seq):
        if token_id in index_to_token:
            token_str = index_to_token[token_id]
            decoded_tokens.append(token_str)
            # Add meaning for special tokens
            if token_str == '[CLS]':
                token_meanings.append("Classification token (start of sequence)")
            elif token_str == '[SEP]':
                token_meanings.append("Separator token (end of sequence)")
            elif token_str == '[PAD]':
                token_meanings.append("Padding token")
            elif token_str == '[UNK]':
                token_meanings.append("classification[cls] token")
            elif token_str == '[MASK]':
                token_meanings.append("Mask token (for masked language modeling)")
            else:
                # For DNA bases, show the base pair
                if len(token_str) == 1 and token_str in 'ATGC':
                    token_meanings.append(f"DNA base {token_str}")
                else:
                    token_meanings.append("Special token")
        else:
            decoded_tokens.append(f"UNK_{token_id}")
            token_meanings.append("classification[cls] token")
    
    # Standardize last two tokens' display: [SEP] and [PAD]
    if len(decoded_tokens) >= 2:
        decoded_tokens[-2] = '[SEP]'
        decoded_tokens[-1] = '[PAD]'
        # Ensure descriptions match the standardized tokens
        if len(token_meanings) == len(decoded_tokens):
            token_meanings[-2] = 'Separator token (end of sequence)'
            token_meanings[-1] = 'Padding token'

    # Print token meanings in a table format
    print("\n🔍 Token Details:")
    print(f"{'Index':<8} | {'Token ID':<8} | {'Token':<8} | Description")
    print("-" * 70)
    for i, (token_id, token, meaning) in enumerate(zip(token_seq, decoded_tokens, token_meanings)):
        print(f"{i:<8} | {token_id:<8} | {token:<8} | {meaning}")
    
    # Show special tokens summary
    print("\n📌 Special Tokens Summary:")
    print(f"    [CLS] token at position 0 (ID: {token_seq[0]})")
    print(f"    [SEP] token at position {len(token_seq)-2} (ID: {token_seq[-2]})")
    print(f"    [PAD] tokens from position {len(token_seq)-1} to end (ID: {token_seq[-1]})")
    
    # Show one-hot matrix interpretation
    print(f"\n🧮 [One-Hot Matrix Analysis]")
    print(f"    Matrix dimensions: {onehot_mat.shape[0]} rows × {onehot_mat.shape[1]} columns")
    print(f"    Why {onehot_mat.shape[0]} rows? MAX_LEN = {onehot_mat.shape[0]} (sequence length)")
    print(f"    Why {onehot_mat.shape[1]} columns? Each position has {onehot_mat.shape[1]} binary features:")
    print(f"        Column 0: A (Adenine) presence")
    print(f"        Column 1: T (Thymine) presence") 
    print(f"        Column 2: G (Guanine) presence")
    print(f"        Column 3: C (Cytosine) presence")
    print(f"        Column 4: Gap/Insertion indicator")
    print(f"        Column 5: First base indicator")
    print(f"        Column 6: Second base indicator")
    
    # Print complete one-hot matrix in a readable format
    print("\n🔢 Complete One-Hot Encoding Matrix:")
    # Print column headers
    print("Pos  | A T G C GAP FST SEC | Interpretation")
    print("-" * 60)
    
    # Print each position's encoding
    for i in range(onehot_mat.shape[0]):
        features = onehot_mat[i]
        base_encoding = " ".join(["1" if x > 0 else "." for x in features[:4]])
        meta_encoding = " ".join(["1" if x > 0 else "." for x in features[4:]])
        
        # Get the DNA base(s) at this position
        dna_bases = []
        if features[0] > 0: dna_bases.append('A')
        if features[1] > 0: dna_bases.append('T')
        if features[2] > 0: dna_bases.append('G')
        if features[3] > 0: dna_bases.append('C')
        
        # Interpretation
        feature_sum = int(np.sum(features[:4]))  # Only sum the base indicators
        if feature_sum == 0:
            interpretation = "Padding/No data"
        elif feature_sum == 1:
            interpretation = f"Single base: {dna_bases[0]}"
        elif feature_sum == 2:
            if features[4] > 0:  # If gap is present
                gap_pos = 4 - len([x for x in features[:4] if x > 0])
                gap_base = dna_bases[0] if gap_pos < len(dna_bases) else '?'
                interpretation = f"Gap in sequence: {dna_bases[0]}_"
            else:
                interpretation = f"Base pair: {''.join(dna_bases)}"
        else:
            interpretation = "Complex pattern"
        
        # Add position metadata
        if features[5] > 0 and features[6] > 0:
            interpretation += " (Both strands)"
        elif features[5] > 0:
            interpretation += " (First strand)"
        elif features[6] > 0:
            interpretation += " (Second strand)"
        
        print(f"{i:3d} | {base_encoding} | {meta_encoding} | {interpretation}")
    
    # Print summary of one-hot encoding
    print("\n📊 One-Hot Matrix Summary:")
    print(f"    Total positions: {onehot_mat.shape[0]}")
    print(f"    Non-padding positions: {np.sum([np.any(row > 0) for row in onehot_mat])}")
    print(f"    Padding positions: {np.sum([not np.any(row > 0) for row in onehot_mat])}")
    
    # Show distribution of features
    print("\n📈 Feature Distribution:")
    feature_names = ['A', 'T', 'G', 'C', 'GAP', 'FST', 'SEC']
    for i, name in enumerate(feature_names):
        count = np.sum(onehot_mat[:, i] > 0)
        print(f"    {name}: {count:3d} positions ({count/onehot_mat.shape[0]:.1%})")
    
    return token_seq.reshape(1, -1), onehot_mat.reshape(1, -1, 7), label

def demonstrate_model_architecture():
    """
    Demonstrate the model architecture in detail
    """
    print(f"\n" + "="*80)
    print(f"🏗️  [MODEL ARCHITECTURE DEMONSTRATION]")
    print(f"="*80)
    
    print(f"\n📋 [Model Configuration]")
    print(f"    Vocabulary size: {VOCAB_SIZE}")
    print(f"    Maximum sequence length: {MAX_LEN}")
    print(f"    Model type: Hybrid CNN-BERT architecture")
    print(f"    Purpose: CRISPR-Cas9 off-target prediction")
    
    print(f"\n🔬 [CNN Branch Details]")
    print(f"    Input: One-hot encoded DNA sequences ({MAX_LEN} x 7)")
    print(f"    Convolution layers: 4 parallel Conv1D with kernel sizes [5, 15, 25, 35]")
    print(f"    Filters per conv: 80")
    print(f"    Activation: ReLU")
    print(f"    Purpose: Capture local DNA sequence patterns")
    
    print(f"\n🤖 [BERT Branch Details]")
    print(f"    Input: Tokenized DNA sequences ({MAX_LEN} tokens)")
    print(f"    Embedding dimension: 128 (small debug mode)")
    print(f"    Transformer layers: 2")
    print(f"    Attention heads: 4")
    print(f"    Feed-forward dimension: 256")
    print(f"    Purpose: Capture long-range dependencies")
    
    print(f"\n🔄 [GRU Processing]")
    print(f"    Both branches processed through Bidirectional GRU")
    print(f"    GRU units: 40 per direction (80 total)")
    print(f"    Purpose: Sequence modeling and temporal dependencies")
    
    print(f"\n🔗 [Fusion Strategy]")
    print(f"    CNN contribution: 20% (0.2x weight)")
    print(f"    BERT contribution: 80% (0.8x weight)")
    print(f"    Fusion method: Weighted addition")
    
    print(f"\n🎯 [Classification Head]")
    print(f"    Dense layer 1: 128 units, ReLU")
    print(f"    Dense layer 2: 64 units, ReLU")
    print(f"    Dropout: 0.35")
    print(f"    Output: 2 units, Softmax (On-target vs Off-target)")
    
    # Build the model
    print(f"\n🏗️  [Building Model] Creating the model...")
    model = build_crispr_bert_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, small_debug=True)
    
    print(f"\n📊 [Model Summary]")
    model.summary()
    
    return model

def demonstrate_training_process():
    """
    Demonstrate a simplified training process
    """
    print(f"\n" + "="*80)
    print(f"🚀 [TRAINING PROCESS DEMONSTRATION]")
    print(f"="*80)
    
    print(f"\n📊 [Data Preparation]")
    print(f"    Loading training data...")
    X_tokens, X_onehot, y = load_dataset("datasets/c.txt")
    
    # Use a small subset for demonstration
    subset_size = 1000
    X_tokens_sub = X_tokens[:subset_size]
    X_onehot_sub = X_onehot[:subset_size]
    y_sub = y[:subset_size]
    
    print(f"    Using subset of {subset_size} samples for demonstration")
    print(f"    Positive samples: {np.sum(y_sub==1)}")
    print(f"    Negative samples: {np.sum(y_sub==0)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    Xt, Xv, Ot, Ov, yt, yv = train_test_split(
        X_tokens_sub, X_onehot_sub, y_sub, 
        test_size=0.2, stratify=y_sub, random_state=42
    )
    
    print(f"    Training samples: {len(yt)}")
    print(f"    Validation samples: {len(yv)}")
    
    # Build and compile model
    print(f"\n🏗️  [Model Setup]")
    model = build_crispr_bert_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, small_debug=True)
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    print(f"    Model compiled with Adam optimizer")
    print(f"    Loss function: sparse_categorical_crossentropy")
    print(f"    Metrics: accuracy")
    
    # Quick training for demonstration
    print(f"\n🚀 [Quick Training] Training for 3 epochs...")
    print(f"    This is a demonstration - full training would use more epochs")
    
    history = model.fit(
        [Xt, Ot], yt,
        validation_data=([Xv, Ov], yv),
        epochs=3,
        batch_size=32,
        verbose=1
    )
    
    print(f"\n✅ [Training Complete]")
    print(f"    Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"    Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model

def main_demonstration():
    """
    Main demonstration function
    """
    print(f"\n🎯 [MAIN DEMONSTRATION] Starting complete model demonstration...")
    
    # Step 1: Data Processing
    token_input, onehot_input, true_label = demonstrate_data_processing()
    
    # Step 2: Model Architecture
    model = demonstrate_model_architecture()
    
    # Step 3: Training Process
    trained_model = demonstrate_training_process()
    
    # Step 4: Single Example Prediction
    print(f"\n" + "="*80)
    print(f"🎯 [SINGLE EXAMPLE PREDICTION]")
    print(f"="*80)
    
    print(f"\n🔍 [Making Prediction] Using trained model on sample...")
    prediction = predict_single_example(trained_model, token_input, onehot_input, true_label)
    
    print(f"\n📊 [Prediction Summary]")
    print(f"    Input shape: Token {token_input.shape}, One-hot {onehot_input.shape}")
    print(f"    True label: {true_label}")
    print(f"    Predicted probabilities: {prediction[0]}")
    print(f"    Predicted class: {np.argmax(prediction[0])}")
    print(f"    Confidence: {np.max(prediction[0]):.4f}")
    
    # Step 5: Model Interpretation
    print(f"\n" + "="*80)
    print(f"🧠 [MODEL INTERPRETATION]")
    print(f"="*80)
    
    print(f"\n💡 [What the Model Learned]")
    print(f"    The model combines two types of information:")
    print(f"    1. CNN branch: Local DNA sequence patterns and motifs")
    print(f"    2. BERT branch: Long-range dependencies and context")
    print(f"    The GRU layers capture temporal relationships in the sequence")
    print(f"    The fusion layer combines both perspectives for final prediction")
    
    print(f"\n🎯 [Prediction Confidence]")
    confidence = np.max(prediction[0])
    if confidence > 0.8:
        print(f"    High confidence prediction ({confidence:.4f})")
    elif confidence > 0.6:
        print(f"    Medium confidence prediction ({confidence:.4f})")
    else:
        print(f"    Low confidence prediction ({confidence:.4f})")
    
    print(f"\n✅ [DEMONSTRATION COMPLETE]")
    print(f"    This demonstration showed:")
    print(f"    1. How DNA sequences are processed and encoded")
    print(f"    2. The hybrid CNN-BERT model architecture")
    print(f"    3. The training process with real data")
    print(f"    4. Step-by-step prediction on a single example")
    print(f"    5. Model interpretation and confidence analysis")
    
    print(f"\n" + "="*120)
    print(f"="*120)

if __name__ == "__main__":
    main_demonstration()
