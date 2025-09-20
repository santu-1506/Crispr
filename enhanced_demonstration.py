# enhanced_demonstration.py - Enhanced demonstration with detailed one-hot matrix explanation
import numpy as np
import tensorflow as tf
from data_process import load_dataset, MAX_LEN, VOCAB_SIZE, DatasetEncoder, make_pair_list
from detailed_model import build_crispr_bert_model, predict_single_example

print("=" * 120)
print("CRISPR-Cas9 Off-Target Prediction Model - ENHANCED DEMONSTRATION")
print("=" * 120)

def explain_onehot_matrix_detailed():
    """
    Detailed explanation of the one-hot matrix structure and meaning
    """
    print(f"\n" + "="*80)
    print(f"🧮 [ONE-HOT MATRIX DETAILED EXPLANATION]")
    print(f"="*80)
    
    print(f"\n📐 [Matrix Dimensions: 26 x 7]")
    print(f"    Why 26 rows?")
    print(f"    - MAX_LEN = 26 (maximum sequence length)")
    print(f"    - Each row represents one position in the DNA sequence")
    print(f"    - Position 0: CLS token (special start token)")
    print(f"    - Positions 1-24: Actual DNA sequence positions")
    print(f"    - Position 25: SEP token (special end token)")
    print(f"    - If sequence < 24, remaining positions are PAD tokens")
    
    print(f"\n🔢 [Why 7 columns?]")
    print(f"    Each position has 7 binary features representing DNA structural properties:")
    print(f"    Column 0: A (Adenine) presence")
    print(f"    Column 1: T (Thymine) presence") 
    print(f"    Column 2: G (Guanine) presence")
    print(f"    Column 3: C (Cytosine) presence")
    print(f"    Column 4: Gap/Insertion indicator")
    print(f"    Column 5: First base indicator")
    print(f"    Column 6: Second base indicator")
    
    print(f"\n🧬 [DNA Base Pair Encoding Examples]")
    print(f"    'AA' -> [1,0,0,0,0,0,0] (A=1, T=0, G=0, C=0, no gap, first=A, second=A)")
    print(f"    'AT' -> [1,1,0,0,0,1,0] (A=1, T=1, G=0, C=0, no gap, first=A, second=T)")
    print(f"    'G_' -> [0,0,1,0,1,1,0] (A=0, T=0, G=1, C=0, gap=1, first=G, second=gap)")
    print(f"    '_C' -> [0,0,0,1,1,0,1] (A=0, T=0, G=0, C=1, gap=1, first=gap, second=C)")
    print(f"    '--' -> [0,0,0,0,0,0,0] (padding/no data)")
    
    print(f"\n🔍 [Sum Values Explanation]")
    print(f"    The sum of each row indicates the type of DNA pair:")
    print(f"    Sum = 0: Padding/no data ('--')")
    print(f"    Sum = 1: Single base pair (AA, TT, GG, CC)")
    print(f"    Sum = 2: Mismatched pair (AT, AG, AC, TA, TG, TC, GA, GT, GC, CA, CT, CG)")
    print(f"    Sum = 3: Gap-containing pair (A_, T_, G_, C_, _A, _T, _G, _C)")
    print(f"    Sum = 4: Invalid/error state (should not occur)")
    
    print(f"\n💡 [Why This Encoding?]")
    print(f"    This encoding captures:")
    print(f"    1. Base composition (which nucleotides are present)")
    print(f"    2. Base pairing (matches vs mismatches)")
    print(f"    3. Structural gaps (insertions/deletions)")
    print(f"    4. Position information (which base comes first)")
    print(f"    This rich representation helps the CNN learn DNA structural patterns!")

def demonstrate_onehot_analysis():
    """
    Demonstrate one-hot matrix analysis on real data
    """
    print(f"\n" + "="*80)
    print(f"🔬 [ONE-HOT MATRIX ANALYSIS ON REAL DATA]")
    print(f"="*80)
    
    # Load data
    print(f"\n📁 [Loading Real Data]")
    X_tokens, X_onehot, y = load_dataset("datasets/c.txt")
    print(f"    Dataset loaded: {len(y)} samples")
    
    # Analyze first few samples
    print(f"\n🔍 [Analyzing First 3 Samples]")
    for i in range(min(3, len(y))):
        print(f"\n    Sample {i}:")
        print(f"    Token sequence: {X_tokens[i]}")
        print(f"    One-hot matrix shape: {X_onehot[i].shape}")
        print(f"    True label: {y[i]} ({'Off-target' if y[i] == 1 else 'On-target'})")
        
        # Analyze one-hot matrix
        print(f"    One-hot matrix analysis:")
        for pos in range(min(10, X_onehot[i].shape[0])):  # Show first 10 positions
            features = X_onehot[i][pos]
            feature_sum = np.sum(features)
            print(f"        Position {pos:2d}: {features} (sum={feature_sum})")
            
            # Interpret the features
            if feature_sum == 0:
                print(f"                    -> Padding/no data")
            elif feature_sum == 1:
                if features[0] == 1: print(f"                    -> AA (A-A pair)")
                elif features[1] == 1: print(f"                    -> TT (T-T pair)")
                elif features[2] == 1: print(f"                    -> GG (G-G pair)")
                elif features[3] == 1: print(f"                    -> CC (C-C pair)")
            elif feature_sum == 2:
                if features[0] == 1 and features[1] == 1: print(f"                    -> AT (A-T pair)")
                elif features[0] == 1 and features[2] == 1: print(f"                    -> AG (A-G pair)")
                elif features[0] == 1 and features[3] == 1: print(f"                    -> AC (A-C pair)")
                elif features[1] == 1 and features[2] == 1: print(f"                    -> TG (T-G pair)")
                elif features[1] == 1 and features[3] == 1: print(f"                    -> TC (T-C pair)")
                elif features[2] == 1 and features[3] == 1: print(f"                    -> GC (G-C pair)")
            elif feature_sum == 3:
                if features[4] == 1: print(f"                    -> Gap-containing pair")
                if features[0] == 1 and features[4] == 1: print(f"                    -> A_ (A with gap)")
                elif features[1] == 1 and features[4] == 1: print(f"                    -> T_ (T with gap)")
                elif features[2] == 1 and features[4] == 1: print(f"                    -> G_ (G with gap)")
                elif features[3] == 1 and features[4] == 1: print(f"                    -> C_ (C with gap)")
        
        print(f"    ... (showing first 10 positions)")

def demonstrate_cnn_processing():
    """
    Show how CNN processes the one-hot matrix
    """
    print(f"\n" + "="*80)
    print(f"🔬 [CNN PROCESSING OF ONE-HOT MATRIX]")
    print(f"="*80)
    
    print(f"\n📊 [CNN Input Analysis]")
    print(f"    CNN receives: (batch_size, 26, 7) one-hot matrices")
    print(f"    Each sample: 26 positions × 7 features = 182 total features")
    print(f"    CNN processes this as a 1D sequence with 7 channels")
    
    print(f"\n🔄 [CNN Convolution Process]")
    print(f"    Conv1D layers with different kernel sizes:")
    print(f"    - Kernel size 5: Captures local patterns (5 consecutive positions)")
    print(f"    - Kernel size 15: Captures medium-range patterns (15 positions)")
    print(f"    - Kernel size 25: Captures long-range patterns (25 positions)")
    print(f"    - Kernel size 35: Captures very long patterns (35 positions)")
    print(f"    Each kernel slides across the 26 positions, learning DNA motifs!")
    
    print(f"\n🎯 [Why Different Kernel Sizes?]")
    print(f"    DNA has patterns at different scales:")
    print(f"    - Short motifs: 3-5 base pairs (promoter elements)")
    print(f"    - Medium motifs: 10-15 base pairs (protein binding sites)")
    print(f"    - Long motifs: 20+ base pairs (regulatory regions)")
    print(f"    The CNN learns to recognize these hierarchical patterns!")

def main_enhanced_demonstration():
    """
    Main enhanced demonstration
    """
    print(f"\n🎯 [ENHANCED DEMONSTRATION] Starting comprehensive analysis...")
    
    # Step 1: Explain one-hot matrix in detail
    explain_onehot_matrix_detailed()
    
    # Step 2: Analyze real data
    demonstrate_onehot_analysis()
    
    # Step 3: Show CNN processing
    demonstrate_cnn_processing()
    
    # Step 4: Build and test model
    print(f"\n" + "="*80)
    print(f"🏗️  [MODEL BUILDING AND TESTING]")
    print(f"="*80)
    
    print(f"\n🏗️  [Building Model]")
    model = build_crispr_bert_model(VOCAB_SIZE, MAX_LEN, small_debug=True)
    
    print(f"\n🔍 [Testing Single Prediction]")
    X_tokens, X_onehot, y = load_dataset("datasets/c.txt")
    
    # Take first sample
    token_sample = X_tokens[0:1]
    onehot_sample = X_onehot[0:1]
    label_sample = y[0]
    
    print(f"\n📊 [Sample Details]")
    print(f"    Token shape: {token_sample.shape}")
    print(f"    One-hot shape: {onehot_sample.shape}")
    print(f"    True label: {label_sample}")
    
    # Show detailed one-hot analysis
    print(f"\n🧮 [Detailed One-Hot Analysis]")
    features = onehot_sample[0]
    print(f"    Matrix shape: {features.shape}")
    print(f"    All positions analysis:")
    for pos in range(features.shape[0]):
        pos_features = features[pos]
        pos_sum = np.sum(pos_features)
        print(f"        Position {pos:2d}: {pos_features} (sum={pos_sum})")
        
        # Detailed interpretation
        if pos == 0:
            print(f"                    -> CLS token (start)")
        elif pos == 25:
            print(f"                    -> SEP token (end)")
        elif pos_sum == 0:
            print(f"                    -> PAD token (padding)")
        elif pos_sum == 1:
            if pos_features[0] == 1: print(f"                    -> AA pair")
            elif pos_features[1] == 1: print(f"                    -> TT pair")
            elif pos_features[2] == 1: print(f"                    -> GG pair")
            elif pos_features[3] == 1: print(f"                    -> CC pair")
        elif pos_sum == 2:
            print(f"                    -> Mismatched pair")
        elif pos_sum == 3:
            print(f"                    -> Gap-containing pair")
    
    # Make prediction
    print(f"\n🎯 [Making Prediction]")
    prediction = model.predict([token_sample, onehot_sample], verbose=0)
    print(f"    Prediction probabilities: {prediction[0]}")
    print(f"    Predicted class: {np.argmax(prediction[0])}")
    print(f"    Confidence: {np.max(prediction[0]):.4f}")
    
    print(f"\n✅ [ENHANCED DEMONSTRATION COMPLETE]")
    print(f"    This demonstration showed:")
    print(f"    1. Why the matrix is 26×7 (sequence length × features)")
    print(f"    2. What each of the 7 features represents")
    print(f"    3. How the sum values indicate pair types")
    print(f"    4. How CNN processes this rich representation")
    print(f"    5. Real data analysis with detailed interpretation")
    
    print(f"\n" + "="*120)
    print(f"🎉 ENHANCED DEMONSTRATION COMPLETE - PERFECT FOR PROFESSOR! 🎉")
    print(f"="*120)

if __name__ == "__main__":
    main_enhanced_demonstration()
