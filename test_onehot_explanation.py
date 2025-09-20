# test_onehot_explanation.py - Simple test showing one-hot matrix explanation
import numpy as np
from data_process import load_dataset, MAX_LEN, VOCAB_SIZE

print("=" * 80)
print("ONE-HOT MATRIX EXPLANATION - SIMPLE TEST")
print("=" * 80)

def explain_onehot_matrix():
    """Explain the one-hot matrix structure"""
    print(f"\n🧮 [ONE-HOT MATRIX EXPLANATION]")
    print(f"="*50)
    
    print(f"\n📐 [Matrix Dimensions: 26 × 7]")
    print(f"    Why 26 rows? MAX_LEN = 26 (sequence length)")
    print(f"    Why 7 columns? Each position has 7 binary features:")
    print(f"        Column 0: A (Adenine) presence")
    print(f"        Column 1: T (Thymine) presence") 
    print(f"        Column 2: G (Guanine) presence")
    print(f"        Column 3: C (Cytosine) presence")
    print(f"        Column 4: Gap/Insertion indicator")
    print(f"        Column 5: First base indicator")
    print(f"        Column 6: Second base indicator")
    
    print(f"\n🔢 [Sum Values Explanation]")
    print(f"    Sum = 0: Padding/no data ('--')")
    print(f"    Sum = 1: Single base pair (AA, TT, GG, CC)")
    print(f"    Sum = 2: Mismatched pair (AT, AG, AC, etc.)")
    print(f"    Sum = 3: Gap-containing pair (A_, T_, G_, C_, _A, _T, _G, _C)")
    print(f"    Sum = 4: Invalid/error state (should not occur)")

def analyze_real_data():
    """Analyze real one-hot matrix data"""
    print(f"\n🔍 [REAL DATA ANALYSIS]")
    print(f"="*50)
    
    # Load data
    print(f"\n📁 [Loading Data]")
    X_tokens, X_onehot, y = load_dataset("datasets/c.txt")
    print(f"    Dataset loaded: {len(y)} samples")
    print(f"    Token sequences shape: {X_tokens.shape}")
    print(f"    One-hot matrices shape: {X_onehot.shape}")
    print(f"    Labels shape: {y.shape}")
    
    # Analyze first sample
    print(f"\n🔬 [First Sample Analysis]")
    sample_idx = 0
    token_seq = X_tokens[sample_idx]
    onehot_mat = X_onehot[sample_idx]
    label = y[sample_idx]
    
    print(f"    Sample {sample_idx}:")
    print(f"    Token sequence: {token_seq}")
    print(f"    One-hot matrix shape: {onehot_mat.shape}")
    print(f"    True label: {label} ({'Off-target' if label == 1 else 'On-target'})")
    
    print(f"\n🧮 [One-Hot Matrix Analysis]")
    print(f"    Matrix dimensions: {onehot_mat.shape[0]} × {onehot_mat.shape[1]}")
    print(f"    Detailed analysis of first 10 positions:")
    
    for pos in range(min(10, onehot_mat.shape[0])):
        features = onehot_mat[pos]
        feature_sum = np.sum(features)
        print(f"        Position {pos:2d}: {features} (sum={feature_sum})")
        
        # Interpret the features
        if pos == 0:
            print(f"                    -> CLS token (start)")
        elif pos == 25:
            print(f"                    -> SEP token (end)")
        elif feature_sum == 0:
            print(f"                    -> PAD token (padding)")
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

def show_cnn_processing():
    """Show how CNN processes the one-hot matrix"""
    print(f"\n🔬 [CNN PROCESSING EXPLANATION]")
    print(f"="*50)
    
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

def main():
    """Main function"""
    print(f"\n🎯 [Starting One-Hot Matrix Explanation]")
    
    # Step 1: Explain the structure
    explain_onehot_matrix()
    
    # Step 2: Analyze real data
    analyze_real_data()
    
    # Step 3: Show CNN processing
    show_cnn_processing()
    
    print(f"\n✅ [Explanation Complete]")
    print(f"    This shows exactly why the matrix is 26×7")
    print(f"    and what the sum values mean!")
    print(f"    Perfect for professor presentation!")

if __name__ == "__main__":
    main()
