# One-Hot Matrix Explanation - CRISPR Model

## 🧮 **Matrix Dimensions: 26 × 7**

### **Why 26 rows?**

- **MAX_LEN = 26** (maximum sequence length)
- Each row represents **one position** in the DNA sequence
- **Position 0**: CLS token (special start token)
- **Positions 1-24**: Actual DNA sequence positions
- **Position 25**: SEP token (special end token)
- **If sequence < 24**: Remaining positions are PAD tokens

### **Why 7 columns?**

Each position has **7 binary features** representing DNA structural properties:

| Column | Feature       | Description                         |
| ------ | ------------- | ----------------------------------- |
| 0      | A (Adenine)   | Presence of Adenine base            |
| 1      | T (Thymine)   | Presence of Thymine base            |
| 2      | G (Guanine)   | Presence of Guanine base            |
| 3      | C (Cytosine)  | Presence of Cytosine base           |
| 4      | Gap/Insertion | Indicator for gaps or insertions    |
| 5      | First base    | Which base comes first in the pair  |
| 6      | Second base   | Which base comes second in the pair |

## 🔢 **Sum Values Explanation**

The **sum of each row** indicates the type of DNA pair:

| Sum   | Meaning             | Examples                   |
| ----- | ------------------- | -------------------------- |
| **0** | Padding/no data     | `[0,0,0,0,0,0,0]` → `'--'` |
| **1** | Single base pair    | `[1,0,0,0,0,0,0]` → `'AA'` |
| **2** | Mismatched pair     | `[1,1,0,0,0,1,0]` → `'AT'` |
| **3** | Gap-containing pair | `[0,0,1,0,1,1,0]` → `'G_'` |
| **4** | Invalid/error       | Should not occur           |

## 🧬 **DNA Base Pair Encoding Examples**

### **Perfect Matches (Sum = 1)**

```
'AA' → [1,0,0,0,0,0,0]  (A=1, T=0, G=0, C=0, no gap, first=A, second=A)
'TT' → [0,1,0,0,0,0,0]  (A=0, T=1, G=0, C=0, no gap, first=T, second=T)
'GG' → [0,0,1,0,0,0,0]  (A=0, T=0, G=1, C=0, no gap, first=G, second=G)
'CC' → [0,0,0,1,0,0,0]  (A=0, T=0, G=0, C=1, no gap, first=C, second=C)
```

### **Mismatched Pairs (Sum = 2)**

```
'AT' → [1,1,0,0,0,1,0]  (A=1, T=1, G=0, C=0, no gap, first=A, second=T)
'AG' → [1,0,1,0,0,1,0]  (A=1, T=0, G=1, C=0, no gap, first=A, second=G)
'AC' → [1,0,0,1,0,1,0]  (A=1, T=0, G=0, C=1, no gap, first=A, second=C)
'TG' → [0,1,1,0,0,1,0]  (A=0, T=1, G=1, C=0, no gap, first=T, second=G)
'TC' → [0,1,0,1,0,1,0]  (A=0, T=1, G=0, C=1, no gap, first=T, second=C)
'GC' → [0,0,1,1,0,1,0]  (A=0, T=0, G=1, C=1, no gap, first=G, second=C)
```

### **Gap-Containing Pairs (Sum = 3)**

```
'A_' → [1,0,0,0,1,1,0]  (A=1, T=0, G=0, C=0, gap=1, first=A, second=gap)
'T_' → [0,1,0,0,1,1,0]  (A=0, T=1, G=0, C=0, gap=1, first=T, second=gap)
'G_' → [0,0,1,0,1,1,0]  (A=0, T=0, G=1, C=0, gap=1, first=G, second=gap)
'C_' → [0,0,0,1,1,1,0]  (A=0, T=0, G=0, C=1, gap=1, first=C, second=gap)
'_A' → [1,0,0,0,1,0,1]  (A=1, T=0, G=0, C=0, gap=1, first=gap, second=A)
'_T' → [0,1,0,0,1,0,1]  (A=0, T=1, G=0, C=0, gap=1, first=gap, second=T)
'_G' → [0,0,1,0,1,0,1]  (A=0, T=0, G=1, C=0, gap=1, first=gap, second=G)
'_C' → [0,0,0,1,1,0,1]  (A=0, T=0, G=0, C=1, gap=1, first=gap, second=C)
```

### **Padding (Sum = 0)**

```
'--' → [0,0,0,0,0,0,0]  (No data/padding)
```

## 💡 **Why This Encoding?**

This rich encoding captures:

1. **Base composition** - Which nucleotides are present
2. **Base pairing** - Matches vs mismatches
3. **Structural gaps** - Insertions/deletions
4. **Position information** - Which base comes first
5. **Sequence context** - Relationship between positions

## 🔬 **How CNN Uses This**

The CNN processes this as a **1D sequence with 7 channels**:

- **Input shape**: `(batch_size, 26, 7)`
- **Each sample**: 26 positions × 7 features = **182 total features**
- **Different kernel sizes** capture patterns at different scales:
  - **Kernel 5**: Local patterns (5 consecutive positions)
  - **Kernel 15**: Medium-range patterns (15 positions)
  - **Kernel 25**: Long-range patterns (25 positions)
  - **Kernel 35**: Very long patterns (35 positions)

## 🎯 **Real Example Analysis**

For a sample with one-hot matrix:

```
Position 0: [0,0,0,0,0,0,0] (sum=0) → CLS token
Position 1: [0,0,1,1,0,1,0] (sum=3) → GC pair with gap
Position 2: [1,0,0,0,1,0,1] (sum=3) → A_ gap-containing pair
Position 3: [1,0,0,0,0,0,0] (sum=1) → AA perfect match
Position 4: [0,0,1,0,0,0,0] (sum=1) → GG perfect match
...
Position 25: [0,0,0,0,0,0,0] (sum=0) → SEP token
```

This encoding allows the CNN to learn **hierarchical DNA patterns** at multiple scales, making it perfect for CRISPR off-target prediction!

## 🎉 **Perfect for Professor Presentation!**

This explanation shows:

- ✅ **Why** the matrix is 26×7
- ✅ **What** each feature represents
- ✅ **How** the sum values indicate pair types
- ✅ **Why** this encoding is powerful for DNA analysis
- ✅ **How** the CNN processes this rich representation
