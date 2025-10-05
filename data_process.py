# data_process.py (patched for your .txt datasets)
import numpy as np
import pandas as pd
import pickle

CLS_IDX = 0
SEP_IDX = 1
PAD_IDX = 0
MAX_LEN = 26

# Use the exact token mapping provided by the user (no dynamic construction)
token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,
    'AA': 2, 'AC': 3, 'AG': 4, 'AT': 5,
    'CA': 6, 'CC': 7, 'CG': 8, 'CT': 9,
    'GA': 10, 'GC': 11, 'GG': 12, 'GT': 13,
    'TA': 14, 'TC': 15, 'TG': 16, 'TT': 17,
    'A_': 18, '_A': 19, 'C_': 20, '_C': 21, 'G_': 22,
    '_G': 23, 'T_': 24, '_T': 25, '--': 26
}

# Expose the same names expected by the rest of the pipeline
token_to_index = token_dict
index_to_token = {v: k for k, v in token_to_index.items()}
VOCAB_SIZE = max(token_to_index.values()) + 1

class DatasetEncoder:
    def __init__(self, max_len=MAX_LEN):
        self.max_len = max_len
        self.encoded_dict_indel = {
            'AA':[1,0,0,0,0,0,0],'AT':[1,1,0,0,0,1,0],'AG':[1,0,1,0,0,1,0],'AC':[1,0,0,1,0,1,0],
            'TA':[1,1,0,0,0,0,1],'TT':[0,1,0,0,0,0,0],'TG':[0,1,1,0,0,1,0],'TC':[0,1,0,1,0,1,0],
            'GA':[1,0,1,0,0,0,1],'GT':[0,1,1,0,0,0,1],'GG':[0,0,1,0,0,0,0],'GC':[0,0,1,1,0,1,0],
            'CA':[1,0,0,1,0,0,1],'CT':[0,1,0,1,0,0,1],'CG':[0,0,1,1,0,0,1],'CC':[0,0,0,1,0,0,0],
            '_A':[1,0,0,0,1,0,1],'_T':[0,1,0,0,1,0,1],'_G':[0,0,1,0,1,0,1],'_C':[0,0,0,1,1,0,1],
            '--':[0,0,0,0,0,0,0]
        }

    def encode_token_list(self, pair_list):
        """Encode using exact token_dict mapping: [CLS], tokens, [SEP], pad with '--'"""
        toks = [token_to_index['[CLS]']]
        for p in pair_list[:self.max_len-2]:
            toks.append(token_to_index.get(p, token_to_index['--']))
        toks.append(token_to_index['[SEP]'])
        while len(toks) < self.max_len:
            toks.append(token_to_index['--'])
        return np.array(toks, dtype=np.int32)

    def encode_onehot_matrix(self, pair_list):
        mat = [self.encoded_dict_indel['--']]
        for p in pair_list[:self.max_len-2]:
            mat.append(self.encoded_dict_indel.get(p, self.encoded_dict_indel['--']))
        mat.append(self.encoded_dict_indel['--'])
        while len(mat) < self.max_len:
            mat.append(self.encoded_dict_indel['--'])
        return np.array(mat, dtype=np.float32)

def make_pair_list(seq1, seq2):
    """Align sgRNA and target into pair tokens"""
    pair_list = []
    for a, b in zip(seq1, seq2):
        if a == "-" or a == "_":
            token = "_" + b
        elif b == "-" or b == "_":
            token = a + "_"
        else:
            token = a + b
        pair_list.append(token)
    return pair_list
def load_txt_dataset(path):
    toks, mats, labels = [], [], []
    enc = DatasetEncoder()
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3: 
                continue
            seq1, seq2, label = parts[0], parts[1], parts[2]
            # Convert label to float, handling the case where there might be additional columns
            try:
                label = float(label)
            except ValueError:
                print(f"Warning: Could not parse label '{label}' in line: {line.strip()}")
                continue
            pair_list = make_pair_list(seq1, seq2)
            toks.append(enc.encode_token_list(pair_list))
            mats.append(enc.encode_onehot_matrix(pair_list))
            labels.append(int(round(label)))
    return np.array(toks), np.array(mats), np.array(labels)

def load_csv_dataset(path, pair_col='pair_list', label_col='label'):
    df = pd.read_csv(path)
    enc = DatasetEncoder()
    toks, mats, labels = [], [], []
    for _, row in df.iterrows():
        pair_list = [s.strip() for s in str(row[pair_col]).split(',') if s.strip()]
        toks.append(enc.encode_token_list(pair_list))
        mats.append(enc.encode_onehot_matrix(pair_list))
        labels.append(int(row[label_col]))
    return np.array(toks), np.array(mats), np.array(labels)

def load_pkl_dataset(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    toks, mats, labels = [], [], []
    enc = DatasetEncoder()
    if isinstance(data, dict) and "seq1" in data and "seq2" in data:
        for s1, s2, lab in zip(data["seq1"], data["seq2"], data["label"]):
            pair_list = make_pair_list(s1, s2)
            toks.append(enc.encode_token_list(pair_list))
            mats.append(enc.encode_onehot_matrix(pair_list))
            labels.append(int(round(lab)))
    else:
        raise ValueError(f"Unsupported pkl structure: {path}")
    return np.array(toks), np.array(mats), np.array(labels)

def load_dataset(path):
    if path.endswith('.csv'):
        return load_csv_dataset(path)
    elif path.endswith('.txt'):
        return load_txt_dataset(path)
    elif path.endswith('.pkl'):
        return load_pkl_dataset(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
