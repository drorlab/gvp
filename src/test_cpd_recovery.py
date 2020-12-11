import sys

from datetime import datetime
from datasets import *
import random, pdb, sys
import numpy as np
import tqdm, util
from models import *
model = CPDModel(node_features=(8, 100), edge_features=(1,32), hidden_dim=(16,100))
optimizer = tf.keras.optimizers.Adam()

util.load_checkpoint(model, optimizer, sys.argv[1])  


if sys.argv[2] == "cath":
    _, _, testset = cath_dataset(1)
elif sys.argv[2] == "short":
    _, _, testset = cath_dataset(1, filter_file='../data/test_split_L100.json')
elif sys.argv[2] == "sc":
    _, _, testset = cath_dataset(1, filter_file='../data/test_split_sc.json')
elif sys.argv[2] == "ts50":
    testset = ts50_dataset(1)

def recovery(designs, orig):
    N, L = designs.shape
    arr = (designs == orig)
    arr = arr.sum(1) / L
    return np.mean(arr)
    
def sample(model, structure, mask, n, T=0.1): # [1, N, 4, 3]
    structure = tf.repeat(structure, n, axis=0)
    mask = tf.repeat(mask, n, axis=0)
    return model.sample(structure, mask, temperature=0.1)

num = 0
for structure, seq, mask in tqdm.tqdm(testset):
    num += 1
    length = seq.shape[1]
    N = 100
    
    n = min(int(20000 / length), N)
    design = np.zeros((N, length))
    losses = np.zeros(N)
    idx = 0
    while (idx < N):
        my_n = min(n, N-idx)
        pred = sample(model, structure, mask, my_n)
        design[idx:idx+my_n] = tf.cast(pred, tf.int32).numpy()
        idx += min(n, N-idx)
        
    seq = seq.numpy()
    res = recovery(design, seq)
    print(res)
    with open(sys.argv[3], 'a+') as f:
        f.write(str(res) + '\n')