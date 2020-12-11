# Portions of this code were adapted from 
# https://github.com/jingraham/neurips19-graph-protein-design

import tensorflow as tf
import numpy as np
import tqdm
from datetime import datetime
import json, pdb
import pandas as pd
from collections import defaultdict
jsonl_file = '../data/chain_set.jsonl'
split_file = '../data/chain_set_splits.json'

# These abbreviations and one-hot indices are not intrinsic to the model
# or the CATH 4.2 dataset, so feel free to use your own if training
# a new model or task. However, the pretrained model uses these indices.
abbrev = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

# Generic method for loading a structure dataset
# Batch size: number of amino acids
def load_dataset(path, batch_size, shuffle=True):
    data = json.loads(open(path).read())
    data = DynamicLoader(data, batch_size, shuffle=shuffle)
    output_types = (tf.float32, tf.int32, tf.float32)
    return tf.data.Dataset.from_generator(data.__iter__, output_types=output_types).prefetch(3)

def ts50_dataset(batch_size):
    return load_dataset('../data/ts50.json', batch_size)

# Loads the CATH 4.2 dataset while splitting into
# train, validation, and test sets. You can safely ignore this
# function for any other purpose.
def cath_dataset(batch_size, jsonl_file=jsonl_file, filter_file=None):
    print('Loading from', jsonl_file)
    with open(split_file) as f:
        dataset_splits = json.load(f)

    if filter_file:
      print('Filtering from', filter_file)
      filter_file = json.load(open(filter_file))["test"]
      
    train_list, val_list, test_list = dataset_splits['train'], dataset_splits['validation'], dataset_splits['test']
    trainset, valset, testset = [], [], []
    with open(jsonl_file) as f:
        lines = f.readlines()
    for i, line in tqdm.tqdm(enumerate(lines)):
        entry = json.loads(line)
        seq = entry['seq']
        name = entry['name']
        if filter_file and name not in filter_file: 
            continue
        for key, val in entry['coords'].items():
            entry['coords'][key] = np.asarray(val)
        if name in train_list: trainset.append(entry)
        elif name in val_list: valset.append(entry)
        elif name in test_list: testset.append(entry)
            
    trainset = DynamicLoader(trainset, batch_size)
    valset = DynamicLoader(valset, batch_size)
    testset = DynamicLoader(testset, batch_size)
    
    output_types = (tf.float32, tf.int32, tf.float32)
    trainset = tf.data.Dataset.from_generator(trainset.__iter__, output_types=output_types).prefetch(3)
    valset = tf.data.Dataset.from_generator(valset.__iter__, output_types=output_types).prefetch(3)
    testset = tf.data.Dataset.from_generator(testset.__iter__, output_types=output_types).prefetch(3)
    
    return trainset, valset, testset

# Given a single batch, featurizes from the JSON key-value representation
# into a tuple of tensors supplied by the data loader. 
# You will likely need to modify this function for other applications.
def parse_batch(batch):
    B = len(batch)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)
    
    for i, b in enumerate(batch):
        l = len(b['seq'])
        x = b['coords']

        # coords in the CATH 4.2 json are a dict, not a nested list
        if type(x) == dict: 
            x = np.stack([x[c] for c in ['N', 'CA', 'C', 'O']], 1)        

        # Pad to the maximum length in the batch
        X[i] = np.pad(x, [[0,L_max-l], [0,0], [0,0]],
                        'constant', constant_values=(np.nan, ))
        if type(b['seq']) == str:
            S[i, :l] = np.asarray([lookup[a] for a in b['seq']], dtype=np.int32)
        else:
            S[i, :l] = b['seq']
            
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)
    
    return X, S, mask
    
class DynamicLoader(): 
    def __init__(self, dataset, batch_size=3000, shuffle=True): 
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def batch(self):
        dataset = self.dataset
        lengths = [len(b['seq']) for b in dataset]

        clusters, batch = [], []
        for ix in np.argsort(lengths):
            size = lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
            else:
                if len(batch) > 0: clusters.append(batch)
                batch = [ix]
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters
        print(len(clusters), 'batches', len(dataset), 'structures')

    def __iter__(self):
        self.batch()
        if self.shuffle: np.random.shuffle(self.clusters)
        N = len(self.clusters)
        for b_idx in self.clusters[:N]:
            batch = [self.dataset[i] for i in b_idx]
            yield parse_batch(batch)
