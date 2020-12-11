# Geometric Vector Perceptron

Code to accompany [Learning from Protein Structure with Geometric Vector Perceptrons](https://arxiv.org/abs/2009.01411) by B Jing, S Eismann, P Suriana, RJL Townshend, and RO Dror.

This repository serves two purposes. If you would like to use the architecture for protein design, we provide the pipeline for our experiments as well as our final trained model. If you are interested in adapting the architecture for other purposes, we provide instructions for general use of the GVP.

## Requirements
* UNIX environment
* python==3.7.6
* numpy==1.18.1
* scipy==1.4.1
* pandas==1.0.3
* tensorflow==2.1.0
* tqdm==4.42.1

## Protein design
Our training pipeline uses the CATH 4.2 dataset curated by [Ingraham, et al, NeurIPS 2019](https://github.com/jingraham/neurips19-graph-protein-design). We provide code to train, validate, and test the model on this dataset. We also provide a pretrained model in `models/cath_pretrained`. If you want to test a trained model on new structures, see the section "Using the CPD model" below.

### Fetching the datasets
Run `getCATH.sh` in `data/` to fetch the CATH 4.2 dataset. If you are interested in testing on the TS 50 test set, also run `grep -Fv -f ts50remove.txt chain_set.jsonl > chain_set_ts50.jsonl` to produce a training set without overlap with the TS 50 test set. 

### Training the CPD model
Run  `python3 train_cpd.py [dataset]` in `src/` where `[dataset]` is the complete CATH 4.2 dataset, `../data/chain_set.jsonl`, or the CATH 4.2 with overlap with TS50 removed, `../data/chain_set_ts50.jsonl`. Model checkpoints are saved to `models/` identified by the timestamp of the start of the run and the epoch number.

### Evaluating the CPD model
#### Perplexity
To evaluate perplexity, run `python3 test_cpd_perplexity.py ../models/cath_pretrained` in `src/`.

| Command     | Output      |
| ----------- | ----------- |
| `python3 test_cpd_perplexity.py ../models/cath_pretrained` |  ALL TEST PERPLEXITY 5.29298734664917 <br> SHORT TEST PERPLEXITY 7.0954108238220215 <br> SINGLE CHAIN TEST PERPLEXITY 7.4412713050842285

#### Recovery
To evaluate recovery, run `python3 test_cpd_recovery.py [model] [dataset] [output]` in `src/`. `[dataset]` should be one of `cath`, `short`, `sc`, `ts50`. `[model]` should be `../models/ts50_pretrained` if evaluating on the TS50 test set and `../models/cath_pretrained` otherwise. Recoveries for each target will be dumped into the file `[output]`. To get the median recovery, run `python3 analyze.py [output]`.

Because the recovery can take some time to run, we have supplied outputs in `outputs/`.

| Command     | Output      |
| ----------- | ----------- |
| `python3 analyze.py ../outputs/cath.out`     |  0.40187938705576753
| `python3 analyze.py ../outputs/short.out`    |  0.32149746594868545
| `python3 analyze.py ../outputs/sc.out`       |  0.319731182795699
| `python3 analyze.py ../outputs/ts50.out`     |  0.44852965747702583

### Using the CPD model

To use the CPD model on your own backbone structures, first convert the structures into a `json` format as follows:
```
[
    {
        "seq": "TQDCSFQHSP...",
        "coords": [[[74.46, 58.25, -21.65],...],...]
    }
    ...
]
```
For each structure, `coords` should be a `num_residues x 4 x 3` nested list of the positions of the backbone N, C-alpha, C, and O atoms of each residue (in that order). If only backbone information is available, you can use a placeholder sequence of the same length. Then, run the below instructions (the function `sample` is defined in `test_cpd_recovery.py`)

```
dataset = datasets.load_dataset(PATH_TO_JSON, batch_size=1, shuffle=False)
for structure, seq, mask in dataset:
    n = 1 # number of sequences to sample
    # model is a pretrained CPD model
    design = sample(model, structure, mask, n)
    design = tf.cast(design, tf.int32).numpy()
```
The output `design` now an `n x num_residues` array of `n` designs, with amino acids represented as integers according to the encodings used to train the model. The encodings used by our pretrained model are in `/src/datasets.py`.

## General usage

We describe our implementation in several levels of abstraction to make it as easy as possible to adapt the GVP to your uses. If you have any questions, please contact bjing@stanford.edu.

### Using the core GVP modules

The core GVP modules are implemented in `src/GVP.py`. It contains code for the GVP itself, the vector/scalar dropout, and the vector/scalar batch norm, each of which is a `tf.keras.layers.Module`. These modules are initialized as follows:
```
gvp = GVP(vi, vo, so)
dropout = GVPDropout(drop_rate, nv)
layernorm = GVPLayerNorm(nv)
```
In the code and comments, `vi`, `vo`, `si`, `so` refer to number of vector/scalar channels in/out. `nv` and `ns` are the number of scalar/vector channels, and `nls` and `nlv` are the scalar/vector nonlinearities. The value `si` doesn't need to be specified because TensorFlow imputes it at the first forward pass.

Because the modules are designed to easily replace dense layers in a GNN, they are designed to take a _single_ tensor `x` instead of seperate scalar/vector channel tensors. This is accomplished by assigning the first `3*nv` channels in the input tensor to be the `nv` vector channels and the remaining channels to be the `ns` scalar channels. We provide utility functions `merge` and `split` to convert between seperate tensors where the vector tensor has dims `[..., 3, nv]` and the scalar tensor has dims `[..., ns]`, and a single tensor with dims `[..., 3*nv + ns]`. For example:
```
v, s = input_data
x = merge(v, s)
x = gvp(x)
x = dropout(x, training=True)
x = layernorm(x)
v, s = split(x, nv=v.shape[-1])
```
Use `vs_concat(x1, x2, nv1, nv2)` to concatenate tensors `x1` and `x2` with `nv1` and `nv2` implicit vector channels.

### Using the protein GNN

Our protein GNN is defined in `src/models.py` and is adapted from the protein GNN in [Ingraham, et al, NeurIPS 2019](https://github.com/jingraham/neurips19-graph-protein-design). We provide two fully specified networks which take in raw protein representations and output a single global scalar prediction (`MQAModel`) or a 20-dimensional feature vector at each residue (`CPDModel`). Note that the `CPDModel` currently uses sequence information autoregressively. Sample usage:
```
mqa_model = MQAModel(node_dims, edge_dims, hidden_dims, num_layers)
X, S, mask = input_batch
output = mqa_model(X, S, mask) # dims [batch_size, 1]
```
The input `X` is a `float` tensor with dims `[batch_size, num_residues, 4, 3]` and has the backbone coordinates of N, C-alpha, C, and O atoms of each residue (in that order). `S` contains the sequence information as a integer tensor with dims `[batch_size, num_residues]`. The integer encodings can be arbitrary but the ones used by our pretrained model are defined in `src/datasets.py`. The `mask` is a `float` tensor with dims `[batch, num_nodes]` that is `1` for residues that exist and `0` for nodes that do not.

The three `dims` arguments should each be tuples `(nv, ns)` describing the number of vector and scalar channels to use in each embedding. The protein graph is first built using structural features to produce node embeddings with dims `node_dims`, then transformed into `hidden_dims` after adding sequence information. Therefore the two arguments are somewhat redundant. Note that the edge embeddings are static and are generated with only one vector feature, so anything greater than `edge_nv = 1` is redundant.

If adapting one of the two provided models is insufficient, next we describe the building blocks of the protein GNN.

#### Structural features

The `StructuralFeatures` module converts the tensor `X` of raw backbone coordinates into a proximity graph with structure-based node and edge embeddings described in the paper.
```
feature_builder = StructuralFeatures(node_dims, edge_dims, top_k=30) # k nearest neighbors
h_V, h_E, E_idx = feature_builder(X, mask) # mask is as described above
```
`h_V` is the node embedding tensor with dims `[batch, num_nodes, 3*node_nv+node_ns]`, `h_E` is the edge embedding tensor with dims `[batch, num_nodes, top_k, 3*edge_nv+edge_ns]`, and `E_idx` is the tensor of neighbor node indices with dims `[batch, num_nodes, top_k]`.

#### Message passing layers

A `MPNNLayer` is a single message-passing layer that takes in a tensor of incoming messages `h_M` from edges and neighboring nodes to update node embeddings. The layer is initialized as follows:
```
mpnn_layer = MPNNLayer(vec_in, hidden_dim)
```
Here, `vec_in` is the number of vector channels in the incoming message message (`node_nv + edge_nv`). The layer is then used as follows:
```
h_V = mpnn_layer(h_V, h_M, mask=None)
```
The optional `mask` is as described above. It is also possible to use an edgewise mask `mask_attend` with dims `[batch, num_nodes, num_nodes]` --- in autoregressive sampling, for example.

Note that while we also use the _local_ node embedding as part of the message, the `MPNNLayer` itself will perform this concatenation, so you should only pass in the edge embeddings concatenated to the _neighbor_ node embeddings. That is, `h_M` should have dims `[batch, num_nodes, top_k, 3*vec_in+node_ns+edge_ns]`. This tensor can be formed as:
```
h_M = cat_neighbors_nodes(h_V, h_E, E_idx, node_nv, edge_nv)
```
The `Encoder` module is a stack of `MPNNLayers` that performs multiple graph propagation steps directly using the node and edge embeddings `h_V` and `h_E`:
```
encoder = Encoder(node_dims, edge_dims, num_layers)
h_V = encoder(h_V, h_E, E_idx, mask=None)
```
The `Decoder` module is similar, except it incorporates sequence information autoregressively as described in the paper. If you are doing something other than autoregressive protein design, `Decoder` will likely be less useful to you.

### Data pipeline

While we provide a data pipeline in `src/datasets.py`, it is specific for the training points/labels in protein design, so you will probably need to write your own for a different application. At minimum, you should modify `load_dataset` and `parse_batch` to convert your input representation to the model inputs `X`, `S`, and any necessary training labels.

## Acknowledgements
The initial implementation of portions of the protein GNN and the input data pipeline were adapted from [Ingraham, et al, NeurIPS 2019](https://github.com/jingraham/neurips19-graph-protein-design).
