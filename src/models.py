# Portions of this code were adapted from 
# https://github.com/jingraham/neurips19-graph-protein-design

import numpy as np
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from gvp import *

class MQAModel(Model):
    def __init__(self, node_features, edge_features,
        hidden_dim, num_layers=3, k_neighbors=30, dropout=0.1):
            
        super(MQAModel, self).__init__()
        
        # Hyperparameters
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features
        
        # Featurization layers
        self.features = StructuralFeatures(node_features, edge_features, top_k=k_neighbors)
    
        # Embedding layers
        self.W_s = Embedding(20, self.hs)        
        self.W_v = GVP(vi=self.nv, vo=self.hv, so=self.hs,
                        nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.ev, so=self.hs,
                        nls=None, nlv=None)
        
        self.encoder = Encoder(hidden_dim, edge_features, num_layers=num_layers, 
                                dropout=dropout)
                                
        self.W_V_out = GVP(vi=self.hv, vo=0, so=self.hs,
                          nls=None, nlv=None)
        
        self.dense = Sequential([
            Dense(2 * self.hs, activation='relu'),
            Dropout(rate=dropout),
            Dense(2 * self.hs, activation='relu'),
            Dropout(rate=dropout),
            LayerNormalization(),
            Dense(1, activation=None)])
        
    def call(self, X, S, mask, train=False):
        # X [B, N, 4, 3], S [B, N], mask [B, N]

        V, E, E_idx = self.features(X, mask)
        h_S = self.W_s(S)
        V = vs_concat(V, h_S, self.nv, 0)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=train)
        
        h_V_out = self.W_V_out(h_V) 
        mask = tf.expand_dims(mask, -1) # [B, N, 1]

        if train:
          h_V_out = tf.math.reduce_mean(h_V_out * mask, -2) # [B, N, D] -> [B, D]
        else:
          h_V_out = tf.math.reduce_sum(h_V_out * mask, -2) # [B, N, D] -> [B, D]
          h_V_out = tf.math.divide_no_nan(h_V_out, tf.math.reduce_sum(mask, -2)) # [B, D]
        out = h_V_out
        out = tf.squeeze(self.dense(out, training=train), -1) + 0.5 # [B]
        
        return out

class CPDModel(Model):
    def __init__(self, node_features, edge_features,
        hidden_dim, num_layers=3, num_letters=20, k_neighbors=30, dropout=0.1):
        super(CPDModel, self).__init__()
        
        # Hyperparameters
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features
        
        # Featurization layers
        self.features = StructuralFeatures(node_features, edge_features, top_k=k_neighbors)
    
        # Embedding layers
        self.W_v = GVP(vi=self.nv, vo=self.hv, so=self.hs,
                        nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.ev, so=self.hs,
                        nls=None, nlv=None)
        self.W_s = Embedding(num_letters, self.hs)        
        self.encoder = Encoder(hidden_dim, edge_features, num_layers=num_layers)
        self.decoder = Decoder(hidden_dim, edge_features, s_features=(0, self.hs), num_layers=num_layers)
        self.W_out = GVP(vi=self.hv, vo=0, so=num_letters,
                          nls=None, nlv=None)


    def call(self, X, S, mask, train=False):
        # X [B, N, 4, 3], S [B, N], mask [B, N]

        V, E, E_idx = self.features(X, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=train)
        h_S = self.W_s(S)
        h_V = self.decoder(h_V, h_S, h_E, E_idx, mask, train=train)
        logits = self.W_out(h_V) 
        
        return logits
        
    def sample(self, X, mask=None, temperature=0.1):
        V, E, E_idx = self.features(X,  mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        h_V = self.encoder(h_V, h_E, E_idx, mask, train=False)
        return self.decoder.sample(h_V, h_E, E_idx, mask, W_s=self.W_s, W_out=self.W_out, temperature=0.1)
    
class Encoder(Model):
    def __init__(self, node_features, edge_features, num_layers=3, dropout=0.1):
        super(Encoder, self).__init__()
        
        # Hyperparameters
        self.nv, ns = node_features
        self.ev, _ = edge_features
        
        # Encoder layers 
        self.vglayers = [                               
            MPNNLayer(self.nv + self.ev, node_features, dropout=dropout)
            for _ in range(num_layers)
        ]

    def call(self, h_V, h_E, E_idx, mask, train=False):
        # Encoder is unmasked self-attention
        
        mask_attend = tf.squeeze(gather_nodes(tf.expand_dims(mask,-1),E_idx),-1)
        # [B, N] => [B, N, 1] => [B, N, K, 1] => [B, N, K] 
        mask_attend = tf.expand_dims(mask,-1) * mask_attend
        
        for layer in self.vglayers:
            h_M = cat_neighbors_nodes(h_V, h_E, E_idx, self.nv, self.ev) # nv = self.hv + 1
            h_V = layer(h_V, h_M, mask_V=mask, mask_attend=mask_attend, train=train)
            
        return h_V
    
class Decoder(Model): # DECODER
    def __init__(self, node_features, edge_features,
                 s_features, num_layers=3, dropout=0.1):
        super(Decoder, self).__init__()
        
        # Hyperparameters
        self.nv, self.ns = node_features
        self.ev, self.es = edge_features
        self.sv, self.ss = s_features
        
        # Decoder layers
        self.vglayers = [
            MPNNLayer(self.nv + self.ev, node_features, dropout=dropout)
            for _ in range(num_layers)              
        ]
        
    def call(self, h_V, h_S, h_E, E_idx, mask, train=False):
        # h_V [B, N, *], h_S [B, N, *], mask [B, N]
        # h_E [B, N, K, *], E_idx [B, N, K]
        
        # Concatenate sequence embeddings for autoregressive decoder
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx, 0, self.ev) # nv = 1

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(tf.zeros_like(h_S), h_E, E_idx, self.sv, self.ev) # nv = 1
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx, self.nv, self.sv + self.ev) # nv = self.nv+1

        # Decoder uses masked self-attention
        mask_attend = tf.expand_dims(autoregressive_mask(E_idx), -1)
        mask_1D = tf.cast(tf.expand_dims(tf.expand_dims(mask, -1), -1), tf.float32)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        
        for layer in self.vglayers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx, self.nv, self.ev) # nv = self.nv + 1
            h_M = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = layer(h_V, h_M, mask_V=mask, train=train)
        
        return h_V
        
    # This is slow because TensorFlow doesn't allow indexed tensor writes
    # at runtime, so we have to move between CPU/GPU at every step.
    # If you can find a way around this, it will run a lot faster 
    def sample(self, h_V, h_E, E_idx, mask, W_s, W_out, temperature=0.1):
        mask_attend = tf.expand_dims(autoregressive_mask(E_idx), -1)
        mask_1D = tf.reshape(mask, [mask.shape[0], mask.shape[1], 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        
        N_batch, N_nodes = h_V.shape[0], h_V.shape[1] 
        
        h_S = np.zeros((N_batch, N_nodes, self.ss), dtype=np.float32)
        S = np.zeros((N_batch, N_nodes), dtype=np.int32)
        h_V_stack = [tf.split(h_V, N_nodes, 1)] + [tf.split(tf.zeros_like(h_V), N_nodes, 1) for _ in range(len(self.vglayers))]
        for t in tqdm.trange(N_nodes):
            # Hidden layers
            E_idx_t = E_idx[:,t:t+1,:]
            h_E_t = h_E[:,t:t+1,:,:]
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t, 0, self.ev)
            # Stale relational features for future states
            h_ESV_encoder_t = mask_fw[:,t:t+1,:,:] * cat_neighbors_nodes(h_V, h_ES_t, E_idx_t, self.nv, self.ev)
            for l, layer in enumerate(self.vglayers):
                # Updated relational features for future states
                h_ESV_decoder_t = cat_neighbors_nodes(tf.stack(h_V_stack[l], 1), h_ES_t, E_idx_t, self.nv, self.ev)
                h_V_t = h_V_stack[l][t] #[:,t:t+1,:]
                h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_ESV_encoder_t
                mask_to_pass = mask[:,t:t+1]
                tmp = layer(h_V_t, h_ESV_t, mask_V=mask_to_pass)
                h_V_stack[l+1][t] = tmp
            # Sampling step
            h_V_t = tf.squeeze(h_V_stack[-1][t], 1) #[:,t,:]
            logits = W_out(h_V_t) / temperature # this is the main issue, where to get W_out?
            #probs = F.softmax(logits, dim=-1)
            S_t = tf.squeeze(tf.random.categorical(logits, 1), -1)

            # Update
            h_S[:,t,:] = W_s(S_t) # where to get W_S?
            S[:,t] = S_t
        return S

class MPNNLayer(Model):
    def __init__(self, vec_in, num_hidden, dropout=0.1):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.vec_in = vec_in
        self.vo, self.so = vo, so = num_hidden
        self.norm = [GVPLayerNorm(vo) for _ in range(2)]
        self.dropout = GVPDropout(dropout, vo)
        
        # this receives the vec_in message AND the receiver node
        self.W_EV = Sequential([GVP(vi=vec_in+vo, vo=vo, so=so), 
                             GVP(vi=vo, vo=vo, so=so),
                             GVP(vi=vo, vo=vo, so=so, nls=None, nlv=None)])
        
        self.W_dh = Sequential([GVP(vi=vo, vo=2*vo, so=4*so),
                                 GVP(vi=2*vo, vo=vo, so=so, nls=None, nlv=None)])
        
    def call(self, h_V, h_M, mask_V=None, mask_attend=None, train=False):
        # Concatenate h_V_i to h_E_ij
        h_V_expand = tf.tile(tf.expand_dims(h_V,-2), [1,1,tf.shape(h_M)[-2],1])
        h_EV = vs_concat(h_V_expand, h_M, self.vo, self.vec_in)
        h_message = self.W_EV(h_EV)
        if mask_attend is not None:
            h_message = tf.cast(tf.expand_dims(mask_attend, -1), tf.float32) * h_message
        dh = tf.math.reduce_mean(h_message, -2)
        h_V = self.norm[0](h_V + self.dropout(dh, training=train))

        # Position-wise feedforward
        dh = self.W_dh(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh, training=train))

        if mask_V is not None:
            mask_V = tf.cast(tf.expand_dims(mask_V, -1), tf.float32)
            h_V = mask_V * h_V
            
        return h_V

def autoregressive_mask(E_idx):
    N_nodes = tf.shape(E_idx)[1]
    ii = tf.range(N_nodes)
    ii = tf.reshape(ii, [1, -1, 1])
    mask = E_idx - ii < 0
    mask = tf.cast(mask, tf.float32)
    return mask
    
def normalize(tensor, axis=-1):
    return tf.math.divide_no_nan(tensor, tf.linalg.norm(tensor, axis=axis, keepdims=True))

def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    edge_features = tf.gather(edges, neighbor_idx, axis=2, batch_dims=2)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK]
    neighbors_flat = tf.reshape(neighbor_idx, [neighbor_idx.shape[0], -1])

    # Gather and re-pack
    # nodes [B, N, C], neighbors_flat [B, NK, C] => [B, NK, C]
    # tf: nf[i][j][k] = nodes[i][nf[i][j]][k]
    neighbor_features = tf.gather(nodes, neighbors_flat, axis=1, batch_dims=1)     
    neighbor_features = tf.reshape(neighbor_features, list(neighbor_idx.shape)[:3] + [-1]) # => [B, N, K, C]
    return neighbor_features
   
def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx, nv_nodes, nv_neighbors):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return vs_concat(h_neighbors, h_nodes, nv_neighbors, nv_nodes)
    
class PositionalEncodings(Model):
    def __init__(self, num_embeddings, period_range=[2,1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range 

    def call(self, E_idx):
        # i-j
        N_batch = tf.shape(E_idx)[0]
        N_nodes = tf.shape(E_idx)[1]
        N_neighbors = tf.shape(E_idx)[2]
        ii = tf.reshape(tf.cast(tf.range(N_nodes), tf.float32), (1, -1, 1))
        d = tf.expand_dims((tf.cast(E_idx, tf.float32) - ii), -1)
        # Original Transformer frequencies
        frequency = tf.math.exp(
            tf.cast(tf.range(0, self.num_embeddings, 2), tf.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = d * tf.reshape(frequency, (1,1,1,-1))
        E = tf.concat((tf.math.cos(angles), tf.math.sin(angles)), -1)
        return E

class StructuralFeatures(Model):
    def __init__(self, node_features, edge_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30):
        super(StructuralFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        
        # Normalization and embedding
        vo, so = node_features
        ve, se = edge_features
        self.node_embedding = GVP(vi=3, vo=vo, so=so,
                                   nlv=None, nls=None)
        self.edge_embedding = GVP(vi=1, vo=ve, so=se,
                                   nlv=None, nls=None)
        self.norm_nodes = LayerNormalization()
        self.norm_edges = LayerNormalization()
    
    def _dist(self, X, mask, eps=1E-6): # [B, N, 3]
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask = tf.cast(mask, tf.float32)
        mask_2D = tf.expand_dims(mask,1) * tf.expand_dims(mask,2)
        dX = tf.expand_dims(X,1) - tf.expand_dims(X,2)
        D = mask_2D * tf.math.sqrt(tf.math.reduce_sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max = tf.math.reduce_max(D, -1, keepdims=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = tf.math.top_k(-D_adjust, 
                                k=min(self.top_k, tf.shape(X)[1]))
        D_neighbors = -D_neighbors
        mask_neighbors = gather_edges(tf.expand_dims(mask_2D, -1), E_idx)
        
        return D_neighbors, E_idx, mask_neighbors
   
    def _directions(self, X, E_idx):
      
        dX = X[:,1:,:] - X[:,:-1,:]
        X_neighbors = gather_nodes(X, E_idx)
        dX = X_neighbors - tf.expand_dims(X, -2)
        dX = normalize(dX, axis=-1)      
        return dX
        
    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = tf.linspace(D_min, D_max, D_count)
        D_mu = tf.reshape(D_mu, [1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = tf.expand_dims(D, -1)
        RBF = tf.math.exp(-((D_expand - D_mu) / D_sigma)**2)
        
        return RBF
    
    def _orientations(self, X):
        # X: B, N, 3
        forward = normalize(X[:,1:] - X[:,:-1])
        backward = normalize(X[:,:-1] - X[:,1:])
        forward = tf.pad(forward, [[0,0], [0, 1], [0,0]])
        backward = tf.pad(backward, [[0,0], [1, 0], [0,0]])
        return tf.concat([tf.expand_dims(forward, -1), tf.expand_dims(backward,-1)], -1) # B, N, 3, 2
        
    def _sidechains(self, X):
        # ['N', 'CA', 'C', 'O']
        # X: B, N, 4, 3
        n, origin, c = X[:,:,0,:], X[:,:,1,:], X[:,:,2,:]
        c, n = normalize(c-origin), normalize(n-origin)
        bisector = normalize(c + n)
        perp = normalize(tf.linalg.cross(c, n))
        vec = -bisector * tf.math.sqrt(1/3) - perp * tf.math.sqrt(2/3)
        return vec # B, N, 3

    
    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = tf.reshape(X[:,:,:3,:], [tf.shape(X)[0], 3*tf.shape(X)[1], 3])
        
        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = normalize(dX, axis=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        
        # Backbone normals
        n_2 = normalize(tf.linalg.cross(u_2, u_1), axis=-1)
        n_1 = normalize(tf.linalg.cross(u_1, u_0), axis=-1)

        # Angle between normals
        cosD = tf.math.reduce_sum(n_2 * n_1, -1)
        cosD = tf.clip_by_value(cosD, -1+eps, 1-eps)
        D = tf.math.sign(tf.math.reduce_sum(u_2 * n_1, -1)) * tf.math.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = tf.pad(D, [[0,0], [1,2]]) # what dims!
        D = tf.reshape(D, [tf.shape(D)[0], int(tf.shape(D)[1]/3), 3])
        
        # Lift angle representations to the circle
        D_features = tf.concat([tf.math.cos(D), tf.math.sin(D)], 2)
        return D_features
        
    def call(self, X, mask):
        """ Featurize coordinates as an attributed graph """
        
        # Build k-Nearest Neighbors graph
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)
        
        # Pairwise features
        E_directions = self._directions(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)
        E_positional = self.embeddings(E_idx)
        
        # Full backbone angles
        V_dihedrals = self._dihedrals(X)
        V_orientations = self._orientations(X_ca)
        V_sidechains = self._sidechains(X)
        
        V_vec = tf.concat([tf.expand_dims(V_sidechains, -1), V_orientations], -1)
        V = merge(V_vec, V_dihedrals)
        E = tf.concat([E_directions, RBF, E_positional], -1)
        
        # Embed the nodes
        Vv, Vs = self.node_embedding(V, return_split=True)
        V = merge(Vv, self.norm_nodes(Vs))
        
        Ev, Es = self.edge_embedding(E, return_split=True)
        E = merge(Ev, self.norm_edges(Es))
        
        return V, E, E_idx

# Aliases for compatability with the pretrained model
VGEncoder = Encoder
VGDecoder = Decoder
