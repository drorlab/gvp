import tensorflow as tf
import tqdm, pdb
from tensorflow.keras import Model
from tensorflow.keras.layers import *

def norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = tf.maximum(tf.math.reduce_sum(tf.math.square(x), axis, keepdims), eps)
    return (tf.sqrt(out) if sqrt else out)

class GVP(Model):
    def __init__(self, vi, vo, so, 
                 nlv=tf.math.sigmoid, nls=tf.nn.relu):
        '''[v/s][i/o] = number of [vector/scalar] channels [in/out]'''
        super(GVP, self).__init__()
        if vi: self.wh = Dense(max(vi, vo))
        self.ws = Dense(so, activation=nls)
        if vo: self.wv = Dense(vo)
        self.vi, self.vo, self.so, self.nlv = vi, vo, so, nlv
        
    def call(self, x, return_split=False):
        # X: [..., 3*vi + si]
        # if split, returns: [..., 3, vo], [..., so[
        # if not split, returns [..., 3*vo + so]
        v, s = split(x, self.vi)
        if self.vi:
            vh = self.wh(v)
            vn = norm_no_nan(vh, axis=-2)
            out = self.ws(tf.concat([s, vn], -1))
        else: out = self.ws(s)
        if self.vo: 
            vo = self.wv(vh)
            if self.nlv: vo *= self.nlv(norm_no_nan(vo, axis=-2, keepdims=True))
            out = (vo, out) if return_split else merge(vo, out)
        return out

# Dropout that drops vector and scalar channels separately
class GVPDropout(Layer):
    def __init__(self, rate, nv):
        super(GVPDropout, self).__init__()
        self.nv = nv
        self.vdropout = Dropout(rate, noise_shape=[1, nv])
        self.sdropout = Dropout(rate)
    def call(self, x, training):
        if not training: return x
        v, s = split(x, self.nv)
        v, s = self.vdropout(v), self.sdropout(s)
        return merge(v, s)

# Normal layer norm for scalars, nontrainable norm for vectors
class GVPLayerNorm(Layer):
    def __init__(self, nv):
        super(GVPLayerNorm, self).__init__()
        self.nv = nv
        self.snorm = LayerNormalization()
    def call(self, x):
        v, s = split(x, self.nv)
        vn = norm_no_nan(v, axis=-2, keepdims=True, sqrt=False) # [..,1, nv]
        vn = tf.sqrt(tf.math.reduce_mean(vn, axis=-1, keepdims=True))
        return merge(v/vn, self.snorm(s))

# [..., 3*nv + ns] -> [..., 3, nv], [..., ns]
# nv = number of vector channels
# ns = number of scalar channels
# vector channels are ALWAYS at the top!
def split(x, nv):
    v = tf.reshape(x[..., :3*nv], x.shape[:-1] + [3, nv])
    s = x[..., 3*nv:]
    return v, s

# [..., 3, nv], [..., ns] -> [..., 3*nv + ns]
def merge(v, s):
    v = tf.reshape(v, v.shape[:-2] + [3*v.shape[-1]])
    return tf.concat([v, s], -1)

# Concat in a way that keeps vector channels at the top
def vs_concat(x1, x2, nv1, nv2):
    
    v1, s1 = split(x1, nv1)
    v2, s2 = split(x2, nv2)
    
    v = tf.concat([v1, v2], -1)
    s = tf.concat([s1, s2], -1)
    return merge(v, s)

# Aliases for compatability with the pretrained model
Velu = GVP
VGDropout = GVPDropout
VGLayerNorm = GVPLayerNorm
