import tensorflow as tf
import numpy as np
import tqdm
import scipy.stats, pdb
from collections import defaultdict

models_dir = '../models/{}_{}'

# Here these lookups are only used for labeling the confusion matrix,
# so feel free to use your own / disregard them
three_to_id = {'CYS': 4, 'ASP': 3, 'SER': 15, 'GLN': 5, 'LYS': 11, 'ILE': 9, 'PRO': 14, 'THR': 16, 'PHE': 13, 'ALA': 0, 'GLY': 7, 'HIS': 8, 'GLU': 6, 'LEU': 10, 'ARG': 1, 'TRP': 17, 'VAL': 19, 'ASN': 2, 'TYR': 18, 'MET': 12}
three_to_one = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
id_to_one = {val : three_to_one[key] for key, val in three_to_id.items()}

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)

# Train / val / test loop for protein design
def loop(dataset, model, train=False, optimizer=None):
  acc_metric.reset_states()
  loss_metric.reset_states()
  mat = np.zeros((20, 20))
  for batch in tqdm.tqdm(dataset):
    X, S, M = batch
    if train:
      with tf.GradientTape() as tape:
        logits = model(*batch, train=True)
        loss_value = loss_fn(S, logits, sample_weight=M)
    else:
      logits = model(*batch, train=False)
      loss_value = loss_fn(S, logits, sample_weight=M)
    if train:
      grads = tape.gradient(loss_value, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
    acc_metric.update_state(S, logits, sample_weight=M)
    loss_metric.update_state(S, logits, sample_weight=M)
    pred = tf.math.argmax(logits, axis=-1)
    mat += tf.math.confusion_matrix(tf.reshape(S, [-1]),
              tf.reshape(pred, [-1]), weights=tf.reshape(M, [-1]))
  loss, acc = loss_metric.result(), acc_metric.result()
  return loss, acc, mat

# Save model and optimizer state
def save_checkpoint(model, optimizer, model_id, epoch):
  path = models_dir.format(str(model_id).zfill(3), str(epoch).zfill(3))
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  ckpt.write(path) 
  print('CHECKPOINT SAVED TO ' + path)

# Load model and optimizer state
def load_checkpoint(model, optimizer, path):
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
  ckpt.restore(path)
  print('CHECKPOINT RESTORED FROM ' + path)

# Pretty print confusion matrix
def save_confusion(mat):
  counts = mat.numpy()
  mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
  mat = np.round(mat * 1000).astype(np.int32)
  res = '\n'
  for i in range(20):
    res += '\t{}'.format(id_to_one[i])
  res += '\n'
  for i in range(20):
    res += '{}\t'.format(id_to_one[i])
    res += '\t'.join('{}'.format(n) for n in mat[i])
    res += '\t{}\n'.format(sum(counts[i]))
  print(res)

