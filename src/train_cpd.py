import tensorflow as tf
#tf.debugging.enable_check_numerics()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from datetime import datetime
from datasets import *
import random
import tqdm, sys
import util, pdb
from models import *
def make_model():
   model = CPDModel(node_features=(8, 100), edge_features=(1,32), hidden_dim=(16,100))
   return model

def main():
  trainset, valset, testset = cath_dataset(1800, jsonl_file=sys.argv[1])# batch size = 1800 residues
  optimizer = tf.keras.optimizers.Adam()
  model = make_model()
  
  model_id = int(datetime.timestamp(datetime.now()))

  NUM_EPOCHS = 100
  loop_func = util.loop
  best_epoch, best_val = 0, np.inf
    
  for epoch in range(NUM_EPOCHS):   
    loss, acc, confusion = loop_func(trainset, model, train=True, optimizer=optimizer)
    util.save_checkpoint(model, optimizer, model_id, epoch)
    print('EPOCH {} TRAIN {:.4f} {:.4f}'.format(epoch, loss, acc))
    util.save_confusion(confusion)
    loss, acc, confusion = loop_func(valset, model, train=False)
    if loss < best_val:
        best_epoch, best_val = epoch, loss
    print('EPOCH {} VAL {:.4f} {:.4f}'.format(epoch, loss, acc))
    util.save_confusion(confusion)

  # Test with best validation loss
  path = util.models_dir.format(str(model_id).zfill(3), str(epoch).zfill(3))
  util.load_checkpoint(model, optimizer, path)  
  loss, acc, confusion = loop_func(testset, model, train=False)
  print('EPOCH TEST {:.4f} {:.4f}'.format(loss, acc))
  util.save_confusion(confusion)
    

main()
