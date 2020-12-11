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

_, _, testset = cath_dataset(3000) # fix this to only give individual amino acids
loss, acc, confusion = util.loop(testset, model, train=False)
print('ALL TEST PERPLEXITY {}, ACCURACY {}'.format(np.exp(loss), acc))
util.save_confusion(confusion)

_, _, testset = cath_dataset(3000, filter_file='../data/test_split_L100.json')
loss, acc, confusion = util.loop(testset, model, train=False)
print('SHORT TEST PERPLEXITY {}, ACCURACY {}'.format(np.exp(loss), acc))
util.save_confusion(confusion)

_, _, testset = cath_dataset(3000, filter_file='../data/test_split_sc.json')
loss, acc, confusion = util.loop(testset, model, train=False)
print('SINGLE CHAIN TEST PERPLEXITY {}, ACCURACY {}'.format(np.exp(loss), acc))
util.save_confusion(confusion)