import tensorflow as tf
import numpy as np

# preprocessed data

from data import parser
from data import data_utils

# load data from pickle and npy files
parser.process_data();
metadata, idx_q, idx_a = parser.load_data(PATH='data/')
(trainX, trainY), (testX, testY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

from model import seq2seq as seq2seq_wrapper

# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )


# In[8]:

val_batch_gen = data_utils.rand_batch_gen(testX, testY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


# In[9]:
#sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)