import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from seq2seq_model import Seq2SeqModel, train
import pandas as pd
import helpers

tf.reset_default_graph()
tf.set_random_seed(1)

with tf.Session() as session:

    # with bidirectional encoder, decoder state size should be
    # 2x encoder state size
    model = Seq2SeqModel(encoder_cell=LSTMCell(10),
                         decoder_cell=LSTMCell(20), 
                         vocab_size=10,
                         embedding_size=10,
                         attention=True,
                         dropout=None,
                         bidirectional=True,
                         EOS_ID = 0,
                         PAD_ID = 1,
                         GO_ID  = 2,
                         num_layers=2)
    session.run(tf.global_variables_initializer())
    names = [v.name for v in tf.trainable_variables()]
    for name in names:
    	print(name)

    train(session, model,
                       length_from=3, length_to=8,
                       vocab_lower=2, vocab_upper=10,
                       batch_size=10,
                       max_batches=3000,
                       batches_in_epoch=100,
                       verbose=True,
                       input_keep_prob=0.8,
                       output_keep_prob=0.8,
                       state_keep_prob=1)

    # a = helpers.random_sequences(length_from=3, length_to=8,vocab_lower=2, vocab_upper=10,batch_size =100)
    # print(a)
    # for i in a:
    # 	print(np.shape(a))
