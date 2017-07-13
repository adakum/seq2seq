import math,random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from data_utils import *
import helpers

class Seq2SeqModel():
    """Seq2Seq model usign blocks from new `tf.contrib.seq2seq`.
    Requires TF 1.0.0-alpha"""


    def __init__(self, encoder_cell_size, vocab_size, embedding_size,
                 bidirectional=True,
                 attention=False,
                 debug=False,
                 dropout=None,
                 PAD_ID=0,
                 GO_ID =1,
                 EOS_ID=2,
                 num_layers=1):

        # self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_layers = num_layers
        self.dropout = dropout

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # add ability to choose cell
        self.encoder_cell = LSTMCell(encoder_cell_size)
        
        if self.bidirectional:
            self.decoder_cell = LSTMCell(2*encoder_cell_size)
        else :
            self.decoder_cell = LSTMCell(encoder_cell_size)

    
        self.EOS_ID  = EOS_ID
        self.PAD_ID  = PAD_ID
        self.GO_ID  = GO_ID
        
        self._make_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size

    def _make_graph(self):
        self._init_placeholders()
        self._init_cells()

        # self._init_decoder_train_connectors()
        self._init_embeddings()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        self._init_decoder()

        self._init_optimizer()

    def _init_cells(self):

        if self.dropout is not None:
            self.encoder_cell = tf.contrib.rnn.DropoutWrapper(
                                                                self.encoder_cell,
                                                                input_keep_prob  = self.input_keep_prob,
                                                                output_keep_prob = self.output_keep_prob
                                                                # , state_keep_prob  = self.state_keep_prob
                                                                )
            self.decoder_cell = tf.contrib.rnn.DropoutWrapper(
                                                                self.decoder_cell,
                                                                input_keep_prob  = self.input_keep_prob,
                                                                output_keep_prob = self.output_keep_prob
                                                                # , state_keep_prob  = self.state_keep_prob
                                                                )
        if self.num_layers > 1:
            self.encoder_cell = tf.contrib.rnn.MultiRNNCell([self.encoder_cell] * self.num_layers)
            self.decoder_cell = tf.contrib.rnn.MultiRNNCell([self.decoder_cell] * self.num_layers)

    def _init_placeholders(self):

        # A list of 1D int32 Tensors of shape [batch_size].
        self.encoder_inputs = tf.placeholder(
            shape = [None, None],
            dtype = tf.int32,
            name  = 'encoder_inputs',
        )
        
        # A list of shape [batch_size]
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',)
        
        self.decoder_train_inputs = tf.placeholder(
            shape = (None, None),
            dtype = tf.int32,
            name = 'decoder_train_inputs',)

        self.decoder_train_targets = tf.placeholder(
            shape = (None, None),
            dtype = tf.int32,
            name = 'decoder_train_targets')

        self.decoder_train_length = tf.placeholder(
            shape = (None,),
            dtype = tf.int32,
            name  = 'decoder_train_length')

        self.loss_weights = tf.placeholder(
            shape = (None, None),
            dtype = tf.float32,
            name = 'loss_weights')

        # # decoder length, A list of shape [batch_size]
        # self.decoder_targets_length = tf.placeholder(
        #     shape=(None,),
        #     dtype=tf.int32,
        #     name='decoder_targets_length',
        # )

        self.input_keep_prob  = tf.placeholder(tf.float32, name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
        # self.state_keep_prob  = tf.placeholder(tf.float32, name="state_keep_prob")


    # def _init_decoder_train_connectors(self):
    #     """
    #     During training, `decoder_targets`
    #     and decoder logits. This means that their shapes should be compatible.
    #     Here we do a bit of plumbing to set this up.
        
    #     """
    #     with tf.name_scope('DecoderTrainFeeds'):
    #         # sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))
    #         batch_size, sequence_size = tf.unstack(tf.shape(self.decoder_targets))

    #         # 
    #         # EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS_ID
    #         # PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD_ID
    #         EOS_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self.EOS_ID
    #         PAD_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self.PAD_ID
    #         GO_SLICE  = tf.ones([batch_size, 1], dtype=tf.int32) * self.GO_ID

    #         # self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
    #         # self.decoder_train_length = self.decoder_targets_length + 1
    #         # self.decoder_train_inputs = tf.concat([self.decoder_targets, EOS_SLICE], axis=1)
            
    #         # decoder train input will have just a GO at the start
    #         self.decoder_train_inputs  = tf.concat([GO_SLICE, self.decoder_targets]  , axis=1)
    #         self.decoder_train_length  = self.decoder_targets_length + 1 #  1-GO
            
    #         # decoder train target will have EOS at end
    #         self.decoder_train_targets = tf.concat([self.decoder_targets, EOS_SLICE] , axis=1)

    #         # decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
    #         # decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
    #         # decoder_train_targets_eos_mask   = tf.one_hot(self.decoder_train_length - 1,
    #         #                                             decoder_train_targets_seq_len,
    #         #                                             on_value=self.EOS, off_value=self.PAD,
    #         #                                             dtype=tf.int32)
    #         # decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])

    #         # hacky way using one_hot to put EOS symbol at the end of target sequence
    #         # decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)
    #         # self.decoder_train_targets = decoder_train_targets

    #         self.loss_weights = tf.ones([batch_size, tf.reduce_max(self.decoder_train_length)], dtype = tf.float32, name="loss_weights")

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:

            # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.encoder_inputs)

            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=False,
                                  dtype=tf.float32)
                )

    def _init_bidirectional_encoder(self):

        with tf.variable_scope("BidirectionalEncoder") as scope:
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw =self.encoder_cell,
                                                cell_bw =self.encoder_cell,
                                                inputs  =self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=False,
                                                dtype=tf.float32)
                )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            self.encoder_state = []

            if self.num_layers>1:
                for i in range(self.num_layers):
                    if isinstance(encoder_fw_state[i], LSTMStateTuple):
                        encoder_state_c = tf.concat((encoder_fw_state[i].c, encoder_bw_state[i].c), 1, name='bidirectional_concat_c')
                        encoder_state_h = tf.concat((encoder_fw_state[i].h, encoder_bw_state[i].h), 1, name='bidirectional_concat_h')
                        encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
                    elif isinstance(encoder_fw_state[i], tf.Tensor):
                        encoder_state = tf.concat((encoder_fw_state[i], encoder_bw_state[i]), 1, name='bidirectional_concat')
                    self.encoder_state.append(encoder_state)

                self.encoder_state = tuple(self.encoder_state)
            else:
                encoder_state_c = tf.concat((encoder_fw_state.c, encoder_bw_state.c), 1, name ='bidirectional_concat_c')
                encoder_state_h = tf.concat((encoder_fw_state.h, encoder_bw_state.h), 1, name ='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            
            # if self.num_layers>1:
            #     self.encoder_state   = tf.concat((encoder_fw_state, encoder_bw_state), -1)
            #     # self.encoder_state   = tf.transpose(self.encoder_state, perm=[2,0,1,3])

            # elif isinstance(encoder_fw_state, LSTMStateTuple):
            #     print("blah")
            #     encoder_state_c = tf.concat((encoder_fw_state.c, encoder_bw_state.c), 1, name ='bidirectional_concat_c')
            #     encoder_state_h = tf.concat((encoder_fw_state.h, encoder_bw_state.h), 1, name ='bidirectional_concat_h')
            #     self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            # elif isinstance(encoder_fw_state, tf.Tensor):
            #     self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name  ='bidirectional_concat')

    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                # return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)
                return tf.contrib.layers.fully_connected(outputs, self.vocab_size, scope=scope)

            if not self.attention:
                decoder_fn_train     = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.GO_ID,
                    end_of_sequence_id=self.EOS_ID,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size,
                )
            else:

                # attention_states: size [batch_size, max_time, num_units]
                # [adkuma] Already in that format
                # attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
                attention_states = self.encoder_outputs 
                print("decoder hidden ")
                print(self.decoder_hidden_units)
                (attention_keys,
                attention_values,
                attention_score_fn,
                attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )

                decoder_fn_train  = seq2seq.attention_decoder_fn_train(
                    encoder_state = self.encoder_state,
                    attention_keys = attention_keys,
                    attention_values = attention_values,
                    attention_score_fn = attention_score_fn,
                    attention_construct_fn = attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.GO_ID,
                    end_of_sequence_id=self.EOS_ID,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size,
                )


            (self.decoder_outputs_train,
             self.decoder_state_train,
             self.decoder_context_state_train) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=False,
                    scope=scope,
                )
            )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=False,
                    scope=scope,
                )
            )

            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

    def _init_optimizer(self):
        logits  = self.decoder_logits_train
        targets = self.decoder_train_targets
        self.loss     = seq2seq.sequence_loss(logits=logits, targets=targets, weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def make_train_inputs(self, input_seq, target_seq):
        inputs_, inputs_length_ = helpers.batch(input_seq)
        targets_, targets_length_ = helpers.batch(target_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_,
        }

    def make_train_inputs_II(self, input_seq, target_seq, input_keep_prob, output_keep_prob, state_keep_prob):
        inputs_, inputs_length_ = helpers.batch_II(input_seq)
        targets_, targets_length_ = helpers.batch_II(target_seq)
        return{
            self.encoder_inputs:inputs_,
            self.encoder_inputs_length:inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length : targets_length_,
            self.input_keep_prob : input_keep_prob,
            self.output_keep_prob : output_keep_prob
            # self.state_keep_prob  : state_keep_prob
        }
    def make_input_data_feed_dict(self, encoder_inputs, encoder_input_len,  decoder_inputs, decoder_targets, decoder_input_len, loss_weights, input_keep_prob, output_keep_prob, state_keep_prob):
        return{
            self.encoder_inputs : encoder_inputs,
            self.encoder_inputs_length :  encoder_input_len,
            self.decoder_train_inputs :  decoder_inputs,
            self.decoder_train_targets : decoder_targets,
            self.decoder_train_length : decoder_input_len,
            self.loss_weights : loss_weights,
            self.input_keep_prob : input_keep_prob,
            self.output_keep_prob : output_keep_prob
        }


    def make_inference_inputs(self, input_seq):
        inputs_, inputs_length_ = helpers.batch_II(input_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
        }
    def make_inference_inputs_II(self, input_seq, input_seq_len):
        # inputs_, inputs_length_ = helpers.batch_II(input_seq)
        return {
            self.encoder_inputs: input_seq,
            self.encoder_inputs_length: input_seq_len,
        }

    def get_batch(self, data, bucket_id, buckets, batch_size):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = buckets[bucket_id]
        encoder_inputs, decoder_inputs, decoder_targets = [], [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        real_encoder_input_len = []
        real_decoder_input_len = []

        for _ in range(batch_size):
          # print(bucket_id)
          # print(np.shape(data[bucket_id]))
          encoder_input, decoder_input = random.choice(data[bucket_id])
          real_encoder_input_len.append(len(encoder_input))

          # Encoder inputs are padded and then reversed.
          encoder_pad = [self.PAD_ID] * (encoder_size - len(encoder_input))
          # TO-DO : Should consider reversing or not, since bidirectional mein does it matter
          encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

          # Decoder inputs get an extra "GO" symbol, and are padded then.
          real_decoder_input_len.append(len(decoder_input) + 1) # Adding 1 to consider for GO and EOS
          decoder_pad_size = decoder_size - len(decoder_input) - 1
          
          decoder_inputs.append([self.GO_ID] + decoder_input + [self.PAD_ID] * decoder_pad_size)
          decoder_targets.append(decoder_input +[self.EOS_ID] + [self.PAD_ID]* decoder_pad_size)


        batch_weights = np.ones([batch_size, decoder_size])
        
        for batch_idx in range(batch_size):
            for len_idx in range(decoder_size):
                if len_idx == decoder_size-1:
                    batch_weights[batch_idx, len_idx] = 0
                else:
                    if decoder_inputs[batch_idx][len_idx + 1] == self.PAD_ID:
                        batch_weights[batch_idx, len_idx] = 0

        # # Now we create batch-major vectors from the data selected above.
        # batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # # Batch encoder inputs are just re-indexed encoder_inputs.
        # for length_idx in xrange(encoder_size):
        #   batch_encoder_inputs.append(
        #       np.array([encoder_inputs[batch_idx][length_idx]
        #                 for batch_idx in xrange(batch_size)], dtype=np.int32))

        # # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        # for length_idx in xrange(decoder_size):
        #   batch_decoder_inputs.append(
        #       np.array([decoder_inputs[batch_idx][length_idx]
        #                 for batch_idx in xrange(batch_size)], dtype=np.int32))

        #   # Create target_weights to be 0 for targets that are padding.
        #   batch_weight = np.ones(batch_size, dtype=np.float32)
        #   for batch_idx in xrange(batch_size):
        #     # We set weight to 0 if the corresponding target is a PAD symbol.
        #     # The corresponding target is decoder_input shifted by 1 forward.
        #     if length_idx < decoder_size - 1:
        #       target = decoder_inputs[batch_idx][length_idx + 1]
        #     if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
        #       batch_weight[batch_idx] = 0.0
        #   batch_weights.append(batch_weight)
        # return batch_encoder_inputs, batch_decoder_inputs, batch_weights
        return encoder_inputs, real_encoder_input_len,  decoder_inputs, decoder_targets, real_decoder_input_len, batch_weights


def make_seq2seq_model(**kwargs):
    args = dict(encoder_cell=LSTMCell(10),
                decoder_cell=LSTMCell(20),
                vocab_size=10,
                embedding_size=10,
                attention=True,
                bidirectional=True,
                debug=False)
    args.update(kwargs)
    return Seq2SeqModel(**args)


# def train(session, model,
#                 length_from=3, length_to=8,
#                 vocab_lower=2, vocab_upper=10,
#                 batch_size=100,
#                 max_batches=5000,
#                 batches_in_epoch=1000,
#                 verbose=True,
#                 input_keep_prob=1,
#                 output_keep_prob=1,
#                 state_keep_prob=1):
    
#     print(input_keep_prob)
#     batches = helpers.random_sequences(length_from=length_from, length_to=length_to,
#                                        vocab_lower=vocab_lower, vocab_upper=vocab_upper,
#                                        batch_size=batch_size)
#     loss_track = []
#     try:
#         for batch in range(max_batches+1):
#             batch_data = next(batches)
#             fd = model.make_train_inputs_II(batch_data, batch_data, input_keep_prob, output_keep_prob, state_keep_prob)
#             fd_inference = model.make_inference_inputs(batch_data)
#             _, l = session.run([model.train_op, model.loss], fd)
#             loss_track.append(l)

#             if verbose:
#                 if batch == 0 or batch % batches_in_epoch == 0:
#                     print('batch {}'.format(batch))
#                     print('  minibatch loss: {}'.format(session.run(model.loss, fd)))
#                     for i, (e_in, dt_pred) in enumerate(zip(
#                             fd[model.encoder_inputs],
#                             session.run(model.decoder_prediction_train, fd)
#                         )):
#                         print('  sample {}:'.format(i + 1))
#                         print('    enc input           > {}'.format(e_in))
#                         print('    dec train predicted > {}'.format(dt_pred))
#                         if i >= 2:
#                             break
#                     for i, (e_in, dt_pred) in enumerate(zip(
#                             fd_inference[model.encoder_inputs],
#                             session.run(model.decoder_prediction_inference, fd_inference)
#                         )):
#                         print('  sample {}:'.format(i + 1))
#                         print('    enc input           > {}'.format(e_in))
#                         print('    dec train predicted > {}'.format(dt_pred))
#                         if i >= 2:
#                             break
#                     print()
#     except KeyboardInterrupt:
#         print('training interrupted')

#     return loss_track
