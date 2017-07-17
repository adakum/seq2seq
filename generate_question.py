import os,sys,time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from seq2seq_model import Seq2SeqModel#, train
import pandas as pd
import helpers
import data_utils 

tf.reset_default_graph()
tf.set_random_seed(1)

tf.app.flags.DEFINE_float("learning_rate", 0.03, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95,"Learning rate decays by this much.")

# tf.app.flags.DEFINE_integer("optimizer", "adam", "which optimizer - adam, gradientDescent")

tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,"Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 150000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 150000, "French vocabulary size.")
tf.app.flags.DEFINE_integer("num_samples", 512, "Num samples for sampled softmax.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Num samples for sampled softmax.")

tf.app.flags.DEFINE_integer("dropout",None,"use dropout or not")
tf.app.flags.DEFINE_integer("input_keep_prob",  1,"use dropout or not")
tf.app.flags.DEFINE_integer("output_keep_prob", 1 ,"use dropout or not")
tf.app.flags.DEFINE_integer("state_keep_prob", 1 ,"use dropout or not")


tf.app.flags.DEFINE_integer("use_attention", True,"use attention or not")
tf.app.flags.DEFINE_integer("use_bidirectional", True,"use bi-directional or not")

tf.app.flags.DEFINE_integer("max_train_data_size",  0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1500, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_integer("total_epoch", 6, "Number of epoch to run")
tf.app.flags.DEFINE_integer("data_size", 13000000, "Size of Data")

# added flags for Philly use
tf.app.flags.DEFINE_boolean("local", False, "Local run status")
tf.app.flags.DEFINE_string("lstm_type", "basic", "Use BasicLSTMCell(basic) or LSTMCell(advance)"  )
tf.app.flags.DEFINE_string("tmp_model_folder", "/tmp", "Safe file to write tmp checkpoints to")
# tf.app.flags.DEFINE_string("model_folder", str(os.path.join("/hdfs/pnrsy/sys/jobs", os.environ['PHILLY_JOB_ID'], "models")),"master file to write models to")
tf.app.flags.DEFINE_string("model_folder", str(os.path.join("/hdfs/pnrsy/sys/jobs", "models")),"master file to write models to")
tf.app.flags.DEFINE_string("model_dir", "default/","master file to write models to")
tf.app.flags.DEFINE_string("test_file", "dev.in","test file to read")
# tf.app.flags.DEFINE_string("model_folder", str(os.path.join("/hdfs/pnrsy/sys/jobs", "models")),"master file to write models to")
tf.app.flags.DEFINE_string("data_dir", "./TrainingData", "Data directory")
tf.app.flags.DEFINE_string("embedding_file", "glove.840B.300d.txt", "Data directory")

tf.app.flags.DEFINE_string("train_dir", "./models", "Training directory.") # this is not actually used
# tf.app.flags.DEFINE_string("second_data_dir", sys.argv[1], 'Training directory.')
FLAGS = tf.app.flags.FLAGS
_buckets = [(2, 11), (4, 12), (6, 17), (7, 20)]
# _buckets = [(10, 11)]

print("TF Version : %s", tf.__version__)
print(".....................Printing Parameters.........................")
print("sys.version : " + sys.version)
print("learning_rate = %f" %  FLAGS.learning_rate)
print("learning_rate_decay_factor = %f" %  FLAGS.learning_rate_decay_factor)
print("max_gradient_norm = %d" %  FLAGS.max_gradient_norm)
print("batch_size = %d" %  FLAGS.batch_size)
print("num_layers = %d" %  FLAGS.num_layers)
print("hidden_size = %d" %  FLAGS.size)
print("en_vocab_size = %d" %  FLAGS.en_vocab_size)
print("fr_vocab_size = %d" %  FLAGS.fr_vocab_size)
print("num_samples = %d" %  FLAGS.num_samples)
print("_buckets = %s" %  _buckets)
print("steps_per_checkpoint = %d" %  FLAGS.steps_per_checkpoint)
print("use_lstm = True")
print("lstm_type : " + FLAGS.lstm_type)
print("total_epoch : %d" % FLAGS.total_epoch)
MAX_ITERATION_COUNT = FLAGS.total_epoch * (FLAGS.data_size)/(FLAGS.batch_size)
print("total_iteration : %d" % MAX_ITERATION_COUNT)
print("data dir : " + FLAGS.data_dir)
print("model dir : " + FLAGS.model_dir)

if not os.path.exists(FLAGS.model_dir):
  os.makedirs(FLAGS.model_dir)
  print("Created Folder !")

print(".................... Printing Parameters end .....................")

def load_embedding(file_path):
  f = open(file_path, "r")
  embedding = {}
  all_words = f.readlines()
  for x in all_words:
    w_rep = x.split(' ')
    word = w_rep[0]
    rep  = [float(x) for x in w_rep[1:]]
    embedding[word] = rep
  return embedding
 


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        # not adding now,  will add in get_batch 
        # target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          #  (len(target_ids) + 1) --> Added 1 to compensate for EOS or GO which is going to get added in get_batch part
          if len(source_ids) < source_size and (len(target_ids) + 1)< target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session):
  """Create translation model and initialize or load parameters in session."""
  
  print("To-Do Items in this !!")
  # Check later
  # _buckets use nai hua
  # max gradient norm

  model = Seq2SeqModel(encoder_cell_size=FLAGS.size,
                         vocab_size=FLAGS.en_vocab_size,
                         embedding_size=FLAGS.embedding_size,
                         attention=FLAGS.use_attention,
                         dropout=FLAGS.dropout,
                         bidirectional=FLAGS.use_bidirectional,
                         num_layers = FLAGS.num_layers,
                         learning_rate = FLAGS.learning_rate,
                         optimizer = "adam")

  # model = Seq2SeqModel(
  #     FLAGS.en_vocab_size, FLAGS.fr_vocab_size, _buckets,
  #     FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
  #     FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, use_lstm=True,
  #     num_samples = FLAGS.num_samples, forward_only=forward_only, bidirectional=True, lstm_type=FLAGS.lstm_type)

  # print the trainable variables
  for v in tf.trainable_variables():
    print(v.name)

  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  print(ckpt)

  try:
    print("Master, Trying to read model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    print("Master, Parameters Read !!")
  except Exception as e:
    print("Master, Creating model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    print("Master, Created model with fresh parameters.")
  
  return model


def train():
  print("Preparing Q2Q data in %s" % FLAGS.data_dir)
  en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_q2q_data(FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)
  
  print("Master, Going to get GLOVE!")
  embedding_file = os.path.join(FLAGS.data_dir, FLAGS.embedding_file)
  embedding = load_embedding(embedding_file)
  print("Master, GLOVE Done")
  
  with tf.Session() as sess:
    # Create model.
    print("Creating Model")
    model = create_model(sess)
    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."% FLAGS.max_train_data_size)
    print("en_dev Path : " + en_dev )
    print("fr_dev Path : " + fr_dev )
    dev_set = read_data(en_dev, fr_dev)
    dev_bucket_sizes = [len(dev_set[b]) for b in range(len(_buckets))]
    print("Development Data read !")
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
    print("Training Data read")
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]

    # loading
    en_vocab_path       = os.path.join(FLAGS.data_dir, "vocab%d.in"  % FLAGS.en_vocab_size)
    fr_vocab_path       = os.path.join(FLAGS.data_dir, "vocab%d.out" % FLAGS.fr_vocab_size)
    _, rev_en_vocab     = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab     = data_utils.initialize_vocabulary(fr_vocab_path)


    print("**** Bucket Sizes :  ")
    print(dev_bucket_sizes)
    print(train_bucket_sizes)
    print("****")

    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    model_checkpoint_list = [] # maintain list of past checkpoints



    while current_step < MAX_ITERATION_COUNT:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()

      # encoder_inputs, encoder_input_len, decoder_inputs, decoder_targets, decoder_input_len, loss_weights = model.get_batch(train_set, bucket_id, _buckets, FLAGS.batch_size)
      encoder_inputs, encoder_input_len, decoder_inputs, decoder_targets, decoder_input_len, loss_weights = model.test_get_batch(len_from = 4, len_to = 8, vocab_lower=4, vocab_upper=11, batch_size=20)
      loss_track = []
      
      fd = model.make_input_data_feed_dict(encoder_inputs, encoder_input_len, decoder_inputs, decoder_targets, decoder_input_len, loss_weights, FLAGS.input_keep_prob, FLAGS.output_keep_prob, FLAGS.state_keep_prob)
      # fd_inference = model.make_inference_inputs_II(encoder_inputs, encoder_input_len)
      # run model 
      _ , loss = sess.run([model.train_op, model.loss], fd)
      # break
      if current_step % FLAGS.steps_per_checkpoint == 0:
        print("Current Step > {}".format(model.global_step.eval()))
        print("Loss > {}".format(loss))
        loss_track.append(loss)

        # save model  
        print("Master, Going to Save Model !")
        model_checkpoint_path = os.path.join(FLAGS.model_dir, "my-model")
        model.saver.save(sess, model_checkpoint_path, global_step = model.global_step) 
        print("Master, Saving Successful !")
        
        for i, (e_in, dt_pred) in enumerate(zip(
                fd[model.encoder_inputs],
                sess.run(model.decoder_prediction_train, fd)
            )):
            print('  sample {}:'.format(i + 1))
            print('    enc input           > {}'.format(' '.join([rev_en_vocab[x] for x in e_in])))
            print('    dec train predicted > {}'.format(' '.join([rev_fr_vocab[x] for x in dt_pred])))
            print('    dec train actual > {}'.format(' '.join([rev_fr_vocab[x] for x in fd[model.decoder_train_targets][i]])))

            if i >= 10:
                break

      # step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      current_step += 1



def test():
  print("Testing")
  with tf.Session() as sess:
    # Create model.
    print("Creating Model")
    model = create_model(sess)

    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    model_checkpoint_list = [] # maintain list of past checkpoints


    while current_step < MAX_ITERATION_COUNT:
      start_time = time.time()
      encoder_inputs, encoder_input_len, decoder_inputs, decoder_targets, decoder_input_len, loss_weights = model.test_get_batch(len_from = 4, len_to = 8, vocab_lower=4, vocab_upper=11, batch_size=20)
      loss_track = []
      fd = model.make_input_data_feed_dict(encoder_inputs, encoder_input_len, decoder_inputs, decoder_targets, decoder_input_len, loss_weights, FLAGS.input_keep_prob, FLAGS.output_keep_prob, FLAGS.state_keep_prob)
      _ , loss = sess.run([model.train_op, model.loss], fd)
      # break
      if current_step % FLAGS.steps_per_checkpoint == 0:
        print("Current Step > {}".format(model.global_step.eval()))
        print("Loss > {}".format(loss))
        loss_track.append(loss)

        # save model  
        # print("Master, Going to Save Model !")
        # model_checkpoint_path = os.path.join(FLAGS.model_dir, "my-model")
        # model.saver.save(sess, model_checkpoint_path, global_step = model.global_step) 
        # print("Master, Saving Successful !")
        
        for i, (e_in, dt_pred) in enumerate(zip(
                fd[model.encoder_inputs],
                sess.run(model.decoder_prediction_train, fd)
            )):
            print('  sample {}:'.format(i + 1))
            print('    enc input           > {}'.format(e_in))
            print('    dec train predicted > {}'.format(dt_pred))
            print('    dec train actual > {}'.format(fd[model.decoder_train_targets][i]))

            if i >= 3:
                break

      # step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      current_step += 1


def decode():
  print("Not Implemented")
  print("Exiting Decoder")















# with tf.Session() as session:

#     # with bidirectional encoder, decoder state size should be
#     # 2x encoder state size
#     # model = Seq2SeqModel(encoder_cell=LSTMCell(10),
#     #                      decoder_cell=LSTMCell(20), 
#     #                      vocab_size=10,
#     #                      embedding_size=10,
#     #                      attention=True,
#     #                      dropout=None,
#     #                      bidirectional=True,
#     #                      EOS_ID = 0,
#     #                      PAD_ID = 1,
#     #                      GO_ID  = 2,
#     #                      num_layers=2)

#     model = Seq2SeqModel(encoder_cell_size = 10,
#                          decoder_cell_size = 20, 
#                          vocab_size=10,
#                          embedding_size=10,
#                          attention=True,
#                          dropout=None,
#                          bidirectional=True,
#                          EOS_ID = 0,
#                          PAD_ID = 1,
#                          GO_ID  = 2,
#                          num_layers=2)


#     session.run(tf.global_variables_initializer())
#     names = [v.name for v in tf.trainable_variables()]
#     for name in names:
#       print(name)

#     train(session, model,
#                        length_from=3, length_to=8,
#                        vocab_lower=2, vocab_upper=10,
#                        batch_size=10,
#                        max_batches=5000,
#                        batches_in_epoch=100,
#                        verbose=True,
#                        input_keep_prob=0.8,
#                        output_keep_prob=0.8,
#                        state_keep_prob=1)

#     # a = helpers.random_sequences(length_from=3, length_to=8,vocab_lower=2, vocab_upper=10,batch_size =100)
#     # print(a)
#     # for i in a:
#     #   print(np.shape(a))
def main(_):
  if FLAGS.decode:
    decode()
  elif FLAGS.self_test:
    test()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()