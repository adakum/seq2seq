import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from seq2seq_model import Seq2SeqModel, train
import pandas as pd
import helpers


tf.reset_default_graph()
tf.set_random_seed(1)

tf.app.flags.DEFINE_float("learning_rate", 2, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95,"Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,"Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 150000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 150000, "French vocabulary size.")
tf.app.flags.DEFINE_integer("num_samples", 512, "Num samples for sampled softmax.")

tf.app.flags.DEFINE_integer("dropout",None,"use dropout or not")
tf.app.flags.DEFINE_integer( "input_keep_prob",  1,"use dropout or not")
tf.app.flags.DEFINE_integer("output_keep_prob", 1 ,"use dropout or not")

tf.app.flags.DEFINE_integer("use_attention",True,"use attention or not")
tf.app.flags.DEFINE_integer("use_bidirectional", True,"use bi-directional or not")

tf.app.flags.DEFINE_integer("max_train_data_size",  0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1500, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_integer("total_epoch",6, "Number of epoch to run")
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
tf.app.flags.DEFINE_string("train_dir", "./models", "Training directory.") # this is not actually used
# tf.app.flags.DEFINE_string("second_data_dir", sys.argv[1], 'Training directory.')
FLAGS = tf.app.flags.FLAGS
_buckets = [(3, 7), (5, 12), (7, 17), (9, 20)]

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
                         EOS_ID = 0,
                         PAD_ID = 1,
                         GO_ID  = 2,
                         num_layers = FLAGS.num_layers)

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
    session.run(tf.initialize_all_variables())
    print("Master, Created model with fresh parameters.")
  
  return model


def train():
  print("Preparing Q2Q data in %s" % FLAGS.data_dir)
  en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_q2q_data(FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)

  with tf.Session() as sess:
    # Create model.
    print("Creating Model")
    model = create_model(sess, False)
    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."% FLAGS.max_train_data_size)
    print("en_dev Path : " + en_dev )
    print("fr_dev Path : " + fr_dev )
    dev_set = read_data(en_dev, fr_dev)
    dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
    print("Development Data read ! ")
    train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
    print("Training Data read")
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]

    print("**** Bucket Sizes :  ")
    print(dev_bucket_sizes)
    print(train_bucket_sizes)
    print("****")

    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    model_checkpoint_list = [] # maintain list of past checkpoints
    min_eval_perplex = collections.defaultdict(float)
    
    for bucket_id in xrange(len(_buckets)):
      min_eval_perplex[bucket_id] = float("inf")

    while current_step < MAX_ITERATION_COUNT:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      
      # print(current_step)
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        num = (current_step/MAX_ITERATION_COUNT)*100
        print()
        print('LOSS : %.2f%%' % loss)
        print()
        print('PROGRESS: %.2f%%' % num)
        print()
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        if FLAGS.local is False:
          tmp_model_checkpoint_path = os.path.join(FLAGS.tmp_model_folder, "model.ckpt-" + str(current_step))
          # model_checkpoint_path = os.path.join(FLAGS.model_dir, "model.ckpt-" + str(current_step))
          model_checkpoint_path = os.path.join(FLAGS.model_dir, "my-model")
          tmp_checkpoint_path   = os.path.join(FLAGS.tmp_model_folder, "checkpoint")
          # checkpoint_path = os.path.join("/hdfs/pnrsy/sys/jobs", os.environ['PHILLY_JOB_ID'], "models", "translate.ckpt")
          checkpoint_path = os.path.join(FLAGS.model_dir, "checkpoint")

          if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
            print("Created Folder !")
          try:
            # print(tmp_checkpoint_path)
            # print(tmp_model_checkpoint_path)
            print(model_checkpoint_path)
            model.saver.save(sess, model_checkpoint_path, global_step=model.global_step)
            print("Saved Model")
            # model_checkpoint_list.append(model_checkpoint_path)
          except Exception as e:
           print("FAILED TO COPY FOR CHECKPOINT FOR FILE %s" % model_checkpoint_path)
           try:
             print(e.message)
           except Exception as ee:
             print("NO EXCEPTION MESSAGE")
          if len(model_checkpoint_list) > 5:
            os.remove(model_checkpoint_list[0])
            model_checkpoint_list.pop(0)
        else:
          checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
          # model.saver.save(sess, checkpoint_path, global_step=model.global_step)



















with tf.Session() as session:

    # with bidirectional encoder, decoder state size should be
    # 2x encoder state size
    # model = Seq2SeqModel(encoder_cell=LSTMCell(10),
    #                      decoder_cell=LSTMCell(20), 
    #                      vocab_size=10,
    #                      embedding_size=10,
    #                      attention=True,
    #                      dropout=None,
    #                      bidirectional=True,
    #                      EOS_ID = 0,
    #                      PAD_ID = 1,
    #                      GO_ID  = 2,
    #                      num_layers=2)

    model = Seq2SeqModel(encoder_cell_size = 10,
                         decoder_cell_size = 20, 
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
                       max_batches=5000,
                       batches_in_epoch=100,
                       verbose=True,
                       input_keep_prob=0.8,
                       output_keep_prob=0.8,
                       state_keep_prob=1)

    # a = helpers.random_sequences(length_from=3, length_to=8,vocab_lower=2, vocab_upper=10,batch_size =100)
    # print(a)
    # for i in a:
    #   print(np.shape(a))
