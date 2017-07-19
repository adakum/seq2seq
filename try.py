import tensorflow as tf
tf.app.flags.DEFINE_integer("data_size", 13000000, "Size of Data")
FLAGS = tf.app.flags.FLAGS

x = FLAGS.data_size 

FLAGS.data_size = FLAGS.data_size + 3

print(FLAGS.data_size)