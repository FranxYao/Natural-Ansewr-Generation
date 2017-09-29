import tensorflow as tf
import numpy as np

x = tf.get_variable("x", [3, 4], tf.float32, tf.zeros_initializer())
y = tf.Variable(tf.constant(np.array([1.0, 2.0, 3.0, 4.0])), tf.float32)
x[2] = x[2] + y

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(x))
