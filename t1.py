import tensorflow as tf 
import numpy as np

x = tf.placeholder(tf.int32, [4])
xget = tf.one_hot(x, 3)
embedding = tf.get_variable("embedding", [3, 5], tf.float32, tf.random_uniform_initializer(0.0, 1.0))
embed = tf.nn.embedding_lookup(embedding, x)
embed_c = tf.matmul(xget, embedding)

xin = np.array([1, 0, 2, 0])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run((embed, embed_c, xget), {x: xin}))
