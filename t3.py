import tensorflow as tf 
x = tf.get_variable("x", [2, 3, 2], tf.float32, tf.random_normal_initializer())
W = tf.get_variable("W", [2, 2], tf.float32, tf.random_normal_initializer())
# y = tf.matmul(x, tf.expand_dims(W, 0)) # Note: you cannot do this !
print(y.shape)
