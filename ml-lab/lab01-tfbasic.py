import tensorflow as tf


hi = tf.constant("hi")

sess = tf.Session()

print(sess.run(hi))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))