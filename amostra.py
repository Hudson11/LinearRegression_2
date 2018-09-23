
import tensorflow as tf

g = tf.Graph()

with g.as_default():
    a = tf.add(10, 20)

sess = tf.Session(graph=g)
print(sess.run(a))

