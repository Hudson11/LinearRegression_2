

import tensorflow as tf

hello = tf.Constant('Hello')
sess = tf.Session()
print(sess.run(hello))