
import tensorflow as tf
import numpy as np


vetor1 = np.random.randint(low=5, high=10, size=100)
vetor2 = vetor1 * 2 - np.random.randint(low=5, high=10, size=100)

W = tf.cast(tf.get_variable("W", initializer=tf.constant(1.0)), tf.float64)
b = tf.cast(tf.get_variable("b", initializer=tf.constant(1.0)), tf.float64)

dataset = tf.data.Dataset.from_tensor_slices(vetor1, vetor2)

iterator = dataset.make_initializable_iterator()
X,y = iterator.get_next()

y_pred = W * X + b

loss = tf.reduce_sum(tf.square(y_pred - y))
learning_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(iterator)

    for i in range(1000):
        sess.run()