import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(1000, 1)
y_data = 7 * x_data - 3 + np.random.randn(1000, 1)

W = tf.cast(tf.get_variable("W", initializer=tf.constant(1.0)), tf.float64)
b = tf.cast(tf.get_variable("b", initializer=tf.constant(1.0)), tf.float64)

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

iterator = dataset.make_initializable_iterator()
X, y = iterator.get_next()

y_pred = W * X + b

loss = tf.reduce_mean(tf.square(y_pred - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)

training = optimizer.minimize(loss)

init = tf.global_variables_initializer()

writer = tf.summary.FileWriter('./graphs/graph', tf.get_default_graph())
write_loss = tf.summary.FileWriter('./graphs/loss')
write_w = tf.summary.FileWriter('./graphs/W')

lss = tf.summary.scalar("loss", loss)
w = tf.summary.scalar("W", W)

with tf.Session() as sess:
    sess.run(iterator.initializer)

    sess.run(init)
    dados = []
    for i in range(1000):

        try:

            loss_batch, _ = sess.run([loss, training])
            if i % 5 == 0:
                print('Loss : ', loss_batch)

                dados.append(loss_batch)

                summary = sess.run(lss)
                write_loss.add_summary(summary, i)
                write_loss.flush()

                summary_2 = sess.run(w)
                write_w.add_summary(summary_2, i)
                write_w.flush()

        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)

