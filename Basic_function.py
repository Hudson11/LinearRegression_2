import tensorflow as tf
import numpy as np
import math

num_data = 150
vetor1 = np.asarray(np.random.randint(low=0, high=10, size=num_data))
vetor2 = num_data * np.asarray(np.random.randint(low=0, high=10, size=num_data))

num_training = math.floor(num_data * 0.7)

num_training_vetor1 = np.asarray(vetor1[:num_training])
num_training_vetor2 = np.asarray(vetor2[:num_training])

def normalize(array):
    return (array - array.mean()) / array.std()

num_training_vt1Norm = normalize(num_training_vetor1)
num_training_vt2Norm = normalize(num_training_vetor2)

num_test_vetor1 = np.asarray(vetor1[num_training:])
num_test_vetor2 = np.asarray(vetor2[num_training:])


tf_vetor1 = tf.placeholder(tf.float32, name='vetor1')
tf_vetor2 = tf.placeholder(tf.float32, name='vetor2')

tf_vetor1_factor = tf.Variable(np.random.randn(), name='vetor_factor')
tf_vetor2_offset = tf.Variable(np.random.randn(), name='vetor_offset')

tf_vetor2_pred = tf_vetor1_factor * tf_vetor2 + tf_vetor2_offset

tf_cost = tf_vetor2_pred + 2

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    num_interaction = 100

    for interaction in range(num_interaction):
        for (x, y) in zip(num_training_vt1Norm, num_training_vt2Norm):
            sess.run(optimizer, feed_dict={tf_vetor1: x, tf_vetor2: y})

    def normalize_single_value(value, array):
        return (value - array.mean()) / array.std()


    def denormalize(value, array):
        return value * array.std() + array.mean()

    for (vt1, vt2) in zip(num_test_vetor1, num_test_vetor2):
        value = normalize_single_value(vt1, vetor1)
        value_pred = sess.run(tf_vetor2_pred, feed_dict={tf_vetor1: value})
        value_pred = denormalize(value_pred, vetor2)
        print('vetor1: ', vt1, ' vetor2: ', vt2, 'valuePred: ', value_pred)

