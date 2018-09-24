import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt

# Receiving data on temperature and air velocity

arq1 = open('dados/temperatura.txt', 'r')
arq2 = open('dados/velocidadeDoAr.txt', 'r')

dado1 = arq1.readlines()
dado2 = arq2.readlines()

# convert data
temperature = np.asarray([float(i) for i in dado1])
velocity = np.asarray([float(i) for i in dado2])

# function normalize
def normalize(array):
    return (array - array.mean()) / array.std()


data_size = len(temperature)
# 70% data training
num_training = math.floor(data_size * 0.7)

# Getting the data
train_num_temperature = np.asarray(temperature[:num_training])
train_num_velocity = np.asarray(velocity[:num_training])
train_num_tempNorm = normalize(train_num_temperature)
train_num_veloNorm = normalize(train_num_velocity)

# 30% data test
test_temperature = np.asarray(temperature[num_training:])
test_velocity = np.asarray(velocity[num_training:])


# application tensorFlow

tf_num_temperature = tf.placeholder(tf.float32, name="house_size")
tf_num_velocity = tf.placeholder(tf.float32, name='price')

tf_velocity_factor = tf.Variable(np.random.randn(), name='velocity_factor')
tf_temp_offset = tf.Variable(np.random.randn(), name='temperature_offset')

# Regression Linear Equation of the line
tf_temp_pred = tf_velocity_factor * tf_num_velocity + tf_temp_offset
tf_predTemp = tf.reduce_sum(tf.pow(tf_temp_pred - tf_num_temperature, 2)) / (2 * num_training)

# adjusted value
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_predTemp)

init = tf.global_variables_initializer()

# Initialize session tensorFlow

with tf.Session() as sess:
    sess.run(init)
    num_interaction = 100

    for iteration in range(num_interaction):
        for (x, y) in zip(train_num_veloNorm, train_num_tempNorm):
            sess.run(optimizer, feed_dict={tf_num_velocity: x, tf_num_temperature: y})

    training_temp = sess.run(tf_predTemp, feed_dict={tf_num_velocity: train_num_veloNorm, tf_num_temperature: train_num_tempNorm})

    def normalize_single_value(value, array):
        return (value - array.mean()) / array.std()

    def denormalize(value, array):
        return value * array.std() + array.mean()

    for (vel, temp) in zip(test_velocity, test_temperature):
        value = normalize_single_value(vel, velocity)
        temp_prediction = sess.run(tf_predTemp, feed_dict={tf_num_velocity: value})
        temp_prediction = denormalize(temp_prediction, temperature)
        print('velocity: ',vel,' original temp: ',temp, ' temp predction: ',temp_prediction)

