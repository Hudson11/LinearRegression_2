import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt


#
# A ideia para mudança deste algoritmo é:
#   -> Usar uma base de dados ou usar dados gravados em arquivos do tipo Text.
#   -> Usar para fazer para treinar o modelo de preedição usando uma forma de regressão que pode ser Linear ou polinomial.
#   -> Pesquisa uma forma de receber dados constatemente sem ter que passar no modelo de treinamento
#   -> Otimizar código, usar boas práticas de programação para diminuir a complexidade do algoritmo.
#

def normalize(array):
    return (array - array.mean()) / array.std()


arq1 = open('dados/temperatura.txt', 'r')
arq2 = open('dados/velocidadeDoAr.txt', 'r')

dado1 = arq1.readlines()
dado2 = arq2.readlines()

# convert data
house_size = np.asarray([float(i) for i in dado1])
house_price = np.asarray([float(i) for i in dado2])

num_house = len(house_size)

print(len(house_size))
print(len(house_price))

# ake pegaremós 70% dos dados randômicos gerados para treinar o modelo de predição inteligênte
num_train_samples = math.floor(num_house * 0.7)

# Dados de treinamento 70%
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])
train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# 30% dos valores gerados para o teste
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

#######################################################################################################################
#######################################################################################################################

tf_house_size = tf.placeholder(tf.float32, name='house_size')
tf_price = tf.placeholder(tf.float32, name='price')

tf_size_factor = tf.Variable(np.random.randn(), name='size_factor')
tf_price_offset = tf.Variable(np.random.randn(), name='price_offset')

tf_price_pred = tf_size_factor * tf_house_size + tf_price_offset

tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2)) / (2 * num_train_samples)

learning_rate = 0.01 #taxa de aprendizagem, valor ajustável. Busca o melhor possível.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    num_training_iter = 2000
    for iteration in range(num_training_iter):
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})

    def normalize_single_value(value, array):
        return (value - array.mean()) / array.std()

    def denormalize(value, array):
        return value * array.std() + array.mean()

    dados = []

    for (size, price) in zip(test_house_size, test_house_price):
        value = normalize_single_value(size, house_size)
        price_prediction = sess.run(tf_price_pred, feed_dict={tf_house_size:value})
        price_prediction = denormalize(price_prediction, house_price )
        dados.append(price_prediction)
        print("House size:",size, " Original price:", price, " Price Prediction:", price_prediction, "Diff:", (price_prediction - price))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(test_house_price, color='lightblue', linewidth=3)
ax.plot(dados, color='darkgreen')
plt.show()


