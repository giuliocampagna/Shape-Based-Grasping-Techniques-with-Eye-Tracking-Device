import csv
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.python import keras

emg_data = np.empty([0, 8]) #Array that contains the EMG data
Targets = np.empty([0, 3]) #Array containing the one hot representation vectors
c = 51 #Number of closing action data
o = 44 #Number of opening action data
r = 40 #Number of rest action data
# Define the number of closing, rest and opening data in c, r, o

#Save the data stored in the csv files related to closing
for j in range(c):
    fileh = open(
        "/home/destiny/anaconda3/envs/virtual_environment/catkin_ws/src/ros_myo/scripts/EMG_data/Closing/Data/c" + str(
            j + 1) + ".csv", 'r+')
    import matplotlib.pyplot as plt

    data = pd.read_csv(fileh)
    emg_data_raw = data.data
    dim = len(emg_data_raw)
    for i in range(dim):
        clean = emg_data_raw[i]
        clean = clean[clean.find('[') + 1:]
        clean = clean[:clean.find(']')]
        clean = clean.split(',')
        for k in range(len(clean)):
            clean[k] = int(clean[k])
        emg_data = np.r_[emg_data, np.asarray(clean).reshape(1, 8)]

    # How to create targets closing
c_tot = np.size(emg_data, 0)
#Update the one hot representation array
targets = np.ones([c_tot, 1])
targets = np.c_[targets, np.zeros([c_tot, 2])]

#Save the data stored in the csv files related to rest
for j in range(r):
    fileh = open(
        "/home/destiny/anaconda3/envs/virtual_environment/catkin_ws/src/ros_myo/scripts/EMG_data/Rest/Data/r" + str(
            j + 1) + ".csv", 'r+')
    import matplotlib.pyplot as plt

    data = pd.read_csv(fileh)
    emg_data_raw = data.data
    dim = len(emg_data_raw)
    for i in range(dim):
        clean = emg_data_raw[i]
        clean = clean[clean.find('[') + 1:]
        clean = clean[:clean.find(']')]
        clean = clean.split(',')
        for k in range(len(clean)):
            clean[k] = int(clean[k])
        emg_data = np.r_[emg_data, np.asarray(clean).reshape(1, 8)]

    # How to create targets res
r_tot = np.size(emg_data, 0) - c_tot
#Update the one hot representation array
new_targets = np.zeros([r_tot, 1])
new_targets = np.c_[new_targets, np.ones([r_tot, 1])]
new_targets = np.c_[new_targets, np.zeros([r_tot, 1])]
targets = np.r_[targets, new_targets]

#Save the data stored in the csv files related to opening
for j in range(o):
    fileh = open(
        "/home/destiny/anaconda3/envs/virtual_environment/catkin_ws/src/ros_myo/scripts/EMG_data/Opening/Data/o" + str(
            j + 1) + ".csv", 'r+')
    import matplotlib.pyplot as plt

    data = pd.read_csv(fileh)
    emg_data_raw = data.data
    dim = len(emg_data_raw)
    for i in range(dim):
        clean = emg_data_raw[i]
        clean = clean[clean.find('[') + 1:]
        clean = clean[:clean.find(']')]
        clean = clean.split(',')
        for k in range(len(clean)):
            clean[k] = int(clean[k])
        emg_data = np.r_[emg_data, np.asarray(clean).reshape(1, 8)]

    # How to create targets opening
o_tot = np.size(emg_data, 0) - c_tot - r_tot
#Update the one hot representation array
new_targets = np.zeros([o_tot, 2])
new_targets = np.c_[new_targets, np.ones([o_tot, 1])]
targets = np.r_[targets, new_targets]

#Function used to create a neural network with tensorflow
def create_network():
    d = 8  # number of input features
    c = 3  # number of classes
    hidden_layer_size = 8  # number of neurons in the hidden layer

    X = tf.placeholder(tf.float32, [None, d])
    Y = tf.placeholder(tf.float32, [None, c])

    W1 = tf.get_variable("W1", initializer=tf.random_normal([d, hidden_layer_size]))
    b1 = tf.get_variable("b1", initializer=tf.random_normal([1, hidden_layer_size]))

    W2 = tf.get_variable("W2", initializer=tf.random_normal([hidden_layer_size, c]))
    b2 = tf.get_variable("b2", initializer=tf.random_normal([1, c]))

    A1 = tf.matmul(X, W1) + b1
    O1 = tf.nn.relu(A1)

    A2 = tf.matmul(O1, W2) + b2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=A2))
    optimizer = tf.train.AdamOptimizer()
    # optimizer = tf.train.GradientDescentOptimizer(0.015)

    train_step = optimizer.minimize(loss)
    # train_step = tf.train.MomentumOptimizer(0.015,0.2).minimize(loss)

    correct_predictions = tf.equal(tf.argmax(A2, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) * 100.0

    return X, Y, loss, train_step, accuracy

#Main loop: where is performed the machine learning algorithm
def main():
    # randomizing data
    randomize = np.arange(len(emg_data))
    np.random.shuffle(randomize)
    data = emg_data[randomize, :]
    targets = Targets[randomize, :]

    # splitting into training and test sets
    train_set_data = data[0:int(round(data.shape[0] * 0.7))]
    train_set_targets = targets[0:int(round(data.shape[0] * 0.7))]
    test_set_data = data[int(round(data.shape[0] * 0.7)):, :]
    test_set_targets = targets[int(round(data.shape[0] * 0.7)):, :]

    # creating the network and defining all the needed TensorFlow operations
    X, Y, loss, train_step, accuracy = create_network()
    with tf.Session() as sess:
        # initializing variables
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # number of training epochs
        max_epochs = 20000
        delta_loss = 10000
        prev_loss = 10000000

        # training epochs
        # for i in range(0,max_epochs):
        i = 0
        while (delta_loss > 0 or i < 20000):
            output = sess.run([loss, train_step, accuracy], feed_dict={X: train_set_data, Y: train_set_targets})
            if (i % 800 == 0):
                print("Epoch " + str(i) + ", Loss=" + "{0:f}".format(output[0]) + ", " +
                      "TrainAccuracy=" + "{0:.2f}".format(output[2]) + "%")
                delta_loss = prev_loss - output[0]
                prev_loss = output[0]
            # print( str(i) + "  " + "{0:f}".format(output[0]) + "  " + "{0:.2f}".format(output[2]) + ";")
            i = i + 1
        save_path = saver.save(sess, "my_net/save_net.ckpt") #Save the neural network inside a ckpt file
        print("Model saved in path: %s" % save_path)
        # test
        te_acc = sess.run(accuracy, feed_dict={X: test_set_data, Y: test_set_targets})
        print("TestAccuracy=" + "{0:.2f}".format(te_acc) + "%")


if __name__ == "__main__":
    main()
