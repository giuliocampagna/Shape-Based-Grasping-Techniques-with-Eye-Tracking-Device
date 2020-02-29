import rospy
from ros_myo.msg import EmgArray
from std_msgs.msg import String
import tensorflow as tf
import numpy as np

#Parameters to define a neural network of the shape of the saved one
d = 8
c = 3
hidden_layer_size = 8
W1 = tf.get_variable("W1", initializer=tf.random_normal([d,hidden_layer_size]))
b1 = tf.get_variable("b1", initializer=tf.random_normal([1,hidden_layer_size]))
W2 = tf.get_variable("W2", initializer=tf.random_normal([hidden_layer_size,c]))
b2 = tf.get_variable("b2", initializer=tf.random_normal([1,c]))

global counter #Variable used to publish a message only under determined condition (see the next lines)
counter = np.array([0, 0, 0])

#Restore the neural network
saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, "my_net/save_net.ckpt")
	print("W1",sess.run(W1))
	print("b1",sess.run(b1))
	print("W2",sess.run(W2))
	print("b2",sess.run(b2))
	W1 = W1.eval(session = sess)
	b1 = b1.eval(session = sess)
	W2 = W2.eval(session = sess)
	b2 = b2.eval(session = sess)

#Function used to do the prediction about an incoming EMG reading, done restoring and running the neural network
def prediction(x):
	A1 = x.dot(W1) + b1
	for i in range(A1.shape[1]):
		if A1[0,i]<0:
			A1[0,i]=0
	A2 = A1.dot(W2) + b2
	z = -10000
	index = 0
	for i in range(A2.shape[1]):
		if A2[0,i]>z:
			index = i
			z = A2[0,i]
	return index

states = ["Closing", "Rest", "Opening"] #Vector containing the possible values published by the node

#Callback function. It publishes a "rest action" only when 15 consecutives patterns are predicted, 2 for closing and 1 for opening
def callback(msg):
		#Restore the neural network
	global counter
	indice = prediction(np.array(msg.data))	
	if indice == 0:
		counter = np.array([counter[0]+1, 0, 0])
	if indice == 1:
		counter = np.array([0, counter[1]+1, 0])
	if indice == 2:
		counter = np.array([0, 0, counter[2]+1])
	if np.sum(counter[0]) >= 2 or np.sum(counter[1]) >= 15 or np.sum(counter[2]) >= 1 :
		pub.publish(states[indice]) #con predizione che e' una stringa
		print(states[indice])
	

pub = rospy.Publisher("EMG_prediction", String, queue_size=10)  #The published. It will contain a string: "closing", "opening" or "rest"
rospy.init_node('prediction')
sub = rospy.Subscriber('/myo_emg', EmgArray , callback) #Subscribe to '/myo_emg', the topic containing the EMG data
rospy.spin()
