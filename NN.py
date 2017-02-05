#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import sys
import codecs
import numpy as np

from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import re
from unidecode import unidecode


#reload(sys)  
#sys.setdefaultencoding('utf-8')

def embed():
	file = raw_input("Enter file name: ")

	with open(file, 'r') as protoSTR:
		STR=protoSTR.read()


	wordList = re.sub('[\W\d_]', " ", STR).split()

	word_vec = -1

	word_vecs = [None]*len(wordList)


	#other = {'á':123, 'é':124, 'í':125, 'ó':126, 'ú':127, 'ñ':128, 'Á':123, 'É':124, 'Í':125, 'Ó':126, 'Ú':127, 'Ñ':128, 'ü':129, 'Ü':129}

	for word in wordList:
		word=word.lower()
		word_vec+=1
		word_vecs[word_vec] = [0.0]*702
		for letter in word:
			y = ord(letter)-97
			word_vecs[word_vec][y]+=1

	word_vec = 0

	letter_count = 0

	for words in wordList:
		words = words.lower()
		while letter_count<len(words)-1:
			i = words[letter_count]
			j = words[letter_count+1]
			word_vecs[word_vec][(26*ord(i))+ord(j)-2593]+=1
			letter_count+=1
		word_vec+=1
		letter_count = 0

	word_vec = 0

	for vec in word_vecs:
		SUM = 0.
		for item in vec[0:25]:
			SUM+=item
		word_vecs[word_vec][0:25] = [x/float(SUM) for x in word_vecs[word_vec][0:25]]
		SUM = 0
		for items in vec[26:702]:
			SUM+=items
		if SUM !=0:
			word_vecs[word_vec][26:702] = [x/float(SUM) for x in word_vecs[word_vec][26:702]]
		else:
			continue
		SUM = 0
		word_vec+=1
	return word_vecs
	word_vec = 0



y = []
word_vec_nn = embed()

for vec in word_vec_nn:
	y.append([1, 0])


for vec in embed():
	y.append([0, 1])
	word_vec_nn.append(vec)

print y
x_vals_train=[]
x_vals_test=[]
y_vals_train=[]
y_vals_test=[]
x_vals_validate = []
y_vals_validate = []
for x in range(len(word_vec_nn)):
	
	if x%5==(0 or 1 or 2 or 3):	
		x_vals_train.append(word_vec_nn[x])
		y_vals_train.append(y[x])
	elif x%5==4:
		print word_vec_nn[x]	
		x_vals_test.append(word_vec_nn[x])
		y_vals_test.append(y[x])
		"""if (x-4((x+1)/5))%5==(0 or 1 or 2 or 3):	
			x_vals_test.append(word_vec_nn[x])
			y_vals_test.append(y[x])
		elif (x-4((x+1)/5))%5==4:	
			x_vals_validate.append(word_vec_nn[x])
			y_vals_validate.append(y[x])"""
	else:
		continue
print(x_vals_test, len(x_vals_test))

"""
np.asarray(x_vals_train)
np.asarray(y_vals_train)
np.asarray(x_vals_test)
np.asarray(y_vals_test)
"""



learning_rate = 0.001
training_epochs = 15
x = 50
display_step = 1

x_vals = tf.placeholder("float", [None, 702])
y_vals = tf.placeholder(tf.int64, [None, 2])

print (y_vals_test, len(y_vals_test))


def multilayer_perceptron(word_vec_nn, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(word_vec_nn, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([702, 256])),
    'h2': tf.Variable(tf.random_normal([256, 256])),
    'out': tf.Variable(tf.random_normal([256, 2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([256])),
    'b2': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([2]))
}

# Construct model
pred = multilayer_perceptron(x_vals, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y_vals))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = len(x_vals_train)
        z = 1
        # Loop over all batches
        for i in range(total_batch):
        	length = len(x_vals_train)
        	while length >50:
        		batch_x = x_vals_train[((50*z)-50): x*z]
        		batch_y = y_vals_train[((50*z)-50):z*x]
        		length -= 50
        		z+=1
        	batch_x = x_vals_train[((length*z)-length): x*z]
        	batch_y = y_vals_train[((length*z)-length):z*x]
        	
        	#map(list,zip(*batch_y))
        	c = sess.run([optimizer, cost], feed_dict={x_vals: batch_x, y_vals: batch_y})
        	
        	avg_cost = (avg_cost+ c[1]) / (total_batch)
          
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, axis = 1), tf.argmax(y_vals, axis = 1))
    #print correct_prediction.eval({x_vals: x_vals_test, y_vals: y_vals_test})
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x_vals: x_vals_test, y_vals: y_vals_test}))
