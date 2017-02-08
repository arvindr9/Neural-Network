

import sys
import numpy as np
import tensorflow as tf
import numpy as np
import re
import math



def embed(file):


	with open(file, 'r') as protoSTR:
		STR=protoSTR.read()


	wordList = re.sub(ur"[\d]+", ' ', STR, re.UNICODE).split()


	
	word_vec = -1

	word_vecs = [None]*len(wordList)


	other = {
	'\xc3\x80':123, '\xc3\xa0':123,
	'\xc3\x82':124, '\xc3\xa2':124,
	'\xc3\x86':125, '\xc3\xa6':125,
	'\xc3\x87':126, '\xc3\xa7':126,
	'\xc3\x88':127, '\xc3\xa8':127,
	'\xc3\x89':128, '\xC3\xa9':128,
	'\xc3\x8a':129, '\xC3\xaa':129,
	'\xc3\x8b':130, '\xC3\xab':130,
	'\xc3\x8e':131, '\xc3\xae':131,
	'\xc3\x8f':132, '\xc3\xaf':132,
	'\xc3\x94':133, '\xc3\xb4':133,
	'\xc5\x92':134, '\xc5\x93':134,
	'\xc3\x99':135, '\xc3\xb9':135,
	'\xc3\x9b':136, '\xc3\xbb':136,
	'\xc3\x9c':137, '\xc3\xbc':137,
	'\xc4\x82':138, '\xc4\x83':138,
	'\xc8\x98':139, '\xc8\x99':139,
	'\xc5\x9e':139, '\xc5\x9f':140,
	'\xc8\x9a':141, '\xc8\x9b':141,
	'\xc5\xa2':142, '\xc5\xa3':142,
	'\xc3\xb1':143, '\xc3\x91':143,
	'\xc3\x81':144, '\xc3\xa1':144,
	'\xc3\x8c':145, '\xc3\xac':145,
	'\xc3\x8d':146, '\xc3\xad':146,
	'\xc3\x92':147, '\xc3\xb2':147,
	'\xc3\x93':148, '\xc3\xb3':148,
	'\xc3\x9a':149, '\xc3\xba':149,
	'\xc3\xa3':150, '\xc3\x83':150
	}



	for word in wordList:
		word=word.lower()
		word_vec+=1
		word_vecs[word_vec] = [0.0]*2970
		for letter in word.lower():

			if letter == '\xc3':
				y = other[letter+word[(word.index(letter) +1)]]-97
				word_vecs[word_vec][y]+=1

			elif letter == '\xc4':
				y = other[letter+word[(word.index(letter) +1)]]-97
				word_vecs[word_vec][y]+=1

			elif letter == '\xc5':
				y = other[letter+word[(word.index(letter) +1)]]-97
				word_vecs[word_vec][y]+=1

			elif letter == '\xc8':
				y = other[letter+word[(word.index(letter) +1)]]-97
				word_vecs[word_vec][y]+=1

			elif 97<=ord(letter)<=122:
				y = ord(letter)-97
				word_vecs[word_vec][y]+=1
			else:
				continue

	


	word_vec = 0

	letter_count = 0

	for words in wordList:
		words = words.lower()
		while letter_count<(len(words)-3):
			i = words[letter_count]
			j = words[letter_count+1]
			k = words[letter_count+2]
			l = words[letter_count+3]
			if ord(i) in range(97,123) and ord(j) not in range(97,123) and ord(k) not in range(97,123):
				word_vecs[word_vec][(46*ord(i))+other[j+k]-5281]+=1
				letter_count+=1

			elif ord(i) not in range(97,123) and ord(j) not in range(97,123) and ord(k) in range(97,123):

				word_vecs[word_vec][(53*other[i+j])+ord(k)-5281]+=1
				letter_count+=1

			elif ord(i) in range(97,123) and ord(j) in range(97,123) and ord(k) in range(97,123):
				word_vecs[word_vec][(53*ord(i))+ord(j)-5281]+=1
				letter_count+=1

			elif ord(i) not in range(97,123) and ord(j) not in range(97,123) and ord(k) not in range(97,123) and ord(l) not in range(97,123):
				
				word_vecs[word_vec][(53*other[i+j])+other[k+l]-5281]+=1
				letter_count+=1
			
			else:

				letter_count+=1
				continue
		word_vec+=1
		letter_count = 0


	word_vec = 0

	for vec in word_vecs:
		SUM = 0.
		for item in vec[0:54]:
			SUM+=item
		word_vecs[word_vec][0:54] = [x/float(SUM) for x in word_vecs[word_vec][0:54]]
		SUM = 0
		for items in vec[54:2970]:
			SUM+=items
		if SUM !=0:
			word_vecs[word_vec][54:2970] = [x/float(SUM) for x in word_vecs[word_vec][54:2970]]
		else:
			continue
		SUM = 0
		word_vec+=1
	return word_vecs
	word_vec = 0




y = []
word_vec_nn = embed("e-input.txt")


for vec in word_vec_nn:
	y.append([1, 0])

for vec in embed("i-input.txt"):
	y.append([0, 1])
	word_vec_nn.append(vec)
#for vec in embed("p-input.txt"):
	#y.append([0, 0,1,0,0,0])
	#word_vec_nn.append(vec)
#for vec in embed("r-input.txt"):
	#y.append([0, 0,1,0])
	#word_vec_nn.append(vec)
#for vec in embed("f-input.txt"):
	#y.append([0, 0,0,1])
	#word_vec_nn.append(vec)
#for vec in embed("s-input.txt"):
	#y.append([0, 0,0,0,0,1])
	#word_vec_nn.append(vec)


x_vals_train=[]
x_vals_test=[]
y_vals_train=[]
y_vals_test=[]
x_vals_validate = []
y_vals_validate = []

for x in range(len(word_vec_nn)):
	
	if x%5==0 or x%5==1 or x%5==2 or x%5==3:	
		x_vals_train.append(word_vec_nn[x])
		y_vals_train.append(y[x])
	elif x%5==4:

		x_vals_test.append(word_vec_nn[x])
		y_vals_test.append(y[x])
		"""if (x-4*((x+1)/5))%5==0 or (x-4*((x+1)/5))%5==1 or (x-4*((x+1)/5))%5==2 or (x-4*((x+1)/5))%5==3:	
			x_vals_test.append(word_vec_nn[x])
			y_vals_test.append(y[x])
		elif (x-4*((x+1)/5))%5==4:	
			x_vals_validate.append(word_vec_nn[x])
			y_vals_validate.append(y[x])"""
	else:
		continue




learning_rate = .001
training_epochs = 15
x = 1000
display_step = 1

x_vals = tf.placeholder("float", [None, 2970])
y_vals = tf.placeholder(tf.int64, [None, 2])




def multilayer_perceptron(word_vec_nn, weights, biases):

    layer_1 = tf.add(tf.matmul(word_vec_nn, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.zeros([2970, 256])),
    'h2': tf.Variable(tf.zeros([256, 256])),
    'out': tf.Variable(tf.zeros([256, 2]))
}
biases = {
    'b1': tf.Variable(tf.zeros([256])),
    'b2': tf.Variable(tf.zeros([256])),
    'out': tf.Variable(tf.zeros([2]))
}

pred = multilayer_perceptron(x_vals, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y_vals))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = ((len(x_vals_train)/150))
        
       
        for i in range(total_batch):
        	z = 1

        	length = len(x_vals_train)
        	while length >=150:
        		batch_x = x_vals_train[((150*z)-x): x*z]
        		batch_y = y_vals_train[((150*z)-x):z*x]
        		length -= 150

        		z+=1
        	while length>0:
        		batch_x = x_vals_train[((length*z)-length): x*z]
        		batch_y = y_vals_train[((length*z)-length): z*x]
        		length =-1
        		
        	
        	c = sess.run([optimizer, cost], feed_dict={x_vals: batch_x, y_vals: batch_y})
    
        	avg_cost = (avg_cost+ c[1]) / (total_batch)
        	for i in weights:
        		bla = tf.reduce_sum(weights[i])
        		sess.run(bla)
        	for j in biases:
        		blas = tf.reduce_sum(biases[j])
        		sess.run(blas)
        


        #tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None) 	
          	

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", 
                "{:.9f}".format(avg_cost))

    print("Optimization Finished!")


    correct_prediction = tf.equal(tf.argmax(pred, axis = 1), tf.argmax(y_vals, axis = 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x_vals: x_vals_test, y_vals: y_vals_test}))
