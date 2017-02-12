
import sys
import numpy as np
import tensorflow as tf
import numpy as np
import re
import math
import random


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
	'\xc3\x89':128, '\xc3\xa9':128,
	'\xc3\x8a':129, '\xc3\xaa':129,
	'\xc3\x8b':130, '\xc3\xab':130,
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
		words.lower()
		if '\\' in repr(words):
			while letter_count<(len(words)-2):
				
				j = None
				k = None
				l = None
				i = words[letter_count]
				try:
					j = words[letter_count+1]
				except IndexError:

					pass
				try:
					k = words[letter_count+2]
				except IndexError:
					pass
				try:
					l = words[letter_count+3]
				except IndexError:
					pass

				if '\\' in repr(i) and '\\' in repr(j) and '\\' in repr(k) and '\\' in repr(l):
					word_vecs[word_vec][(54*other[i+j])+other[k+l]-5281]+=1
					letter_count+=1
				elif ('\\' in repr(i) and '\\' in repr(j) and '\\' in repr(k)) or ('\\' in repr(j) and '\\' in repr(k) and '\\' in repr(l)):
					if '\\' in repr(j) and '\\' in repr(k) and '\\' in repr(l):
						word_vecs[word_vec][(54*ord(i))+other[j+k]-5281]+=1
						letter_count+=1
					else:
						pass
				elif ("\\" in repr(i) and "\\" in repr(j)) or ("\\" in repr(j) and "\\" in repr(k)) or ("\\" in repr(k) and "\\" in repr(l)):
					if "\\" in repr(i) and "\\" in repr(j):

						word_vecs[word_vec][(54*other[i+j])+ord(k)-5281]+=1
						letter_count+=1

					elif "\\" in repr(j) and "\\" in repr(k):
						word_vecs[word_vec][(54*ord(i))+other[j+k]-5281]+=1
						letter_count+=1
					elif "\\" in repr(k) and "\\" in repr(l):
						word_vecs[word_vec][(54*ord(i))+ord(j)-5281]+=1
						letter_count+=1
					else:
						pass
				elif "\\" in repr(l):
					word_vecs[word_vec][(54*ord(i))+ord(j)-5281]+=1
					letter_count+=1
				elif ("\\" in repr(i)) and (l == None):
					word_vecs[word_vec][(54*ord(j))+ord(k)-5281]+=1
					letter_count+=1
				elif ('\\' in repr(i)) and ('\\' in repr(l)):
					word_vecs[word_vec][(54*ord(j))+ord(k)-5281]+=1
					letter_count+=1
				elif (k==None and l==None) or (l==None) or ('\\' not in repr(i) and '\\' not in repr(j) and '\\' not in repr(k) and '\\' not in repr(l)):

					word_vecs[word_vec][(54*ord(i))+ord(j)-5281]+=1
					letter_count+=1

				else:
					letter_count+=1

					pass

			word_vec+=1
			letter_count=0
		else:
			while letter_count<(len(words)-1):
 				i = words[letter_count]
 				j = words[letter_count+1]
 				word_vecs[word_vec][(54*ord(i))+ord(j)-5281]+=1

 				letter_count+=1
 			word_vec+=1
 		letter_count=0
	word_vec = 0


	"""for vec in word_vecs:
		SUM = 0.
		for item in vec[:54]:
			SUM+=item
		#print SUM
		word_vecs[word_vec][:54] = [x/float(SUM) for x in word_vecs[word_vec][:54]]
		#print word_vecs[word_vec][:54]
		SUM = 0
		for items in vec[54:]:
			SUM+=items
		#print SUM
		if SUM !=0:
			word_vecs[word_vec][54:] = [x/float(SUM) for x in word_vecs[word_vec][54:]]
		else:
			continue
		
		SUM = 0
		word_vec+=1"""



	word_vec = 0
	return word_vecs





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
"""for vec in embed("r-input.txt"):
	y.append([0, 0,1,0])
	word_vec_nn.append(vec)
for vec in embed("f-input.txt"):
	y.append([0, 0,0,1])
	word_vec_nn.append(vec)"""
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




learning_rate = .0005
training_epochs = 20
x = 50


x_vals = tf.placeholder("float", [None, input_dim])
y_vals = tf.placeholder(tf.int64, [None, 2])



def multilayer_perceptron(word_vec_nn, weights, biases):

    layer_1 = tf.add(tf.matmul(word_vec_nn, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([input_dim, 300])),
    'h2': tf.Variable(tf.random_normal([300, 300])),
    'out': tf.Variable(tf.random_normal([300, 2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([300])),
    'b2': tf.Variable(tf.random_normal([300])),
    'out': tf.Variable(tf.random_normal([2]))
}

pred = multilayer_perceptron(x_vals, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y_vals))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = ((len(x_vals_train)/x))
		indices = []
		for x in range(0,8000):

			indices.append(x)

		random.shuffle(indices)
		
		for i in range(total_batch):
			
			z = 0

			length = len(x_vals_train)
			batch_x = []
			batch_y = []
			for i in range(x*z,x+x*z):

				
				batch_x.append(x_vals_train[indices[i]])
				batch_y.append(y_vals_train[indices[i]])

			z+=1

			c = sess.run([optimizer, cost], feed_dict={x_vals: batch_x, y_vals: batch_y})

			avg_cost = (avg_cost+ c[1]) / (total_batch)



		print("Epoch:", '%04d' %(epoch+1), "cost=", "{:.9f}".format(avg_cost))
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_vals, 1))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		np.asarray(x_vals_test)
		np.asarray(y_vals_test)
		print("Accuracy:", accuracy.eval({x_vals: x_vals_test, y_vals: y_vals_test}))

	print("Optimization Finished!")


	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_vals, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	np.asarray(x_vals_test)
	np.asarray(y_vals_test)
	print("Accuracy:", accuracy.eval({x_vals: x_vals_test, y_vals: y_vals_test}))







