import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.python.framework import ops
import random
#ops.reset_default_graph()


sess = tf.Session()

x = 0
Input = []
while x< 25:
	Input.append([0,0,0,random.randint(0, 20),random.randint(0, 20),random.randint(0, 20), 0])
	x+=1
while 25<=x<50:
	Input.append([random.randint(0,20),random.randint(0, 20),random.randint(0, 20),0,0,0, 1])
	x+=1



x_vals = np.array([[x[0],x[1],x[2],x[3],x[4],x[5]] for x in Input])
y_vals = np.array([(x[6]) for x in Input])
# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare batch size
batch_size = 50

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 6], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
line = tf.placeholder(shape = [2, 1], dtype = tf.float32)
# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[6,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))



# Declare model operations
model_output = tf.sub(tf.matmul(x_data, A), b)

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(A))

# Declare loss function
# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
# L2 regularization parameter, alpha
alpha = tf.constant([0.01])
# Margin term in loss
classification_term = tf.reduce_mean(tf.maximum(0., tf.sub(1., tf.mul(model_output, y_target))))

# Put terms together
loss = tf.add(classification_term, tf.mul(alpha, l2_norm))


# Declare prediction function
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)


# Training loop
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(2000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)
    
    test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)
    
    if (i+1)%100==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))



plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()