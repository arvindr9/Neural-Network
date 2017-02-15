###Screenshots of the output of the program can be found in the screenshots folder

####Refer to README-SETUP.md for instructions on downloading and running program

What is it?

A feed-forward neural network that takes inputs of word vectors created using n-gram analysis and returns outputs of vectors that correspond to languages.

Input:

The occurrences of each letter (unigram) in each word is counted, as are the occurrences of each pair of letters (bigram). Along with the 26 english alphabetic characters, there are an additional 28 characters that appear in other languages than english, like accented characters. The initial vector size is 2970, for the 54 possible single characters and the 2916 (54*54) possible pairs of characters. This is done through an embedding function.

Word Vector Cleaning:
Since many bigrams are never used in words, the program finds any indices for which every word vector in the training and testing sets have a value of zero and deletes them from the initial set of vectors, thereby reducing the size. For the english-italian classification system, the size of the vector after cleaning is 579. This increases the speed of the neural network.

Neural Network:
The neural network used is a single hidden layer that performs computations, consisting of one weight and bias for the hidden layer and one weight and bias for the output layer. Originally, these weights and biases are randomized, they are then adjusted by the optimizer to increase accuracy. The input vector is multiplied by the weight, and then the bias is added to it.

Training loop:
This is a feed-forward neural network, meaning as the neural network is running through the training data, a cost function calculates how inaccurate the network is. An optimizer than attempts to minimize that cost by changing the weights and biases. This continues until the training data is run through 50 times, as specified in the program. At the end, the accuracy is calculated by testing the weights and biases of the neural network on the testing data and seeing how accurate it was.

User Input:
Using the weights and biases determined by the optimizer, the program takes an input of words from the user and returns their language. 
