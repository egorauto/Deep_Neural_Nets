#!/usr/bin/env python
''''
TODO fill in your names:
group member 1: Georgy Antonov
group member 2: Andrei Ilic

'''
#####
#Updated version. The function accuracy was changed as is the way of computing the accuracy.
#The overall model stays the same
#####

#%%
#load the packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%
#don't change this
tf.random.set_seed(1234)


#load the MNIST dataset directly via tensorflow (no need to store it locally)
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

#normalize the data to work with number between 0 and 1
X_train=X_train/255
X_test=X_test/255



X_eval = X_train[0:10000, :, :]
Y_eval = Y_train[0:10000]
X_train = X_train[10000:, :, :]
Y_train = Y_train[10000:]

#X_train=np.concatenate((X_train,X_test))
#Y_train=np.concatenate((Y_train,Y_test))



#%%
#define a linear layer called dense_layer1 where weights[0] and weights[1] are instantiated below
#you should define more layers for parts b,c,d similar to this definition
def dense_layer1( inp , weights ):
    """
        :param input: dim(batch_size,784)
        :param weights: list of weight matrices and vectors as defined by shapes below
        :return: dim(batch_size,10)
    """    
    return tf.matmul( inp , weights[0] ) + weights[1]

def hidden_layer( inp , weights ):
    """
        :param input: dim(batch_size,784)
        :param weights: list of weight matrices and vectors as defined by shapes below
        :return: dim(batch_size,10)
    """    
    return tf.nn.relu(tf.matmul( inp , weights[0] ) + weights[1])

#%%
#define an initializer for the weights. Leave the initializer as is
initializer = tf.initializers.TruncatedNormal()

#function that defines a weight vector variable with given shape and name
def get_weight( shape , name ):
    """
        :param shape: tuple of numbers
        :param name: string that describes the name of the weight matrix/vector
        :return: tf.Variable of according shape
        """
    return tf.Variable( initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

#list of weight matrices and vectors.
#for the linear model we only have the 784x10 weight matrix and the 1x10 offset vector
#include the shapes of the weight matrices and vectors here as well
num_units1 = 500
num_units2 = 300
num_units3 = 100
shapes = {'dense1': [[num_units3, 10],[1,10]], 'hidden1': [[784, num_units1], [1, num_units1]], 
          'hidden2': [[num_units1, num_units2], [1, num_units2]], 
          'hidden3': [[num_units2, num_units3], [1, num_units3]]}

#save all the weight matrices and weight vectors centrally in the list weights
weights = {}
c = 0
for k, v in shapes.items():
        w = get_weight( v[ 0 ] , 'w')
        b = get_weight( v[ 1 ] , 'b')
        weights[k] = [w, b]

#define the model.
#for now it just contains dense_layer1
def model ( x ):
    """
        :param x: dim(batch_size,784)
        :return: dim(batch_size,10)
        """
    #calculate with float32 numbers
    x = tf.cast(x, dtype=tf.float32)
    #flatten the 28x28 image to an 784 vector. the '-1' means that
    #we leave the first dimension (batch dimension) unchosen and will be determined later by the batch size
    flatten = tf.reshape(x, [-1,784])
    h1 = hidden_layer(flatten, weights['hidden1'])
    h2 = hidden_layer(h1, weights['hidden2'])
    h3 = hidden_layer(h2, weights['hidden3'])
    d4 = dense_layer1(h3, weights['dense1'])
    return tf.nn.softmax( d4 )

#compute the loss
def loss( pred , target ):
    """
        :param pred: dim(batch_size,10)
        :param target: dim(batch_size,10)
        :return: tf.constant (just a number)
        """
    return tf.reduce_mean(tf.losses.categorical_crossentropy( target , pred ))

#compute the accuracy
def accuracy( pred , target ):
    """
        :param pred: dim(batch_size,10)
        :param target: dim(batch_size,10)
        :return: tf.constant (just a number)
            """
    return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(target,1)),tf.int32))/len(pred)


#define an optimizer. Here we use simply stochastic gradient descent as you are used to
#this removes the burden of updating the weights manually
optimizer = tf.optimizers.SGD( 0.001 )

#don't change the batch size
batch_size = 100


def train_step( model, inputs , outputs ):
    """
        :param model: the model we defined above
        :param inputs: dim(batch_size,784)
        :param outputs: dim(batch_size,10)
        :return: tf.constant: current_loss, current_acc
            """
    #Here we actually "remember" what the model is doing which we need for the optimizer to update the weights
    with tf.GradientTape() as tape:
        y=model(inputs)
        current_loss = loss(y, outputs)
    current_acc  = accuracy(y, outputs)
    grads = tape.gradient( current_loss , [w[0] for k, w in weights.items()] )
    optimizer.apply_gradients( zip( grads , [w[0] for k, w in weights.items()] ) )
    return current_loss, current_acc


num_epochs = 50

mean_eval_losses = []
mean_train_losses = []
mean_eval_acc = []
mean_train_acc = []

#%%
#start training over num_epochs
for e in range( num_epochs ):
    epoch_loss = 0
    epoch_acc  = 0
    for j in range(len(X_train)//batch_size):
        image = X_train[j:j+batch_size]
        label = Y_train[j:j+batch_size]
        #curent_loss,current_acc is not used in this updated version.
        #instead, after each epoch we compute the train loss on the whole train set
        current_loss, current_acc = train_step( model , image , tf.one_hot( label , depth=10 ) )

    #this is new
    epoch_acc = accuracy(model(X_train), tf.one_hot(Y_train, depth=10))
    epoch_loss = loss(model(X_train), tf.one_hot( Y_train , depth=10 ))

    val_acc = accuracy(model(X_eval), tf.one_hot(Y_eval, depth=10))
    val_loss = loss(model(X_eval), tf.one_hot( Y_eval , depth=10 ))

    mean_train_losses.append(epoch_loss)
    mean_train_acc.append(epoch_acc )
    mean_eval_losses.append(val_loss)
    mean_eval_acc.append(val_acc)



    tf.print("Epoch", e, "train loss was", epoch_loss )
    tf.print("Epoch", e, "train accuracy was", epoch_acc)

    tf.print("Epoch", e, "val loss was", val_loss)
    tf.print("Epoch", e, "val accuracy was", val_acc)



test_loss = loss(model(X_test), tf.one_hot( Y_test , depth =10 ))
test_acc  = accuracy(model(X_test), tf.one_hot(Y_test, depth=10))

tf.print("Test loss is", test_loss)
tf.print("Test accuracy is", test_acc)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(range(num_epochs), mean_train_losses, "r", label="train loss")
ax1.plot(range(num_epochs), mean_eval_losses, "b", label="eval loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()

ax2.plot(range(num_epochs), mean_train_acc, "r", label="train acc")
ax2.plot(range(num_epochs), mean_eval_acc, "b", label="eval acc")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.legend()
plt.show()
# fig.savefig('500hidden1300hidden2100hidden3_lr0001.png')

# %%
