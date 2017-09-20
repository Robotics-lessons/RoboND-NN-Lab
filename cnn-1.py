# %matplotlib inline
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from timeit import default_timer as timer
from datetime import timedelta

mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

learning_rate = 0.001
epochs = 4
batch_size = 128

test_valid_size = 256

n_classes = 10
dropout = 0.5
display_step = 100



cls_trues = np.array([label.argmax() for label in mnist.test.labels])


weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='Wc1'),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='Wc2'),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name='Wd1'),
    'out': tf.Variable(tf.random_normal([1024, n_classes]), name='Wout')}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name='Bc1'),
    'bc2': tf.Variable(tf.random_normal([64]), name='Bc2'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='Bd1'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='Bout')}

def conv2d(x, W, b, strides=1, name='Conv1'):
    with tf.name_scope(name):
    	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    	x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, name='Maxpool1'):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # layer 1: 28*28*1 to 14*14*32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # layer 2: 14*14*32 to 7*7*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name='Conv2')
    conv2 = maxpool2d(conv2, k=2, name='Maxpool2')
    with tf.name_scope('Fc1'):
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])    
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        with tf.name_scope('dropout'):
            fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def plot_images(images, cls_true, cls_pred=None):
#    print(len(images))
#    print(len(cls_true))
#    assert(len(images) == len(cls_true) == 9)
    k = 0
    err_no = len(images)  
    d =   err_no ** 0.5
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(math.ceil(d), math.floor(d))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
 
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape((28, 28)), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        k += 1
        if k >= err_no:
            break

    plt.show()

def plot_example_errors(session):
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_pred, y_pred_cls],
                                    feed_dict={
                        x: mnist.test.images[:test_valid_size],
                        y: mnist.test.labels[:test_valid_size], keep_prob: 1.0})

    # Negate the boolean array.
    incorrect = (correct == False)
#    print("Error number {}".format(len(incorrect)))  
#    print(incorrect)
#    incorrect = incorrect[0:test_valid_size]  
    # Get the images from the test-set that have been
    # incorrectly classified.
    err_images = mnist.test.images[:test_valid_size]
    images_f = err_images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true_1 = cls_trues[:test_valid_size]
    cls_true = cls_true_1[incorrect]
    # Get the true classes for those images.
    err_number = len(cls_pred)
    print("Error number {}".format(err_number))
#    print("cls_true {}".format(cls_true))

    # Plot the first 9 images.
    plot_images(images=images_f[0:err_number],
                cls_true=cls_true[0:err_number],
                cls_pred=cls_pred[0:err_number])



x = tf.placeholder(tf.float32, [None, 28, 28,1], name='inputData')
y = tf.placeholder(tf.float32, [None, n_classes], name='inputLabels')
keep_prob = tf.placeholder(tf.float32, name='dropout')

logits = conv_net(x, weights, biases, keep_prob)


    
with tf.name_scope('Model'):
    prediction = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(prediction, 1)

with tf.name_scope('Cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    with tf.name_scope('total'):
        cost = tf.reduce_mean(diff) 
           
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('AdamOptimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # Op to calculate every variable gradient
    grads = tf.gradients(cost, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Op to update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
    optimizer = optimizer.minimize(cost)

with tf.name_scope('Accuracy'):
    with tf.name_scope('correct_pred'):
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # correct_prediction = tf.equal(y_pred_cls, y_true_cls) = correct_pred
        # y_pred_cls = tf.argmax(y_pred, dimension=1) = tf.argmax(prediction, 1)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('cross_entropy', cost)
tf.summary.scalar('Accuracy', accuracy)

init = tf.global_variables_initializer()

# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# Summarize all gradients
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)
    
step = 0
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    start_time = timer()

#    train_tb = tf.summary.FileWriter('./logs/cnn-1', graph=tf.get_default_graph())
    train_tb = tf.summary.FileWriter('./logs/cnn-1')
    train_tb.add_graph(sess.graph)
#    print(mnist.train.num_examples//batch_size)
    for epoch in range(epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        epoch_time = timer()
        for batch in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size) 
            _, summary = sess.run([optimizer, merged], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout}) 
            step += 1 
            train_tb.add_summary(summary, step)
            if (batch + 1) % display_step == 0 or batch == 0:
#                print("epoch time: ", timedelta(seconds=(timer() - epoch_time)))
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}) 
                valid_acc = sess.run(accuracy, feed_dict={
                        x: mnist.validation.images[:test_valid_size], 
                        y: mnist.validation.labels[:test_valid_size], 
                        keep_prob: 1.0}) 
                print('Epoch {:>2}, Batch {:>3}, Step {:>5} - Loss: {:>10.4f}'
                  ' Validation Accuracy: {:.6f}'.format(epoch + 1, batch + 1, step , loss, valid_acc))
        print("epoch time: ", timedelta(seconds=(timer() - epoch_time)))
    test_acc = sess.run(accuracy, feed_dict={
                        x: mnist.test.images[:test_valid_size],
                        y: mnist.test.labels[:test_valid_size],
                        keep_prob: 1.0})
    print('Testing Accuracy: {}'.format(test_acc))
    print("total time: ", timedelta(seconds=(timer() - start_time)))
    print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs/cnn-1 " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")

#    plot_images(mnist.test.images[:9], cls_trues[0:9])
    plot_example_errors(sess)
