from tensorflow.examples.tutorials.mnist import input_data
from datetime import timedelta
import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)
learning_rates = [0.01, 0.001, 0.0001, 0.00001] #, 0.000001]
epochs = [2, 4, 6, 8]
batch_sizes = [64, 128, 256]
#batch_sizes = [64, 128]

results = np.zeros(shape=[len(batch_sizes), len(learning_rates), len(epochs)])

test_valid_size = 256

n_classes = 10
dropouts = [0.75, 0.5]
display_step = 100

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='Wc1'),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='Wc2'),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name='Wd1'),
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

def make_hparam_str(l_rate, epoch, batch_size, dropout):
    return str(l_rate) + "," + str(epoch) + "," + str(batch_size) + "," + str(dropout)

x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='inputData')
y = tf.placeholder(tf.float32, [None, n_classes], name='inputLabels')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

keep_prob = tf.placeholder(tf.float32, name='dropout')

logits = conv_net(x, weights, biases, keep_prob)
    
with tf.name_scope('Model'):
    prediction = tf.nn.softmax(logits)

with tf.name_scope('Cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    with tf.name_scope('total'):
        cost = tf.reduce_mean(diff)
           
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('train_tb'):
    optimizer = tf.train.AdamOptimizer()
# Op to calculate every variable gradient 
    optimizer = optimizer.minimize(cost)

with tf.name_scope('Accuracy'):
    with tf.name_scope('correct_pred'):
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('cross_entropy', cost)
tf.summary.scalar('Accuracy', accuracy)

init = tf.global_variables_initializer()

# grads = tf.gradients(cost, tf.trainable_variables())
# grads = list(zip(grads, tf.trainable_variables()))
# # Op to update all variables according to their gradient
# apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
# # Create summaries to visualize weights
# for var in tf.trainable_variables():
#     tf.summary.histogram(var.name, var)
# # Summarize all gradients
# for grad, var in grads:
#     tf.summary.histogram(var.name + '/gradient', grad)


merged = tf.summary.merge_all()

def run_model(l_rate, epoch, hparam_str, batch_size, dropout):    
    step = 0
    
    with tf.Session() as sess:
        sess.run(init)
    
    #    train_tb = tf.summary.FileWriter('./logs/cnn-1', graph=tf.get_default_graph())
        train_tb = tf.summary.FileWriter('./tests/cnn1/' + hparam_str)
        train_tb.add_graph(sess.graph)
    #    print(mnist.train.num_examples//batch_size)

        for epoch in range(epoch):
            epoch_time = timer()
            total_batch = int(mnist.train.num_examples/batch_size)
            for batch in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, summary = sess.run([optimizer, merged], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout, learning_rate: l_rate}) 
                step += 1 
                train_tb.add_summary(summary, step)
                if (batch + 1) % display_step == 0 or batch == 0:
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}) 
                    valid_acc = sess.run(accuracy, feed_dict={
                            x: mnist.validation.images[:test_valid_size], 
                            y: mnist.validation.labels[:test_valid_size], 
                            keep_prob: 1.0}) 
                    print('Epoch {:>2}, Batch {:>3}, Step {:>5} - Loss: {:>10.6f}'
                      ' Validation Accuracy: {:.6f}'.format(epoch + 1, batch + 1, step , loss, valid_acc))
            print("epoch time: ", timedelta(seconds=(timer() - epoch_time)))

        test_acc = sess.run(accuracy, feed_dict={
                            x: mnist.test.images[:test_valid_size],
                            y: mnist.test.labels[:test_valid_size],
                            keep_prob: 1.0})
        print('Testing Accuracy: {}'.format(test_acc))
        print()
    return test_acc        

def print_result(dropout):
    print("Dropout: ", dropout)
    for i in range(len(batch_sizes)):
        print("======================================================================")
        print("Batch Size: {}".format(batch_sizes[i]))
        print("                      ", sep=' ', end='', flush=True)
        for k in range(len(epochs)):
            print(" Ecpoch: {}  ".format(epochs[k]), sep=' ', end='', flush=True)
        print(" Average ")  
        print("----------------------------------------------------------------------")  
        for j in range(len(learning_rates)):
        
            print("Learn Rate: {:6f}   ".format(learning_rates[j]), sep=' ', end='', flush=True)
            for k in range(len(epochs)):
                print(" {:2.4f}     ".format(results[i, j, k]), sep=' ', end='', flush=True)
            print(" {:2.4f}     ".format(sum(results[i, j, 0:])/len(epochs)))
        print("Average:               ", sep=' ', end='', flush=True)
        for k in range(len(epochs)):
            print(" {:2.4f}     ".format(sum(results[i, 0:, k])/len(learning_rates)), sep=' ', end='', flush=True)
        print()


#print_result()
#exit()

start_time = timer()

for dropout in dropouts:
    i = 0
    for batch_size in batch_sizes:
        k = 0   
        for epoch in epochs:
            j = 0
            for l_rate in learning_rates:
                hparam_str = make_hparam_str(l_rate, epoch, batch_size, dropout)
                print(hparam_str)
                results[i, j, k] = run_model(l_rate, epoch, hparam_str, batch_size, dropout)
            j += 1
            k += 1
        i += 1
    print("total time: ", timedelta(seconds=(timer() - start_time)))
    print_result(dropout)
    print()
    print("total time: ", timedelta(seconds=(timer() - start_time)))


print("Run the command line:\n" \
      "--> tensorboard --logdir=./tests/cnn1/ " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
