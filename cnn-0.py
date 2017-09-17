from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

learning_rate = 0.001
epochs = 1
batch_size = 128

test_valid_size = 256

n_classes = 10
dropout = 0.75
display_step = 100

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))}

def conv2d(x, W, b, strides=1, name='conv1'):
    with tf.name_scope(name):
    	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    	x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, name='maxpool1'):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # layer 1: 28*28*1 to 14*14*32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # layer 2: 14*14*32 to 7*7*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name='conv2')
    conv2 = maxpool2d(conv2, k=2, name='maxpool2')
    with tf.name_scope('fc1'):
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])    
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

x = tf.placeholder(tf.float32, [None, 28, 28,1], name='Input Data')
y = tf.placeholder(tf.float32, [None, n_classes], name='Input Labels')
keep_prob = tf.placeholder(tf.float32, name='dropout')

with tf.name_scope('Model'):
    logits = conv_net(x, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.name_scope('AdamOptimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    with tf.name_scope('correct_pred'):
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
history = [(0, None, 10)]
step = 0
merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(init)
    
    train_tb = tf.summary.FileWriter('./logs/cnn-1')
    train_tb.add_graph(sess.graph)
#    print(mnist.train.num_examples//batch_size)
    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size) 
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout}) 
            step += 1 
            if (batch + 1) % display_step == 0 or batch == 0:
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}) 
                valid_acc = sess.run(accuracy, feed_dict={
                        x: mnist.validation.images[:test_valid_size], 
                        y: mnist.validation.labels[:test_valid_size], 
                        keep_prob: 1.0}) 
                history.append((step, loss, valid_acc))
                print('Epoch {:>2}, Batch {:>3}, Step {:>5} - Loss: {:>10.4f}'
                  ' Validation Accuracy: {:.6f}'.format(epoch + 1, batch + 1, step , loss, valid_acc))

    test_acc = sess.run(accuracy, feed_dict={
                        x: mnist.test.images[:test_valid_size],
                        y: mnist.test.labels[:test_valid_size],
                        keep_prob: 1.0})
    print('Testing Accuracy: {}'.format(test_acc))
    

steps, loss, acc = zip(*history)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.title('Validation Loss / Accuracy')
ax_loss = fig.add_subplot(111)
ax_acc = ax_loss.twinx()
plt.xlabel('Training Steps')
plt.xlim(0, max(steps))

ax_loss.plot(steps, loss, '-o', color='r')
ax_loss.set_ylabel('Log Loss', color='r');
ax_loss.tick_params('y', colors='r')
ax_loss.set_ylim(1.0, 10000)

ax_acc.plot(steps, acc, '-o', color='b')
ax_acc.set_ylabel('Accuracy [%]', color='b');
ax_acc.tick_params('y', colors='b')
ax_acc.set_ylim(0.0,1.0)

plt.show()

