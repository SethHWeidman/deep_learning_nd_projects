import tensorflow as tf


# Define a simple convolutional layer
def conv_layer(input, channels_in, channels_out):
    w = tf.Variable(tf.zeros([5, 5, channels_in, channels_out]))
    b = tf.Variable(tf.zeros([channels_out]))
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    return act

# And a fully connected layer
def fc_layer(input, channels_in, channels_out):
    w = tf.Variable(tf.zeros([channels_in, channels_out]))
    b = tf.Variable(tf.zeros([channels_out]))
    act = tf.nn.relu(tf.matmul(input, w) + b)
    return act

# Setup placeholders, and reshape the data
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

conv1 = conv_layer(x_image, 1, 32)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

conv2 = conv_layer(pool1, 32, 64)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])

fc1 = fc_layer(flattened, 7 * 7 * 64, 1024)
logits = fc_layer(fc1, 1024, 10)

# Compute cross entropy as our loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# Use an AdamOptimizer to train the network
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# compute the accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("./tmp/mnist_demo/1")
writer.add_graph(sess.graph)

# Import MNIST data
# Source: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Train for 2000 steps
for i in range(200):
    batch = mnist.train.next_batch(100)

    # Occasionally report accuracy
    if i % 20 == 0:
        [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    # Run the training step
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
