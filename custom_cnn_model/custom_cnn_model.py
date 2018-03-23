# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import numpy as np
import random

# #training data
# training_filenames = ["/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_train_00000-of-00005.tfrecord",
#                       "/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_train_00001-of-00005.tfrecord",
#                       "/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_train_00002-of-00005.tfrecord",
#                       "/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_train_00003-of-00005.tfrecord",
#                       "/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_train_00004-of-00005.tfrecord"]
#
# #validation data
# validation_filenames = ["/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_validation_00000-of-00005.tfrecord",
#                         "/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_validation_00001-of-00005.tfrecord",
#                         "/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_validation_00002-of-00005.tfrecord",
#                         "/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_validation_00003-of-00005.tfrecord",
#                         "/home/hci/hyeon/git/FEM/custom_cnn_mdoel/dataset/facialExpression_validation_00004-of-00005.tfrecord"]
#


# image is 48 X 48 = 2304
x = tf.placeholder("float", shape=[None, 2304])
# label is angry, neural, surprised, sad, happy
y_ = tf.placeholder("float", shape=[None, 5])

# reshape by 48 x 48 and black & white 1channel
#
# training_dataset = tf.data.TFRecordDataset(x)
# validation_dataset = tf.data.TFRecordDataset(validation_filenames)

x_image = tf.reshape(x, [-1, 48, 48, 1])

print "x_image="
print x_image


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_input(self, file_path, batch_size, num_of_fake_img=0, shuffle=True, use_noise=True):
    with open(file_path) as csvfile:
        print("Parsing Data....")
        csvfile = csvfile.readlines()[1:-31]
        # csvfile = csvfile.readlines()[1:10]

        csvfile = np.array(csvfile)

        noise_alpha = 0.1

        if shuffle:
            np.random.shuffle(csvfile)

        for i in range(int(round(len(csvfile) / batch_size, 0))):
            y = []
            x = []
            purpose = []
            txt_list = csvfile[i * batch_size:i * batch_size + batch_size]
            for txt in txt_list:
                txt = txt.split(",")
                x.append([np.uint8(x_data) for x_data in txt[1].split()])
                y.append(txt[0])
                purpose.append([txt[2]])

                for j in range(num_of_fake_img):
                    choose_fake_img_type = random.randrange(0, 2)
                    x_rows = []
                    if choose_fake_img_type == 0:
                        noise = random.randrange(-8, 8) / 10

                        for x_data in txt[1].split():
                            x_data = int(x_data)
                            if noise > 0:
                                x_data = x_data + ((255 - x_data) * noise)
                            else:
                                x_data = x_data + ((x_data) * noise)
                            x_rows.append(np.uint8(x_data))
                        x.append(x_rows)
                        y.append(txt[0])
                        purpose.append(txt[2])

                    x_rows = []
                    if choose_fake_img_type == 1:
                        for x_data in txt[1].split():
                            x_data = int(x_data)
                            noise = random.randrange(int(-1 * x_data * noise_alpha),
                                                     int((256 - x_data) * noise_alpha))
                            x_data += noise
                            x_rows.append(np.uint8(x_data))

                        x.append(x_rows)
                        y.append(txt[0])
                        purpose.append(txt[2])

                for j in range(len(x)):
                    change_wjth = random.randrange(0, len(x))
                    x[j], x[change_wjth] = x[change_wjth], x[j]
                    y[j], y[change_wjth] = y[change_wjth], y[j]
                    purpose[j], purpose[change_wjth] = purpose[change_wjth], purpose[j]


    yield x, y, purpose

##first layer having 32 filters
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

##second layer having 64 filters
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##data serialization
W_fc1 = weight_variable([12 * 12 * 64, 2048])
b_fc1 = bias_variable([2048])

h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

##drop out layer
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([2048, 5])
b_fc2 = bias_variable([5])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(20000):
    with open('/home/hci/hyeon/git/FEM/custom_cnn_model/fer2013.csv') as csvfile:
        print("Parsing Data....")
        csvfile = csvfile.readlines()[1:-31]

        csvfile = np.array(csvfile)

batch = mnist.train.next_batch(50)
if i % 10 == 0:
    train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g" % (i, train_accuracy))
sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
