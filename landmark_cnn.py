
import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
import tensorflow as tf
from numpy import array
import numpy

emotions = ["Angry", "Happy", "Neutral", "Surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = RandomForestClassifier(min_samples_leaf=20)
data = {}


def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
        fake_image = [1] * 784
        if self.one_hot:
            fake_label = [1] + [0] * 9
        else:
            fake_label = 0
        return [fake_image for _ in xrange(batch_size)], [
            fake_label for _ in xrange(batch_size)
        ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
        perm0 = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm0)
        self._images = self.images[perm0]
        self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
        # Finished epoch
        self._epochs_completed += 1
        # Get the rest examples in this epoch
        rest_num_examples = self._num_examples - start
        images_rest_part = self._images[start:self._num_examples]
        labels_rest_part = self._labels[start:self._num_examples]
        # Shuffle the data
        if shuffle:
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]
        # Start next epoch
        start = 0
        self._index_in_epoch = batch_size - rest_num_examples
        end = self._index_in_epoch
        images_new_part = self._images[start:end]
        labels_new_part = self._labels[start:end]
        return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate(
            (labels_rest_part, labels_new_part), axis=0)
    else:
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def get_files(emotion):
    print("./save/%s/*" % emotion)
    files = glob.glob("./save/%s/*" % emotion)
    print(len(files))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]

    return training, prediction


def get_landmarks(image):
    landmarks_vectorised = []
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(0, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            # meannp = np.asarray((ymean,xmean))
            # coornp = np.asarray((z,w))
            # dist = np.linalg.norm(coornp-meannp)
            # anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi)
            # landmarks_vectorised.append(dist)
            # landmarks_vectorised.append(anglerelative)
        for i in range(0, 8):
            landmarks_vectorised.append(0)
        data['landmarks_vectorised'] = landmarks_vectorised

    if len(detections) < 1:
        data['landmarks_vectorised'] = "error"


def make_sets():

    training_labels = []
    training_data = []
    prediction_labels = []
    prediction_data = []
    Angry = [1, 0, 0, 0]
    Happy = [0, 1, 0, 0]
    Neutral = [0, 0, 1, 0]
    Surprise = [0, 0, 0, 1]
    length = 0  # count of prediction file
    error_count = 0
    for emotion in emotions:
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                error_count = error_count + 1
            else:
                training_data.append(data['landmarks_vectorised'])
                if emotions.index(emotion) == 0:
                    training_labels.append(Angry)
                elif emotions.index(emotion) == 1:
                    training_labels.append(Happy)
                elif emotions.index(emotion) == 2:
                    training_labels.append(Neutral)
                else:
                    training_labels.append(Surprise)

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            length += 1
            if data['landmarks_vectorised'] == "error":
                error_count = error_count + 1
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
    #     npar_pred = np.array(prediction_data)
    #     pred_pro = clf.predict_proba(npar_pred)
    print(error_count)
    return training_data, training_labels, prediction_data, prediction_labels


training_data, training_labels, prediction_data, prediction_labels = make_sets()
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 144])  # variable to use input layer
y_ = tf.placeholder(tf.float32, [None, 4])  # variable to use output layer
W = tf.Variable(tf.zeros([144, 4]))  # 144*10 matrix
b = tf.Variable(tf.zeros([4]))  # 10 list
y = tf.nn.softmax(tf.matmul(x, W) + b)  # x*w+b


# make matrix size and return
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# make 0.1 and return wanna size
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # add padding to don't reduce matrix
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# layer1 - 12*12 matrix make 32ro using max pool
x_image = tf.reshape(x, [-1, 12, 12, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# layer2 - input is 14*14 matrix 32ro, output matrix 7*7 matrix 64ro
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# layer3 - last data is 7*7*64 = 3136 but using 1024
W_fc1 = weight_variable([3 * 3 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer - 1024 node make percentage 10ro(0~9) using soft max
W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train & save model
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

# traing_data0 = []
# training_data0 = training_data[0:50]
# print(training_data0)
start = 0
end = 50
dummy_data = []
temp = []
start=0
# for a in range(0,1000):
for idx in range(0,len(training_data)):
    dummy_data = training_data[idx]
    dummy_data.append(training_labels[idx])
    temp.append(dummy_data)
    dummy_data = []
    # start = start + 50
# dummy_data[0].append(training_data)
# dummy_data[1].append(training_labels)
#temp = np.asarray(dummy_data)
# print(temp[0])
random.shuffle(temp)
# print(temp[0])

batch = []
batch_image = []
batch_labels = []
for a in range(start, start + 50, 50):
    batch = temp[start: start+50]
    for b in range(0, 50):
        if temp[b] == None :
            break
        batch_image.append(temp[b][0:144])
        batch_labels.append(temp[b][144])
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_image, y_: batch_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (a, train_accuracy))
    # print(batch_image)
    # print(batch_labels)
    start = start + 50

# for i in range(1):
#     #batch = temp.next_batch(50)
#     if i % 10 == 0:
#     sess.run(feed_dict={x: dummy_data[0], y_: dummy_data[1], keep_prob: 0.5})
#  training_data,training_labels, prediction_data, prediction_labels
# print(training_data[0])
# print(training_data[10])
# print(training_data[20])
# print(training_data[0])
# print(len(training_data))
# print(len(training_labels))
#     print(training_labels)
# batch[0] 28*28 image, [1] is number tag, keep_prob : dropout percentage



save_path = saver.save(sess, "model2.ckpt")
print ("Model saved in file: ", save_path)

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: training_data, y_: training_labels, keep_prob: 1.0}))
sess.close()
