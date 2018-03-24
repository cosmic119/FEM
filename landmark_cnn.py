
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

def get_files(emotion):                                     #load images and shuffles
    print("./save/%s/*" % emotion)
    files = glob.glob("./save/%s/*" % emotion)
    print(len(files))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]                        #80% of images is training
    prediction = files[-int(len(files) * 0.2):]                     #20% of images is prediction

    return training, prediction


def get_landmarks(image):                                           # make relative position about central of face
    landmarks_vectorised = []
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(0, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)                                      # center of x
        ymean = np.mean(ylist)                                      # center of y
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)                          # relative position plus list of x
            landmarks_vectorised.append(y)                          # relative position plus list of x
        for i in range(0, 8):                                        #zero padding
            landmarks_vectorised.append(0)
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:                                         #if not found face on image
        data['landmarks_vectorised'] = "error"


def make_sets():                                       #make landmarks of image(*_data) & labels(*_labels
    training_labels = []
    training_data = []
    prediction_labels = []
    prediction_data = []
    Angry = [1, 0, 0, 0]                                            #labels
    Happy = [0, 1, 0, 0]                                            #labels
    Neutral = [0, 0, 1, 0]                                          #labels
    Surprise = [0, 0, 0, 1]                                         #labels
    error_count = 0                                                  #can't find face count
    for emotion in emotions:                                        #find file to save file
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":             #can't find face
                error_count = error_count + 1
            else:
                training_data.append(data['landmarks_vectorised'])  #plus landmarks(144) save to training_data
                if emotions.index(emotion) == 0:                    #save labels to training_labels
                    training_labels.append(Angry)
                elif emotions.index(emotion) == 1:
                    training_labels.append(Happy)
                elif emotions.index(emotion) == 2:
                    training_labels.append(Neutral)
                else:
                    training_labels.append(Surprise)

        for item in prediction:                                     #plus landmarks(144) save to prediction_data
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                error_count = error_count + 1
            else:
                prediction_data.append(data['landmarks_vectorised'])
                if emotions.index(emotion) == 0:                    #save labels to training_labels
                    prediction_labels.append(Angry)
                elif emotions.index(emotion) == 1:
                    prediction_labels.append(Happy)
                elif emotions.index(emotion) == 2:
                    prediction_labels.append(Neutral)
                else:
                    prediction_labels.append(Surprise)
    print(error_count)
    return training_data, training_labels, prediction_data, prediction_labels


training_data, training_labels, prediction_data, prediction_labels = make_sets()
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 144])  # variable to use input layer
y_ = tf.placeholder(tf.float32, [None, 4])  # variable to use output layer
W = tf.Variable(tf.zeros([144, 4]))  # 144*4 matrix
b = tf.Variable(tf.zeros([4]))  # 4 list
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

# layer2 - input is 28*28 matrix 32ro, output matrix 7*7 matrix 64ro
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

dummy_data = []
temp = []
for idx in range(0,len(training_data)):   # training_data[0~143] save to temp[0~143], labels save temp[144]
    dummy_data = training_data[idx]
    dummy_data.append(training_labels[idx])
    temp.append(dummy_data)
    dummy_data = []
random.shuffle(temp)

batch = []
batch_image = []
batch_labels = []
start = 0
for a in range(start, start + 50, 50):   #repeat length of images
    batch = temp[start: start+50]
    for b in range(0, 50):    #batch of 50 images
        if temp[b] == None :
            break
        batch_image.append(temp[b][0:144]) # x,y landmark total is 144 plus batch_image
        batch_labels.append(temp[b][144]) # [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] of 1 is list[144] plus
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_image, y_: batch_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (a, train_accuracy))
    start = start + 50

save_path = saver.save(sess, "model2.ckpt")
print ("Model saved in file: ", save_path)

#print("test accuracy %g" % accuracy.eval(feed_dict={
#    x: training_data, y_: training_labels, keep_prob: 1.0}))
sess.close()

