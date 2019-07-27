# Load pickled data
import pickle
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image

# TODO: Fill this in based on where you saved the training and testing data

training_file = '../traffic-signs-data/train.p'
validation_file = '../traffic-signs-data/valid.p'
testing_file = '../traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
signnames = pd.read_csv('signnames.csv')['SignName']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TO)DO: Number of training examples
n_train = y_train.size

# TODO: Number of validation examples
n_validation = y_valid.size

# TODO: Number of testing examples.
n_test = y_test.size

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[1:].shape

# TODO: How many unique classes/labels there are in the dataset.
all_classes = np.unique(y_train)
n_classes = all_classes.size

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("y_test example =", y_test[1])
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("Shape of X_train =", X_train.shape)
print("Signname example =", signnames[0])
print("signname data type =", type(signnames))

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

# Visualizations will be shown in the notebook.
#%matplotlib inline

# # Randomly show n pictures
# n = 5
# for i in range(n):
#     index = np.random.choice(n_train)
#     image = X_train[index]
#     plt.imshow(image)
#     plt.show()

A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = A4_PORTRAIT[::-1]
n = 10  # Examples to display

for c in range(n_classes):  # Iterate all classes
    idx = np.where(y_train == c)  # Find index for class
    n_images = X_train[np.random.choice(idx[0], n)]  # Pick n random images to display
    f, axes = plt.subplots(1, n)
    f.set_size_inches(A4_LANDSCAPE)
    print(signnames[c])
    for i, image in enumerate(n_images):
        axes[i].imshow(image)
        axes[i].grid(False)
        axes[i].axis('off')
    plt.show()

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def equalize(image):
    return cv2.equalizeHist(image)

def normalize(image):
    min_v, max_v = np.min(image), np.max(image)
    return (image - min_v) / (max_v - min_v)*2 - 1

def preprocess(image):
    return np.expand_dims(normalize(equalize(grayscale(image))), axis=2)

def preprocess_set(dataset):
    dataset_new = []
    for image in dataset:
        image = preprocess(image)
        dataset_new.append(image)
    return np.array(dataset_new)

X_train_new = preprocess_set(X_train)
X_valid_new = preprocess_set(X_valid)
X_test_new = preprocess_set(X_test)
print("Shape of X_train_new =", X_train_new.shape)

# Randomly show k preprocessed pictures
k = 5
for i in range(k):
    index = np.random.choice(n_train)
    image = X_train_new[index][:,:,0]
    plt.imshow(image, cmap='gray')
    plt.show()

### Define your architecture here.
### Feel free to use as many code cells as needed.

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    #  Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x12.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 12), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(12))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    #  Activation.
    conv1 = tf.nn.relu(conv1)

    #  Pooling. Input = 28x28x12. Output = 14x14x12.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #  Layer 2: Convolutional. Output = 10x10x24.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 24), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(24))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    #  Activation.
    conv2 = tf.nn.relu(conv2)

    #  Pooling. Input = 10x10x24. Output = 5x5x24.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #  Flatten. Input = 5x5x24. Output = 600.
    fc0 = flatten(conv2)

    #  Layer 3: Fully Connected. Input = 600. Output = 256.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(600, 256), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(256))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    #  Activation.
    fc1 = tf.nn.relu(fc1)

    #  Layer 4: Fully Connected. Input = 256. Output = 128.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(256, 128), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(128))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    #  Activation.
    fc2 = tf.nn.relu(fc2)

    #  Layer 5: Fully Connected. Input = 128. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(128, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits, conv1, conv2

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
rate = 0.001
EPOCHS = 15
BATCH_SIZE = 128
logits, conv1, conv2 = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_shuffle, y_train_shuffle = shuffle(X_train_new, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_shuffle[offset:end], y_train_shuffle[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        train_accuracy = evaluate(X_train_new, y_train)
        validation_accuracy = evaluate(X_valid_new, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")


### Load the images and plot them here.
### Feel free to use as many code cells as needed.
### Calculate the accuracy for these 5 new images.
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

# # Randomly pick 5 pictures from test set
# l = 5
# X_test_own = []
# y_test_own = []
# for i in range(l):
#     index = np.random.choice(n_test)
#     image = X_test[index]
#     y_label = y_test[index]
#     X_test_own.append(image)
#     y_test_own.append(y_label)
#     plt.imshow(image)
#     plt.show()

# my own 5 images
import os
dir = os.listdir("../download_images")
print(dir)
X_test_own = []
y_test_own = [14,28,27,26,31]
for imgi in dir[1:]:
    image = Image.open("../download_images/"+imgi)
    image = np.array(image.resize((32,32), Image.ANTIALIAS))
    X_test_own.append(image)
    plt.imshow(image)
    plt.show()
X_test_own = np.array(X_test_own)
print("X_test_own shape=", X_test_own.shape)

# test on my own images
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    X_test_own = preprocess_set(X_test_own)
    test_accuracy = evaluate(X_test_own, y_test_own)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    softmaxes = sess.run(tf.nn.softmax(logits), feed_dict={x: X_test_own, y: y_test_own})
    print("output softmaxes shape =", softmaxes.shape)
    for j in range(len(X_test_own)):
        signname = signnames[y_test_own[j]]
        print("The sign name =", signname)
        softmax = softmaxes[j]
        # print("output softmax =", softmax)
        print("output softmax shape =", softmax.shape)
        top_softmaxes = []
        indexs = []
        for i in range(5):
            index = int(np.argmax(softmax))
            indexs.append(index)
            top_softmax = float(softmax[index])
            top_softmaxes.append(top_softmax)
            softmax[index] = 0
        print("The top 5 softmax probabilities for {} are signs".format(signname), signnames[indexs])
        print("The top 5 softmax probabilities for {} =".format(signname), top_softmaxes)

# ## Visualize your network's feature maps here.
# ## Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    index = np.random.choice(n_test)
    random_image = np.expand_dims(X_test_new[index], axis=0)
    print('Feature maps for', signnames[y_test[index]])
    plt.imshow(X_test_new[index][:,:,0])
    plt.show()
    print('First convolutional layer')
    outputFeatureMap(random_image, conv1, plt_num=1)
    plt.show()
    print('Second convolutional layer')
    outputFeatureMap(random_image, conv2, plt_num=2)
    plt.show()