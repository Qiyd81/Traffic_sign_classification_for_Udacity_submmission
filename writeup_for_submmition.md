# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization.jpg "Visualization"
[image2]: ./speed_limit.jpg "example image"
[image3]: ./speed_limit_gray.jpg  "grayscaling image"
[image4]: ./14.jpg "Traffic Sign 1"
[image5]: ./28.jpg "Traffic Sign 2"
[image6]: ./27.jpg "Traffic Sign 3"
[image7]: ./26.jpg "Traffic Sign 4"
[image8]: ./31.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Qiyd81/Traffic_sign_classification_for_Udacity_submmission/Traffic_Sign_classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? (34799, 32, 32, 3)
* The size of the validation set is ? (4410, 32, 32, 3)
* The size of test set is ? (12630, 32, 32, 3)
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data looks like

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color image doesn't really affect the test results, and also computational resource consuming. And also I learned from Mr. Viadanna to use histogram equalization to improve the visibility of the sign. 

Here is an example of a traffic sign image before and after grayscaling.

![before][image2][after][image3]

As a last step, I normalized the image data because it can facilitate the convergence of the optimizer during training.

I didn't do the data augment here yet, as my macbook pro already can cook the eggs after the training on original data, but I tried others code, it works well:
"""I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image]

The difference between the original data set and the augmented data set is the following ... 
"""

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model(LeNet) consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 BGR image   							| 
| Convolution 3x3     	| 1x1 stride, "VALID" padding, outputs 28x28x12 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 3x3	    | 1x1 stride, "VALID" padding, outputs 10x10x24 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24 	     			|
| Fully connected		| flatten first, then outputs 256.   			|
| Fully connected		| outputs 128.              		   			|
| Fully connected		| outputs 43.   								|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an following major hyperparameters, loss, and optimizer:
# rate = 0.001
# EPOCHS = 15
# BATCH_SIZE = 128
# logits, conv1, conv2 = LeNet(x)
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
# loss_operation = tf.reduce_mean(cross_entropy)
# optimizer = tf.train.AdamOptimizer(learning_rate = rate)
# training_operation = optimizer.minimize(loss_operation)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
First, use grayscale, normalize to preprocess the image. As the training accuracy not so good, so added histogram equalization, and the results get better. 
Second, build the LeNet architecture, to train the model, and at the beginning, use only 5 epochs, and then increase to 20 epochs to increase the accuracy. 
My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.945 
* test set accuracy of 0.906


# may try following later:
"""
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
""" 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Stop sign][image4] ![Children crossing][image5] ![Pedestrains][image6] 
![Traffic Signals][image7] ![Wild animal crossing][image8]

The image7 might be difficult to classify because it has more color information which might affect the test.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Children Crossing		| Children Crossing								|
| Pedestrians			| Pedestrains (first time test) 	            |Right-of-Way(second time test) wondering why?
| Traffic Signals   	| General Caution				 				|
| Wild animals crossing	| Wild animals crossing 						|


The model was able to correctly guess 4 of the 5 traffic signs first time, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 90.6%. But when I do it again, it guess 3 of the 5, which gives accuracy of 60%, don't know why yet.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in cell 12.

The sign name = Children crossing
The Children crossing's top 5 softmax probabilities is 
{'Children crossing': 0.9999988079071045, 
 'End of speed limit (80km/h)': 8.554868600185728e-07, 
 'Speed limit (60km/h)': 1.6694242788162228e-07, 
 'Bicycles crossing': 8.684480690135388e-08, 
 'Beware of ice/snow': 3.407000903621338e-08}
The sign name = Stop
The Stop's top 5 softmax probabilities is 
{'Stop': 0.9639245271682739, 
 'Ahead only': 0.013333470560610294, 
 'Speed limit (20km/h)': 0.01261995267122984, 
 'Yield': 0.009365602396428585, 
 'Speed limit (60km/h)': 0.0002345522807445377}
The sign name = Pedestrians
The Pedestrians's top 5 softmax probabilities is 
{'Right-of-way at the next intersection': 0.8233627080917358, 
 'Pedestrians': 0.1766371726989746, 
 'General caution': 9.366412712097372e-08, 
 'Roundabout mandatory': 7.228810883219694e-09, 
 'Double curve': 9.83120043707153e-11}
The sign name = Wild animals crossing
The Wild animals crossing's top 5 softmax probabilities is 
{'Wild animals crossing': 0.9977922439575195, 
 'Double curve': 0.002207312034443021, 
 'Slippery road': 3.0659785466014e-07, 
 'Road narrows on the right': 1.1342833516891915e-07, 
 'Speed limit (30km/h)': 1.5437134237572536e-08}
The sign name = Traffic signals
The Traffic signals's top 5 softmax probabilities is 
{'General caution': 0.8723551630973816, 
 'Traffic signals': 0.12733180820941925, 
 'Right-of-way at the next intersection': 0.0002567098999861628, 
 'Pedestrians': 5.606142804026604e-05, 
 'Roundabout mandatory': 9.945258483412545e-08}

### (Optional) Visualizing the Neural Network (See line 333-366 for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
It uses the edges to make the classification. 


