Project Title: Object Detection using R-CNN

Building and testing a Region Based Convolutional Neural Network from scratch to detect people in an image using
Pytorch on Google Colab.

First off, why did I build this project?
This project fulfills the requirement of completing a project in Computer Vision for the certification course in 
Data Science from Corpnce, Rajajinagar, Bangalore. But more importantly, this is my first major project and it
not only feeds my curiosity in Computer Vision, but also helps me get a deeper understanding of how Neural 
Networks can achieve this fascinating task of detecting objects in an image as me and my project partner built 
the model from the ground up including creating the training, validation and testing datasets, modifying the
resnet-18 network as per the need, training and testing the network and implementing non-max suppression.


Project Description:
Object Detection is a computer vision technique of recognizing and locating instances of objects in images.
A model built to achieve this task achieves this by putting a box around the object and this box is called
a bounding box. A bounding box is defined by 4 points viz., x and y co-ordinates of the top left corner of the box
and width and height (x, y, w, h). The objects can be of several classes and the Object Detection model not only 
recognizes and locates the object in the image also assigns a probability score to each of the bounding boxes 
indicating the presence of the object of a certain class.

![Bounding_box](https://user-images.githubusercontent.com/103943776/200113827-e6b4b460-7f61-4833-b80d-c087b99b40e9.png)

One of the first Object Detection models developed was Regional Convolutional Neural Network - RCNN. This project 
was built with the intention of gaining a deep understanding of the idea behind object detection and so instead of 
starting from more advanced object detection models like Fast-RCNN, Faster-RCNN and YOLO, it'd make more sense
to first work on more basic models like RCNN and then build from there.

The key idea in R-CNN is region proposals and this project uses Selective Search as the region proposal algorithm.

A brief on Selective Search Algorithm:
The algorithm generates sub-segmentations of input image and then combines these small sub-segmentations based on 
similarities in colour, texture, size and fill until the entire image becomes a region itself. The creation of initial regions by the algorithm is done using a 
graph-based greedy algorithm. The algorithm ends up producing thousands of bounding boxes of all scales and aspect ratios.
These regions may or may not contain the object.
Running this algorithm on an image in this project yielded around 2500 bounding boxes per image on an average.

![Selective_search_segmentation](https://user-images.githubusercontent.com/103943776/200113813-f3e6ce3d-95c4-4fda-a39c-c506d73a6379.png)

Each of these proposed bounding boxes in an image generated from the algorithm is passed through the resnet-18
network, whose classification layer is modified to output probability scores for only 2 classes viz., object and background, and has been 
trained on thousands of object and background images (explained in detail in the later sections).

After elimintating the bounding boxes which are classified as background, the rest of the bounding boxes (all of whom supposedly contain the 
object) are saved for further processing. The resnet-18 network has done its job of detecting objects in an image but there's a catch:
for an object in an image, there are going to be multiple bounding boxes proposed, and this is not the fault of the neural network.
The segmentation search produces multiple bounding boxes for an object and so the model classifies all of them as 'object'.

There must be a way to eliminate all the bounding boxes around an object but one, and that technique is called Non-Max Suppression (NMS).

A pre-requisite to understand Non-Max Suppression is Intersection-Over-Union (IOU). IOU is simply 
a measure of overlap of two bounding boxes. As the name suggests, IOU divides the intersection area of the two boxes 
by the union area. 

Non-Max Suppression, in simple terms, keeps the bounding box with highest score and eliminates the rest of them.
Among the list of proposed bounding boxes, NMS first chooses the bounding box with highest score and then calcluates IOU
with every other bounding box. If the IOU is greater than a preset threshold IOU then NMS eliminates this bounding box with lower score.
And then move to the next highest score, this process is repeated a number of times until there is only one bounding box 
around an object for all the objects in the image.

![A test image after passed through the developed Person Detection Model](https://user-images.githubusercontent.com/103943776/200114144-fda1d492-2315-4709-b407-c0df4ab04d76.png)

And voila! We are able to detect people in an image!


Contents Of The Repository:
This repository contains four .ipynb files and a .pt file, the contents and objective of each are explained in
detail next:

1. utils.ipynb
This file contains the function defined to calculate Intersection Over Union (IOU). The function simply accepts 
two bounding boxes as input and outputs the IOU.

2. Pre-processing.ipynb
Creating train, validation and test images for the model is all taken care of by this notebook.
After downloading around 420 images of class 'person' and the corresponding labels from coco-2017 dataset
through an app called fiftyone, the labels file (a json file) is filtered to extract only the neccessary 
data like the bounding box id, bounding box values, image id and the image path. 

![Ground Truth Bounding Box Dataframe](https://user-images.githubusercontent.com/103943776/200114185-a3426be2-27b7-4feb-93bc-e7bf0ff7ba5a.png)

The downloaded dataset is split into train, validation and test set. The training and validation sets are 
further processed to produce cropped images of the two classes: object and background. This is done 
by using selective search segmentation and the concept of IOU. An image from the training set, after 
passing through the segmentation algorithm, produces around 2500 region proposals. Amount of overlap
is calculated between the true bounding box and each of the region proposals using IOU. If the IOU is 
greater than 70%, this indicates that that region proposal most certainly encloses the object and so this 
proposed bounding box, which is nothing more than just a small image cropped from the training set, is saved into
the 'object' folder in the training directory. If the IOU is lesser than 30%, this means that the region proposal
most certainly does not enclose the object, meaning this is a background image and so this region proposal
is cropped from the image and saved inside the 'background' folder in the training directory. The same process is 
repeated for the validation set. The test set has no information about the true bounding boxes and so 
all the test images are saved as is into the test directory.

Thus, all the images needed to train a nueral network model for binary classification task (the two classes
being object and background) are now put together in Google Drive as this drive is to be mounted to Google Colab
notebook (which provides free GPUs for computational efficiency).

![Project Folder uploaded to Google Drive](https://user-images.githubusercontent.com/103943776/200114213-ec6406c2-5dba-4139-96c2-c3a2f6c8da05.png)

3. Resnet_model.ipynb
This is where a Convolutional Neural Network is trained to perform the classification task between two classes:
object and background. The training images and validation images, after passing through transforms, 
are subjected to Imbalanced Data Sampler function to account for the imbalanced dataset (more number of background
images in comparison to number of object images).

The CNN used for this project is resnet-18, whose final layer is modified as per the requirements. During training, 
the model and its parameters are saved when the validation loss is at its least, and this saved model plays 
a pivotal role in the project.

4. Person_detection.ipynb
Non-max Suppression, Intersection-Over-Union, trained resnet network, saved training, validation and test images, 
everything explained so far come together and work as a well-oiled machine here in this notebook.

After going through transforms, a test image is first passed through Selective Search Segmentation which yields 
a number of proposed bounding boxes. Each of these bounding boxes is passed through the trained resnet network
which returns probability scores. All those probability scores of the object class which are greater than the 
preset threshold (70%) are saved along with the bounding box values.

These bounding boxes are then subjected to non-max suppression iteratively until there is only one bounding box 
per object. Although it takes around 20 seconds to run the entire process on a test image (goes without saying, this is one
of the disadvantages of R-CNN models) but the model eventually prints the image and detects all the people in the 
image correctly.

![Model_result](https://user-images.githubusercontent.com/103943776/200114235-b37f5796-05af-433e-bd85-4e564f38c301.png)

Ideas for further work:
1. Deeper CNNs such as resnet-164 can be used inplace of resnet-18 and see by how much this impoves the performace.
Understandably, this would take lot longer to train and test, but it'd be fascinating to look at the results.
2. Improve upon the built object detection RCNN model to implement Fast-RCNN and Faster-RCNN.
3. Modify the project to accomodate more than 2 classes. Along with detecting people, the model can be trained to
detect vehicles, animals or whatever the aim of the project is.
4. Tweak the threshold IOU values when generating the training and validation images from the dataset and see 
if this brings a major imporvement in the CNN's performance.
5. Rewrite the code following OOPS concepts.

References:
1. https://medium.com/visionwizard/object-detection-4bf3edadf07f
2. https://medium.com/@selfouly/r-cnn-3a9beddfd55a
3. https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55
4. https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/
5. https://youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF
