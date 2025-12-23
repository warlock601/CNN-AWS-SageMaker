# CNN-AWS-SageMaker
This repository showcases a project for Image Classification that uses Convolutional Neural Networks.

### Image Classifiers
Image Classifiers work by predicting the class of items that are present in a given image. For eg: we can train a classifier to classify images of cats and dogs. So when we feed a trained classifier an image of a dog, it can predict the label associated with the given image "label=dog".

### Project Overview
Traffic sign classification is an important task for self driving cars. In this project, a Convolutional Neural Network known as LeNet will be used for traffic sign images classification. The Dataset we're gonna use for this project consists of 43 different classes of images hence it is a multi-class classifier. Each image is og 32 X 32 pixels.

### Convolutional Neural Networks
The neuron collects signals from input channels named dendrites, processes information in its nucleus and then generates an output in a long thin branch called the axon. Human Learning occurs adaptively by varying the bond strength between these neurons.
- First the image is passed through a Convolutional layer basically they are Kernels/Feature Detectors (Filters that extract features out of images).
- Non-Linearity is added using ReLU (Rectified Linear Unit).
- Pooling is done using Pooling Filters to compresss all the feature maps. The size of the Feature Maps will get reduced when we apply pooling filters.
- Flattening is applied so input can be feed into the dense fully connected Artificial Neural Networks.
  </br>
  </br>
<img width="1430" height="543" alt="image" src="https://github.com/user-attachments/assets/7acf0448-7b2a-4bd6-94e0-979d8389b379" />
</br>
- Convolutions use a kernel matrix to scan a given image and apply a filter to obtain a certain effect.
- An image kernel is a matrix used to apply effects such as blurring and sharpening.
- Kernels are used in machine learning for feature extraction to select most important pixels of an image.
- Convolution preserves the spatial relationship between pixels.
