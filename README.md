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


### ReLU (Rectified Linear Units)
- ReLU layers are used to add non-linearity in the feature map.
- It also enhances the sparsity or how scattered the feature map is.
- The Gradient of the ReLU does not vanish as we increase x compared to the sigmoid function.
- What ReLU does is, if the input is negative, the output will be 0 and any value that is positive will pass as is.
- When we apply ReLU activation functions especially in the hidden layers, the performance of the network improves dramatically. 
<img width="1332" height="420" alt="image" src="https://github.com/user-attachments/assets/db8a4974-1938-4d1a-bb30-a34a0c6dcabb" />


### Pooling (Downsampling)
- Pooling or downsampling layers are placed after convolutional layers reduce feature map dimensionality.
- This improves the computational efficiency while preserving the features.
- Pooling helps the model to generalize by avoiding overfitting.
- If one of the pixel is shifted, the pooled feature map will be the same.
- Max pooling works by retaining the maximum feature response within a given sample size in a feature map.
- Stride indicates how many pixels it will be shifting to the right.
In the image below, it is taking out the maximum from 2 X 2 sub-matrices of tha original matrix and then it flattens them up.
<img width="1140" height="271" alt="image" src="https://github.com/user-attachments/assets/fced95e9-0d09-4e30-ad00-1c5a3167fa82" />
</br>
