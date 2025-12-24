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


### How to improve CNNs performance?
- Improve accuracy by adding more feature detectors/filters or adding a dropout (eg: Instead of using 32, let say we use 64 filters).
- Dropout refers to dropping out units in a neural network.
- Neurons develop co-dependency amongst each other during training.
- Dropout is a regularization technique for reducing overfitting in neural netwroks.
- It enables training to occur on several architectures of the neural network.
  <img width="730" height="159" alt="image" src="https://github.com/user-attachments/assets/9c878395-e4fa-42e7-b467-90a00843fa1f" />


### Confusion Matrix
- It is a visual way of accessing the performance of the classifier.
- In Rows, we put all the prediction coming from my model and in Columns we do the true class.
- Tue Positives (TP): cases when classifier predicted TRUE (they have the disease) and correct class was TRUE (patient has disease).
- True Negatives (TN): cases when model predicted False (no disease), and correct class was FALSE (patient do not have disease).
- False Positives (FP) (Type 1 error): classifier predicted TRUE but correct class was FALSE (patient did not have disease). 
- Flase Negatives (FN) (Type 2 error): classifier predicted FALSE (patient do not have disease) but they actually do have the disease.
- False Positive is called Type-1 errror and False Negative is called Type-2 error.

### Key Performance Indicators (KPI)
- Classification Accurcy = (TP+TN)/(TP+TN+FP+FN)
- Misclassification rate (Error Rate) = (FP+FN)/(TP+TN+FP+FN)
- Precision = TP/Total TURE Predictions = TP/(TP+FP) (When model predicted TRUE class, how often was it right?)
- Recall = TP/Actual TRUE = TP/(TP+FN) (when the class was actually TURE, how often did the classifier get it right?)


### LENET Network
#### LeNet Architecture
<img width="927" height="312" alt="image" src="https://github.com/user-attachments/assets/d116420e-84cb-4c01-8a25-a457e838595e" />
