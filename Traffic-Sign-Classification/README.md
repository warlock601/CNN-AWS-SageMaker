# TRAFFIC SIGN CLASSIFICATION

### Steps

- Open SageMaker Studio from AWS SageMaker AI, create Domain and then launch a JupyterLab instance and in the launcher use Python3 (ipykernel) notebook. Upload files that are uploaded in this folder in the JupyterLab instance.

- Problem statement
```bash
Our goal is to build a multiclassifier model based on deep learning to classify various traffic signs.

Dataset that we are using to train the model is German Traffic Sign Recognition Benchmark.

Dataset consists of 43 classes:

( 0, b'Speed limit (20km/h)') ( 1, b'Speed limit (30km/h)') ( 2, b'Speed limit (50km/h)') ( 3, b'Speed limit (60km/h)') ( 4, b'Speed limit (70km/h)')

( 5, b'Speed limit (80km/h)') ( 6, b'End of speed limit (80km/h)') ( 7, b'Speed limit (100km/h)') ( 8, b'Speed limit (120km/h)') ( 9, b'No passing')

(10, b'No passing for vehicles over 3.5 metric tons') (11, b'Right-of-way at the next intersection') (12, b'Priority road') (13, b'Yield') (14, b'Stop')

(15, b'No vehicles') (16, b'Vehicles over 3.5 metric tons prohibited') (17, b'No entry')

(18, b'General caution') (19, b'Dangerous curve to the left')

(20, b'Dangerous curve to the right') (21, b'Double curve')

(22, b'Bumpy road') (23, b'Slippery road')

(24, b'Road narrows on the right') (25, b'Road work')

(26, b'Traffic signals') (27, b'Pedestrians') (28, b'Children crossing')

(29, b'Bicycles crossing') (30, b'Beware of ice/snow')

(31, b'Wild animals crossing')

(32, b'End of all speed and passing limits') (33, b'Turn right ahead')

(34, b'Turn left ahead') (35, b'Ahead only') (36, b'Go straight or right')

(37, b'Go straight or left') (38, b'Keep right') (39, b'Keep left')

(40, b'Roundabout mandatory') (41, b'End of no passing')

(42, b'End of no passing by vehicles over 3.5 metric tons')

```


- Upload the required data: In SageMaker JupyterLab session, we'll need data to work with. That data is present in the repository in .zip format. Download all the files, keep them together in a folder and then extract       "traffic-sign-classification-data.zip". It will automatically consider other zipped folders and combine the data as it is .csv type.


- Get the data & visualize it. Basically we will have all the data in: train.py (traning data), valid.py (validation data) & test.p (testing data). We'll open these files and the mode will be "Read Binary". </br>
  The pickle module in Machine Learning is primarily used for serializing (saving) trained models, data transformers, and intermediate results to a file, and then deserializing (loading) them back into memory for later       use, such as making new predictions or resuming a workflow. 
```bash
import pickle

with open("train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)
```

- Obtain features and the labels. Within the train, valid and test wes imply have the features which are the images and the target labels that we're looking at. Using this we will have labeled data. </br>
  We're going to use the validation data set to perform cross-validation. </br>
  For every Epoch what we're going to do is basically run the validation data set to my model to compare the validation loss with my training loss and make sure both of them are going down. </br>
  Once we see that they both are kind of divorcing in a way like the training loss is basically going down and the validation is going up then this means that we have a problem and the model starts to learn all the ins       and outs of the training data and it fails to generalize.
```bash
X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```
- Then to check the shape. Similarly we can check shapes for X_train, y_train, y_test etc.
  ```bash
  X_test.shape                    
  ```


- Checking random sample of an image. We're gonna import numpy and matplotlib and then we'll select a random sample of an image. And then we'll select a random number between 1 & the lens of the testing data. And then print the actual corresponding label of that image. 
```bash
import numpy as np
import matplotlib.pyplot as plt
i = np.random.randint(1, len(X_test))
plt.imshow(X_test[i])
print('label = ', y_test[i])
```


- Import SageMaker/BOTO3 (AWS SDK for Python that allows us to deal with AWS services), Create a session, Define S3 and role.
```bash
# Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python
# Boto3 allows Python developer to write software that makes use of services like Amazon S3 and Amazon EC2

import sagemaker
import boto3

# Let's create a Sagemaker session
sagemaker_session = sagemaker.Session()

# Let's define the S3 bucket and prefix that we want to use in this session
bucket = 'sagemaker-practical' # bucket named 'sagemaker-practical' was created beforehand
prefix = 'traffic-sign-classifier' # prefix is the subfolder within the bucket.

# Let's get the execution role for the notebook instance. 
# This is the IAM role that you created when you created your notebook instance. You pass the role to the training job.
# Note that AWS Identity and Access Management (IAM) role that Amazon SageMaker can assume to perform tasks on your behalf (for example, reading training results, called model artifacts, from the S3 bucket and writing training results to Amazon S3). 
role = sagemaker.get_execution_role()
print(role)
```



- Upload the data to S3. First we will create a directory called "Data" and then we'll save the training and validation data in .npz format. Like training data will be stored as ./data/training.npz and validation data will be stored as ./data/validation.npz. Then upload the training and validation data to S3.
```bash
# Create directory to store the training and validation data

import os
os.makedirs("./data", exist_ok = True)

```

```bash
# Save several arrays into a single file in uncompressed .npz format
# Read more here: https://numpy.org/devdocs/reference/generated/numpy.savez.html

np.savez('./data/training', image = X_train, label = y_train)
np.savez('./data/validation', image = X_test, label = y_test)
```

```bash
# Upload the training and validation data to S3 bucket

prefix = 'traffic-sign'

training_input_path   = sagemaker_session.upload_data('data/training.npz', key_prefix = prefix + '/training')
validation_input_path = sagemaker_session.upload_data('data/validation.npz', key_prefix = prefix + '/validation')

print(training_input_path)
print(validation_input_path)
```
After the upload is complete, we can see the data in our local files as well as S3. </br>
<img width="1788" height="453" alt="image" src="https://github.com/user-attachments/assets/fe553cf1-0bf5-492f-81dc-bf3e5cebe9f7" />


