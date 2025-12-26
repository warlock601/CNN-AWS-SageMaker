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




- Train the CNN LeNet model using SageMaker.
```bash
The model consists of the following layers:

STEP 1: THE FIRST CONVOLUTIONAL LAYER #1
Input = 32x32x3
Output = 28x28x6
Output = (Input-filter+1)/Stride* => (32-5+1)/1=28
Used a 5x5 Filter with input depth of 3 and output depth of 6
Apply a RELU Activation function to the output
pooling for input, Input = 28x28x6 and Output = 14x14x6
Stride is the amount by which the kernel is shifted when the kernel is passed over the image.


STEP 2: THE SECOND CONVOLUTIONAL LAYER #2
Input = 14x14x6
Output = 10x10x16
Layer 2: Convolutional layer with Output = 10x10x16
Output = (Input-filter+1)/strides => 10 = 14-5+1/1
Apply a RELU Activation function to the output
Pooling with Input = 10x10x16 and Output = 5x5x16


STEP 3: FLATTENING THE NETWORK
Flatten the network with Input = 5x5x16 and Output = 400


STEP 4: FULLY CONNECTED LAYER
Layer 3: Fully Connected layer with Input = 400 and Output = 120
Apply a RELU Activation function to the output


STEP 5: ANOTHER FULLY CONNECTED LAYER
Layer 4: Fully Connected Layer with Input = 120 and Output = 84
Apply a RELU Activation function to the output


STEP 6: FULLY CONNECTED LAYER
Layer 5: Fully Connected layer with Input = 84 and Output = 43
```

Code Logic for building and training CNNs. 
```bash
import argparse, os
import numpy as np
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model


# The training code will be contained in a main gaurd (if __name__ == '__main__') so SageMaker will execute the code found in the main. 
# argparse: 
if __name__ == '__main__':
    
    # Parser to get the arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters are being sent as command-line arguments.
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    
    
    
    # The script receives environment variables in the training container instance. 
    # SM_NUM_GPUS: how many GPUs are available for trianing.
    # SM_MODEL_DIR: A string indicating output path where model artifcats will be sent out to.
    # SM_CHANNEL_TRAIN: path for the training channel 
    # SM_CHANNEL_VALIDATION: path for the validation channel

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()
    
    # Hyperparameters
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    # Loading the training and validation data from s3 bucket
    train_images = np.load(os.path.join(training_dir, 'training.npz'))['image']
    train_labels = np.load(os.path.join(training_dir, 'training.npz'))['label']
    test_images  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    test_labels  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

    K.set_image_data_format('channels_last')

    # Adding batch dimension to the input
    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
    test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)
    input_shape = (32, 32, 3)
    
    # Normalizing the data
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255
    test_images /= 255

    train_labels = tensorflow.keras.utils.to_categorical(train_labels, 43)
    test_labels = tensorflow.keras.utils.to_categorical(test_labels, 43)

    
    
    #LeNet Network Architecture
    
    model = Sequential()
    
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape= input_shape))
    
    model.add(AveragePooling2D())
    
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))   
    
    model.add(AveragePooling2D())
    
    model.add(Flatten())
    
    model.add(Dense(units=120, activation='relu'))
    
    model.add(Dense(units=84, activation='relu'))
    
    model.add(Dense(units=43, activation = 'softmax'))
    
    print(model.summary())

    
    # If more than one GPU is available, convert the model to multi-gpu model
    if gpu_count > 1:
        
        model = multi_gpu_model(model, gpus=gpu_count)

    # Compile and train the model
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=batch_size,
                  validation_data=(test_images, test_labels),
                  epochs=epochs,
                  verbose=2)

    # Evaluating the model
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])

    # save trained CNN Keras model to "model_dir" (path specificied earlier)
    sess = K.get_session()
    tensorflow.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})
```



- Then we need to specify the parameters using which TensorFlow will train the job.
```bash
from sagemaker.tensorflow import TensorFlow

# To Train a TensorFlow model, we will use TensorFlow estimator from the Sagemaker SDK

# entry_point: a script that will run in a container. This script will include model description and training. 
# role: a role that's obtained The role assigned to the running notebook. 
# train_instance_count: number of container instances used to train the model.
# train_instance_type: instance type!
# framwork_version: version of Tensorflow
# py_version: Python version.
# script_mode: allows for running script in the container. 
# hyperparameters: indicate the hyperparameters for the training job such as epochs and learning rate


tf_estimator = TensorFlow(entry_point='train-cnn.py', 
                          role=role,
                          train_instance_count=1, 
                          train_instance_type='ml.c5.xlarge',
                          framework_version='1.12', 
                          py_version='py3',
                          script_mode=True,
                          hyperparameters={
                              'epochs': 2 ,
                              'batch-size': 32,
                              'learning-rate': 0.001}
                         )

```


- Take the TensorFlow estimator, appl the fit method to it and feed it in or specify training path and validation path. 
```bash
tf_estimator.fit({'training': training_input_path, 'validation': validation_input_path})

```

- It will start the training job, will take some time and then it will give this kind of output:
```bash
2025-12-26 05:09:16 Starting - Starting the training job...
2025-12-26 05:09:43 Starting - Preparing the instances for training...
2025-12-26 05:10:20 Downloading - Downloading the training image..2025-12-26 05:10:29,652 sagemaker-containers INFO     Imported framework sagemaker_tensorflow_container.training
2025-12-26 05:10:29,658 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
2025-12-26 05:10:29,898 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
2025-12-26 05:10:29,912 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
2025-12-26 05:10:29,922 sagemaker-containers INFO     Invoking user script
Training Env:
{
    "additional_framework_parameters": {},
    "channel_input_dirs": {
        "training": "/opt/ml/input/data/training",
        "validation": "/opt/ml/input/data/validation"
    },
    "current_host": "algo-1",
    "framework_module": "sagemaker_tensorflow_container.training:main",
    "hosts": [
        "algo-1"
    ],
    "hyperparameters": {
        "batch-size": 32,
        "epochs": 2,
        "learning-rate": 0.001,
        "model_dir": "s3://sagemaker-eu-north-1-595512633933/sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764/model"
    },
    "input_config_dir": "/opt/ml/input/config",
    "input_data_config": {
        "training": {
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated",
            "RecordWrapperType": "None"
        },
        "validation": {
            "TrainingInputMode": "File",
            "S3DistributionType": "FullyReplicated",
            "RecordWrapperType": "None"
        }
    },
    "input_dir": "/opt/ml/input",
    "is_master": true,
    "job_name": "sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764",
    "log_level": 20,
    "master_hostname": "algo-1",
    "model_dir": "/opt/ml/model",
    "module_dir": "s3://sagemaker-eu-north-1-595512633933/sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764/source/sourcedir.tar.gz",
    "module_name": "train-cnn",
    "network_interface_name": "eth0",
    "num_cpus": 4,
    "num_gpus": 0,
    "output_data_dir": "/opt/ml/output/data",
    "output_dir": "/opt/ml/output",
    "output_intermediate_dir": "/opt/ml/output/intermediate",
    "resource_config": {
        "current_host": "algo-1",
        "current_instance_type": "ml.c5.xlarge",
        "current_group_name": "homogeneousCluster",
        "hosts": [
            "algo-1"
        ],
        "instance_groups": [
            {
                "instance_group_name": "homogeneousCluster",
                "instance_type": "ml.c5.xlarge",
                "hosts": [
                    "algo-1"
                ]
            }
        ],
        "network_interface_name": "eth0",
        "topology": null
    },
    "user_entry_point": "train-cnn.py"
}
Environment variables:
SM_HOSTS=["algo-1"]
SM_NETWORK_INTERFACE_NAME=eth0
SM_HPS={"batch-size":32,"epochs":2,"learning-rate":0.001,"model_dir":"s3://sagemaker-eu-north-1-595512633933/sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764/model"}
SM_USER_ENTRY_POINT=train-cnn.py
SM_FRAMEWORK_PARAMS={}
SM_RESOURCE_CONFIG={"current_group_name":"homogeneousCluster","current_host":"algo-1","current_instance_type":"ml.c5.xlarge","hosts":["algo-1"],"instance_groups":[{"hosts":["algo-1"],"instance_group_name":"homogeneousCluster","instance_type":"ml.c5.xlarge"}],"network_interface_name":"eth0","topology":null}
SM_INPUT_DATA_CONFIG={"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"validation":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}
SM_OUTPUT_DATA_DIR=/opt/ml/output/data
SM_CHANNELS=["training","validation"]
SM_CURRENT_HOST=algo-1
SM_MODULE_NAME=train-cnn
SM_LOG_LEVEL=20
SM_FRAMEWORK_MODULE=sagemaker_tensorflow_container.training:main
SM_INPUT_DIR=/opt/ml/input
SM_INPUT_CONFIG_DIR=/opt/ml/input/config
SM_OUTPUT_DIR=/opt/ml/output
SM_NUM_CPUS=4
SM_NUM_GPUS=0
SM_MODEL_DIR=/opt/ml/model
SM_MODULE_DIR=s3://sagemaker-eu-north-1-595512633933/sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764/source/sourcedir.tar.gz
SM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"training":"/opt/ml/input/data/training","validation":"/opt/ml/input/data/validation"},"current_host":"algo-1","framework_module":"sagemaker_tensorflow_container.training:main","hosts":["algo-1"],"hyperparameters":{"batch-size":32,"epochs":2,"learning-rate":0.001,"model_dir":"s3://sagemaker-eu-north-1-595512633933/sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764/model"},"input_config_dir":"/opt/ml/input/config","input_data_config":{"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"validation":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764","log_level":20,"master_hostname":"algo-1","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-eu-north-1-595512633933/sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764/source/sourcedir.tar.gz","module_name":"train-cnn","network_interface_name":"eth0","num_cpus":4,"num_gpus":0,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_group_name":"homogeneousCluster","current_host":"algo-1","current_instance_type":"ml.c5.xlarge","hosts":["algo-1"],"instance_groups":[{"hosts":["algo-1"],"instance_group_name":"homogeneousCluster","instance_type":"ml.c5.xlarge"}],"network_interface_name":"eth0","topology":null},"user_entry_point":"train-cnn.py"}
SM_USER_ARGS=["--batch-size","32","--epochs","2","--learning-rate","0.001","--model_dir","s3://sagemaker-eu-north-1-595512633933/sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764/model"]
SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
SM_CHANNEL_TRAINING=/opt/ml/input/data/training
SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
SM_HP_BATCH-SIZE=32
SM_HP_EPOCHS=2
SM_HP_LEARNING-RATE=0.001
SM_HP_MODEL_DIR=s3://sagemaker-eu-north-1-595512633933/sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764/model
PYTHONPATH=/opt/ml/code:/usr/local/bin:/usr/lib/python36.zip:/usr/lib/python3.6:/usr/lib/python3.6/lib-dynload:/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages
Invoking script with the following command:
/usr/bin/python train-cnn.py --batch-size 32 --epochs 2 --learning-rate 0.001 --model_dir s3://sagemaker-eu-north-1-595512633933/sagemaker-tensorflow-scriptmode-2025-12-26-05-09-15-764/model
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 6)         456       
_________________________________________________________________
average_pooling2d (AveragePo (None, 14, 14, 6)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416      
_________________________________________________________________
average_pooling2d_1 (Average (None, 5, 5, 16)          0         
_________________________________________________________________
flatten (Flatten)            (None, 400)               0         
_________________________________________________________________
dense (Dense)                (None, 120)               48120     
_________________________________________________________________
dense_1 (Dense)              (None, 84)                10164     
_________________________________________________________________
dense_2 (Dense)              (None, 43)                3655      
=================================================================
Total params: 64,811
Trainable params: 64,811
Non-trainable params: 0
_________________________________________________________________
None
Train on 34799 samples, validate on 12630 samples
Epoch 1/2

2025-12-26 05:10:25 Training - Training image download completed. Training in progress. - 12s - loss: 1.3942 - acc: 0.6125 - val_loss: 0.9386 - val_acc: 0.7523
Epoch 2/2
 - 11s - loss: 0.4131 - acc: 0.8796 - val_loss: 0.7712 - val_acc: 0.8184
Validation loss    : 0.7711984559268302
Validation accuracy: 0.8183689628247693
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/saved_model/simple_save.py:85: calling SavedModelBuilder.add_meta_graph_and_variables (from tensorflow.python.saved_model.builder_impl) with legacy_init_op is deprecated and will be removed in a future version.
Instructions for updating:
Pass your op to the equivalent parameter main_op instead.
2025-12-26 05:10:56,576 sagemaker-containers INFO     Reporting training SUCCESS

2025-12-26 05:11:14 Uploading - Uploading generated training model
2025-12-26 05:11:14 Completed - Training job completed
Training seconds: 73
Billable seconds: 73
```


- Deploy the Model. This will take sometime to deploy the endpoint.
```bash
# Deploying the model

import time

tf_endpoint_name = 'trafficsignclassifier-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

tf_predictor = tf_estimator.deploy(initial_instance_count = 1,
                         instance_type = 'ml.t2.medium',  
                         endpoint_name = tf_endpoint_name)
```

This exclamation mark at the end means that the model is now deployed.
<img width="868" height="122" alt="image" src="https://github.com/user-attachments/assets/6fae3d97-28fb-47b4-9f26-6ac83d584a7c" />


- Making predictions for the endpoint.
```bash
# Making predictions from the end point


%matplotlib inline
import random
import matplotlib.pyplot as plt

#Pre-processing the images

num_samples = 5
indices = random.sample(range(X_test.shape[0] - 1), num_samples)
images = X_test[indices]/255
labels = y_test[indices]

for i in range(num_samples):
    plt.subplot(1,num_samples,i+1)
    plt.imshow(images[i])
    plt.title(labels[i])
    plt.axis('off')

# Making predictions 

prediction = tf_predictor.predict(images.reshape(num_samples, 32, 32, 3))['predictions']
prediction = np.array(prediction)
predicted_label = prediction.argmax(axis=1)
print('Predicted labels are: {}'.format(predicted_label))

```

Output will be something like this: </br>

<img width="603" height="160" alt="image" src="https://github.com/user-attachments/assets/935753f1-5a1c-4eb7-97f0-a5ea1b58a92b" />


- Deleting the endpoint.
```bash
# Deleting the end-point
tf_predictor.delete_endpoint()

```
