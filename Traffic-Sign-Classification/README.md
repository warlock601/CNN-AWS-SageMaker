# TRAFFIC SIGN CLASSIFICATION

### STEPS

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

- Get the data & visualize it. Basically we will have all the data in: train.py (traning data), valid.py (validation data) & test.p (testing data).
```bash
import pickle

with open("train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)
```
