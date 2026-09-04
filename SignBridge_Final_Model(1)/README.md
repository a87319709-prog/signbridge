
# SignBridge Model

Model input: 63 hand-landmark features
Model output: 36 classes

Pipeline:

Image
-> MediaPipe Hands
-> 21 landmarks
-> x,y,z
-> 63 features
-> MinMaxScaler
-> SignBridge neural network
-> 36-class prediction

Test accuracy: 0.8994
Macro F1: 0.8719

IMPORTANT:
The SAME landmark order and fitted MinMaxScaler must be used
during application inference.

Do not refit the scaler in the application.
