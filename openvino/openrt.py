import onnxruntime as ort
import numpy as np
import time
# from tensorflow import keras
import platform
import tensorflow as tf
from tensorflow.keras import datasets, layers, models # type: ignore 
import matplotlib.pyplot as plt
import numpy as np                                                                                                             
import onnx                                                                                                                      
import tf2onnx
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
'''data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(0.1),layers.RandomZoom(0.1),])
cnn = models.Sequential([data_augmentation,
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=10)
cnn.evaluate(X_test, y_test)
tf.saved_model.save(cnn, 'model')'''
#!python -m tf2onnx.convert --saved-model model --output model.onnx
from openvino import utils
utils.add_openvino_libs_to_path()
def to_numpy(tensor):
    return tensor.numpy() if hasattr(tensor, 'numpy') else tensor
model = onnx.load('model.onnx')
print(ort.get_available_providers())
# print(ort._version_)
x_test_onnx = to_numpy(X_test).astype(np.float32)   
#from tensorflow.keras import datasets,layers,models
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
ort_session = ort.InferenceSession('model.onnx',providers=['OpenVINOExecutionProvider'])

# (x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
# x_train=x_train/255
# x_test=x_test/255
# x_test_onnx = to_numpy(x_test).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: x_test_onnx}
t3=time.time()
ort_outs = ort_session.run(None, ort_inputs)
t4=time.time()
y_pred = np.argmax(ort_outs[0], axis=1)
print('Accuracy of model:', accuracy_score(y_test, y_pred))
print('F1 score of model:', f1_score(y_test, y_pred, average='weighted'))
print('Precision of model:', precision_score(y_test, y_pred, average='weighted'))
print('Recall of model:', recall_score(y_test, y_pred, average='weighted'))
onnx.checker.check_model(onnx_model)
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
ort_session = ort.InferenceSession('model.onnx',providers=['CPUExecutionProvider'])
def to_numpy(tensor):
    return tensor.numpy() if hasattr(tensor, 'numpy') else tensor
x_test_onnx = to_numpy(X_test).astype(np.float32)        
ort_inputs = {ort_session.get_inputs()[0].name: x_test_onnx}
t1=time.time()
ort_outs = ort_session.run(None, ort_inputs)
t2=time.time()
y_pred = np.argmax(ort_outs[0], axis=1)
print('Accuracy of model:', accuracy_score(y_test, y_pred))
print('F1 score of model:', f1_score(y_test, y_pred, average='weighted'))
print('Precision of model:', precision_score(y_test, y_pred, average='weighted'))
print('Recall of model:', recall_score(y_test, y_pred, average='weighted'))

print("cpu:")
print(t2-t1)
print("openvino")
print(t4-t3)