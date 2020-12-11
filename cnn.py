import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train.shape
X_test.shape
y_train.shape
y_train = y_train.reshape(-1,)
y_train[:5]
y_test = y_test.reshape(-1,)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# def print_sample(X, y, index):
#     print(X[index])
#     plt.figure(figsize = (15,2))
#     plt.imshow(X[index])
#     plt.xlabel(classes[y[index]])
X_train = X_train/255
y_train = y_train/255
# print_sample(X_train, y_train, 0)
# ann = models.Sequential([
#     layers.Flatten(input_shape=(32, 32, 3)),
#     layers.Dense(3000, activation='relu'),
#     layers.Dense(1000, activation='relu'),
#     layers.Dense(10, activation='sigmoid')
# ])

# ann.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# ann.fit(X_train, y_train, epochs=5)

# #ANN performed really bad on this dataset with 5 epochs! accuracy = 48.57% 
# y_pred = ann.predict(X_test)
# y_pred_classes = [np.argmax(element) for element in y_pred]
# print('classification report = \n', classification_report(y_test, y_pred_classes))


# CNN


# cnn = models.Sequential([
#     # CNN layers
#     layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3)),
#     layers.MaxPooling2D((2,2)),

#     layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),

#     # Dense network
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# cnn.fit(X_train, y_train, epochs=10)

# cnn.evaluate(X_test, y_test)
# y_test= y_test.reshape(-1,)
# print_sample(X_test, y_test,1)

# y_pred = cnn.predict(X_test)
# y_pred[:5]
# y_classes = [np.argmax(element) for element in y_pred]
# y_classes[:5]
# classes[y_classes[0]]


# tuning
def build_model(hp):  
  model = keras.Sequential([
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(28,28,1)
    ),
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    keras.layers.Dense(10, activation='softmax')
  ])
  
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model

tuner_search=RandomSearch(build_model, objective='val_accuracy', max_trials=5, directory='output', project_name='cifar-10')

# best param from this model (running 2 epochs)
tuner_search.search(X_train, y_train, epochs=3, validation_split=0.1)