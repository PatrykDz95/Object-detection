import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pickle
import time
from keras import backend as K
from keras import optimizers
from keras.layers.normalization import BatchNormalization



def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
NAME = "Dog-vs-Cat-{}.model".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

X = pickle.load(open("X_dogs.pickle","rb"))
y = pickle.load(open("y_dogs.pickle","rb"))

X = X/255.0

pickle_in = open("X_dogs.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_dogs.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(18, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(18, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(12))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(X, y, batch_size=80, epochs=1, validation_split=0.1, callbacks=[tensorboard])

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "examples", "my_model.tflite", as_text=False)

    # Save tf.keras model in HDF5 format.
# keras_file = "keras_model.h5"
# tf.keras.models.save_model(model, keras_file)
#
#     # Convert to TensorFlow Lite model.
#
# converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_file)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)
#
# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_content=tflite_model)
# interpreter.allocate_tensors()
# https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
