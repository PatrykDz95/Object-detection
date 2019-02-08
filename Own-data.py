import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ["C0", "C2"]


def prepare(filepath):
    IMG_SIZE = 200# 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    plt.imshow(new_array)
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.



model = tf.keras.models.load_model("30-epochs-grey-binary-1543143408.model")

prediction = model.predict([prepare('examples\IMG_20181125_122416.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])

# https://pythonprogramming.net/using-trained-model-deep-learning-python-tensorflow-keras/

#
# import numpy as np
# from keras.preprocessing import image
# from keras.applications.inception_v3 import preprocess_input, decode_predictions
#
# model = tf.keras.models.load_model("Poker-hand-cnn-64x10-grey-1542984568.model")
#
# def predict(model, img_path, target_size=(300, 300), top_n=5):
#     img = image.load_img(img_path, target_size=target_size)
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     preds = model.predict(x)
#     return decode_predictions(preds, top=top_n)[0]
#
#
# pred = predict(model, CATEGORIES)
# # plt.imshow(fn)
# # plt.show()
# plot_pred(pred)