import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("results/mask_model.keras")

image_path = input("Enter image path: ")

img = cv2.imread(image_path)
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Mask Not Detected")
else:
    print("Mask Detected")