import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('model.keras')

def classify_image(filepath):
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    max_index = np.argmax(predictions[0])
    labels = ["Apple", "Orange", "Tomato"]
    predicted_label = labels[max_index]
    confidence = predictions[0][max_index]

    print(predicted_label)
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    classify_image(input("Enter the filepath of your image (example: img.jpg): "))
