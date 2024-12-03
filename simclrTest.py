from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

model_save_path = 'model/flower_classification_model.h5'

# Function to load and preprocess images for prediction
def load_and_preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32, 32))
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the class of an image
def predict_image_class(img_path):
    # Load and preprocess the image
    img = load_and_preprocess_image(img_path)
    
    # Load the saved model
    loaded_model = load_model(model_save_path)

    # Predict the class
    predictions = loaded_model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Example usage for testing with new images
test_images_path = ['yellow-daisy.jpg', 'flowers/dandelion/7355522_b66e5d3078_m.jpg']  # Add your test image paths here
for img_path in test_images_path:
    predicted_class = predict_image_class(img_path)
    print(f'Predicted class for {img_path}: {predicted_class}')

