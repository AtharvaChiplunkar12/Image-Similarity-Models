import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Set device (GPU if available)
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Using device: {device_name}")

# Load CIFAR-10 dataset (only training data)
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

# Normalize the data
x_train = x_train / 255.0

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.Resizing(224, 224),  # Resize to VGG16 input size
    layers.RandomFlip("horizontal"),
    layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
])

# Prepare training dataset with data augmentation
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = (
    train_ds
    .shuffle(buffer_size=50000)
    .batch(8)
    .map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# Load the pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model layers

# Add a new classifier head for CIFAR-10 classes
x = layers.Flatten()(base_model.output)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
feature_output = layers.Dense(4096, activation='relu')(x)  # Feature extraction layer
output = layers.Dense(10, activation='softmax')(feature_output)  # CIFAR-10 has 10 classes

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
optimizer = SGD(learning_rate=0.001, momentum=0.9)
loss_fn = SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Train the model using only training data
epochs = 10

with tf.device(device_name):
    history = model.fit(
        train_ds,
        epochs=epochs
    )

# Save the model
os.makedirs('model', exist_ok=True)
model.save('model/vgg_finetuned_cifar10.h5')
print("Model saved to model/vgg_finetuned_cifar10.h5")
