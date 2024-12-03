import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

# Set device (GPU if available)
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Using device: {device_name}")

# Load CIFAR-10 dataset (only training data)
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

# Normalize the data
x_train = x_train / 255.0

# Data augmentation and preprocessing
data_augmentation = tf.keras.Sequential([
    layers.Resizing(224, 224),  # Resize to match ResNet50 input size
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
])

# Prepare training dataset with data augmentation
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = (
    train_ds
    .shuffle(buffer_size=50000)
    .batch(32)
    .map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# Load the pre-trained ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model layers

# Unfreeze the top layers of ResNet50
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Add a new classifier head for CIFAR-10 classes
x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
optimizer = Adam(learning_rate=0.0001)
loss_fn = SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Add early stopping
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

# Function to train the model
def train_model(model, train_ds, epochs=10):
    with tf.device(device_name):
        history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=[early_stopping]
        )
    return model, history

# Train and fine-tune the model
model, history = train_model(model, train_ds, epochs=10)

# Save the model
os.makedirs('model', exist_ok=True)
model.save('model/resnet_finetuned_cifar10.h5')
print("Model saved to model/resnet_finetuned_cifar10.h5")
