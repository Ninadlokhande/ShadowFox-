import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD AND PREPARE DATA
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. BUILD THE CNN MODEL
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10)) # 10 output nodes for 10 classes

# Print a summary of the model
model.summary()

# 3. COMPILE AND TRAIN THE MODEL
print("\n--- Starting Model Training ---")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
print("--- Model Training Complete ---\n")


# 4. TEST THE MODEL ON A SINGLE IMAGE
print("--- Making a Prediction ---")
# Select a random image from the test set
test_image_index = np.random.randint(0, len(test_images))
img_to_predict = test_images[test_image_index]
true_label = class_names[test_labels[test_image_index][0]]

# The model expects a "batch" of images, so we add a dimension
img_batch = np.expand_dims(img_to_predict, 0)

# Make the prediction
prediction = model.predict(img_batch)
predicted_label_index = np.argmax(prediction)
predicted_label = class_names[predicted_label_index]

# Display the result
plt.imshow(img_to_predict)
plt.title(f"Actual name: {true_label}\nPredicted name : {predicted_label}")
plt.axis('off')
plt.show()

print(f"model predicted '{predicted_label}' and actual label '{true_label}'.")