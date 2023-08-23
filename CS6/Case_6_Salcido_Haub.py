import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from time import time
from tensorflow.keras.regularizers import l2

# data = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS6\\all_train.csv")

data.head()

# Split Dataset into X and Y
x = data.iloc[:, 1:]
y = data.iloc[:, :1]


scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)

# Look at input Shape
scaled_data.shape


# Compute the frequency of each color value and divide by the total count to
# get percentages
counts = y.value_counts(normalize=True) * 100

# Set Fig Size
fig, ax = plt.subplots(figsize=(7, 8))

# Create a bar plot of the result
ax = counts.plot(kind='bar', rot=0)

# Add labels and a title to the plot
ax.set_xlabel('Label')
ax.set_ylabel('Percentage')
ax.set_title('Percentage of Label')


# Display the plot
plt.show()
print(counts)

# Reduce dataset to test parameters
reduced_x, large_x = train_test_split(
    scaled_data, test_size=0.7, random_state=42)
reduced_y, large_y = train_test_split(
    y, test_size=0.7, random_state=42)

# Save Reduced Dataset

# Save the DataFrame to a file
with open('reduced_x.pickle', 'wb') as f:
    pickle.dump(reduced_x, f)
with open('reduced_y.pickle', 'wb') as f:
    pickle.dump(reduced_y, f)

# Load the saved DataFrame from the file
with open('reduced_x.pickle', 'rb') as f:
    reduced_x = pickle.load(f)
# Load the saved DataFrame from the file
with open('reduced_y.pickle', 'rb') as f:
    reduced_y = pickle.load(f)

# Split Reduced Dataset 80/20
X_train, X_test, y_train, y_test = train_test_split(
    reduced_x, reduced_y, test_size=0.8, random_state=42)

# Sequential Models Knows in what order to connect layers sequentially
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28,)))
model.add(tf.keras.layers.Dense(100, activation='sigmoid'))
model.add(tf.keras.layers.Dense(50, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd', loss='BinaryCrossentropy', metrics=['accuracy'])

# Use train data to fit
model.fit(reduced_x, reduced_y, epochs=2, batch_size=10)

# Create New Layers
model.add(tf.keras.Input(shape=(28,)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])

# Use train data to fit
model.fit(reduced_x, reduced_y, epochs=2, batch_size=10)

# New Layers
model.add(tf.keras.Input(shape=(28,)))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Use train data to fit
model.fit(reduced_x, reduced_y, epochs=2, batch_size=15)

tb = TensorBoard(log_dir=f"logs\\{time()}")

safety = EarlyStopping(monitor='val_loss', patience=1)

# Add an Extra Layer
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28,)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit Model and Apply Early Stopping
model.fit(X_train, y_train, epochs=100,
          validation_data=(X_test, y_test),
          batch_size=15,
          callbacks=[tb, safety])

# Define Model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28,)))
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(40, activation='relu'))
model.add(tf.keras.layers.Dense(60, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit Model and Apply Early Stopping
model.fit(X_train, y_train, epochs=100,
          validation_data=(X_test, y_test),
          batch_size=15,
          callbacks=[tb, safety])

# Add an Extra Layer
model.add(tf.keras.Input(shape=(28,)))
model.add(tf.keras.layers.Dense(
    30, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    40, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    60, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    1, activation='sigmoid'))
model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Early Stopping Parameters
safety = EarlyStopping(monitor='val_loss', patience=2)

# Fit Model and Apply Early Stopping
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
          batch_size=15, callbacks=[tb, safety])

# Define Model Layers
model.add(tf.keras.Input(shape=(28,)))
model.add(tf.keras.layers.Dense(
    50, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    80, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    1, activation='sigmoid'))
model.compile(
    optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

# Fit Model and Apply Early Stopping
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
          batch_size=15, callbacks=[tb, safety])

# Fit Model On 80/20 Split

X_train, X_test, y_train, y_test = train_test_split(
    scaled_data, y, test_size=0.8, random_state=42)

X_train.shape

X_test.shape

tb = TensorBoard(log_dir=f"logs\\{time()}")

# Define Model Layers
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28,)))
model.add(tf.keras.layers.Dense(
    50, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    80, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(tf.keras.layers.Dense(
    1, activation='sigmoid'))
model.compile(
    optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

# Define Early Stopping Parameters
safety = EarlyStopping(monitor='val_loss', patience=1)

# Fit Model and Apply Early Stopping
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
          batch_size=15, callbacks=[tb, safety])

history = model.fit(X_train, y_train,
                    epochs=1000,
                    validation_data=(X_test, y_test),
                    batch_size=15,
                    callbacks=[tb, safety])

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
