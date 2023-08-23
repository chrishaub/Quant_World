import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from time import time

data = pd.read_csv('fp_data.csv')

data.head()

# Split Dataset into X and Y
x = data.iloc[:, :50]
y = data['y']

# Visualize Dependent Variable

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
ax.set_title('Percentage of Dependent Variable')

# Display the plot
plt.show()
print(counts)

# Explore Variables
x.dtypes

# Explore Categorical Variables
x['x24'].unique()

# Compute the frequency of each color value and divide by the total count to
# get percentages
counts = x['x24'].value_counts(normalize=True, dropna=False) * 100

# Set Fig Size
fig, ax = plt.subplots(figsize=(7, 8))

# Create a bar plot of the result
ax = counts.plot(kind='bar', rot=0)

# Add labels and a title to the plot
ax.set_xlabel('Label')
ax.set_ylabel('Percentage')
ax.set_title('Percentages')

# Display the plot
plt.show()
print(counts)

# Replace NaN values with mode
mode = x['x24'].mode().iloc[0]
x['x24'] = x['x24'].fillna(mode)

x['x29'].unique()

# Compute the frequency of each color value and divide by the total count to
# get percentages
counts = x['x29'].value_counts(normalize=True, dropna=False) * 100

# Set Fig Size
fig, ax = plt.subplots(figsize=(7, 8))

# Create a bar plot of the result
ax = counts.plot(kind='bar', rot=0)

# Add labels and a title to the plot
ax.set_xlabel('Label')
ax.set_ylabel('Percentage')
ax.set_title('Percentages')

# Display the plot
plt.show()
print(counts)

# Replace NaN values with mode
mode = x['x29'].mode().iloc[0]
x['x29'] = x['x29'].fillna(mode)

x['x30'].unique()

# Compute the frequency of each color value and divide by the total count to
# get percentages
counts = x['x30'].value_counts(normalize=True, dropna=False) * 100

# Set Fig Size
fig, ax = plt.subplots(figsize=(7, 8))

# Create a bar plot of the result
ax = counts.plot(kind='bar', rot=0)

# Add labels and a title to the plot
ax.set_xlabel('Label')
ax.set_ylabel('Percentage')
ax.set_title('Percentages')

# Display the plot
plt.show()
print(counts)

# Replace NaN values with mode
mode = x['x30'].mode().iloc[0]
x['x30'] = x['x30'].fillna(mode)

x['x32'].unique()

# Replace NaN values with mode
mode = x['x32'].mode().iloc[0]
x['x32'] = x['x32'].fillna(mode)

# 'x32' can be reformatted and used as Numerical
x['x32'] = x['x32'].str[:-1]
x['x32'] = x['x32'].astype(float)
x['x32'].head()

data['x37'].unique()

# 'x37' can be reformatted and used as Numerical
x['x37'] = x['x37'].str[1:]
x['x37'] = x['x37'].astype(float)
x['x37'].head()

# Create Dummy Variables on Categorical Variables
# Create dummy variables
x24 = pd.get_dummies(x['x24'])
x29 = pd.get_dummies(x['x29'])
x30 = pd.get_dummies(x['x30'])

# Drop categorical columns
x = x.drop(['x24', 'x29', 'x30'], axis=1)

# Combine Original df and Dummy Variables
bool_cols = pd.concat([x24, x29, x30], axis=1)
bool_cols.head()

x.isnull().sum()

# Fill null values with the mean of each column
x = x.fillna(x.mean())
x.isnull().sum()

# 80/20 Split
# Split the data into training and test sets
X_train_bool, X_test_bool, y_train, y_test = train_test_split(
    bool_cols, y, test_size=0.2, stratify=y, random_state=42)
X_train_non_bool, X_test_non_bool, _, _ = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42)

# create a StandardScaler object and fit on the non-boolean training set only
scaler = StandardScaler(with_mean=False)
scaler.fit(X_train_non_bool)

# transform the non-boolean training and testing sets using the scaler
X_train_non_bool_scaled = scaler.transform(X_train_non_bool)
X_test_non_bool_scaled = scaler.transform(X_test_non_bool)

# concatenate scaled non-boolean columns and boolean columns along axis 1
X_train = np.hstack((X_train_non_bool_scaled, X_train_bool))
X_test = np.hstack((X_test_non_bool_scaled, X_test_bool))

# SDG Classifier
# Compute class weights
classes = [0, 1]
weights = compute_class_weight('balanced', classes=classes, y=y_train)

# Create SGDClassifier object with partial_fit method
clf = SGDClassifier(
    loss='log', max_iter=1000, tol=1e-3, random_state=42, alpha=.0001)

# Set number of epochs and shuffle data before each epoch
num_epochs = 10
for epoch in range(num_epochs):
    X_train, y_train = shuffle(X_train, y_train, random_state=epoch)
    clf.partial_fit(
        X_train, y_train, classes=classes, sample_weight=weights[y_train])

# Define the class labels as a list
class_labels = ['0', '1']

# Predict the classes on test data
preds = clf.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, preds)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot()

print(classification_report(y_test, preds, zero_division=0))

# Neural Network Classifier
# Balance Response Variable
# calculate class weights based on frequency of each class in training data
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)

# convert class weights to dictionary
class_weight_dict = dict(enumerate(class_weights))

# concatenate the x_train and x_test arrays along the first axis (rows)
X = np.concatenate((X_train, X_test), axis=0)

# concatenate the y_train and y_test arrays along the first axis (rows)
y = np.concatenate((y_train, y_test), axis=0)

tb = TensorBoard(log_dir=f"logs\\{time()}")

# Define Model Layers
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(67,)))
model.add(tf.keras.layers.Dense(
    50, activation='relu', kernel_regularizer=l2(0.001)))
model.add(
    tf.keras.layers.Dense(80, activation='relu', kernel_regularizer=l2(0.001)
                          ))
model.add(
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(0.001)
                          ))
model.add(
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(0.001)
                          ))
model.add(
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(0.001)
                          ))
model.add(
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(0.001)
                          ))
model.add(
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(0.001)
                          ))
model.add(
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(0.001)
                          ))
model.add(
    tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(
    optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

# Define Early Stopping Parameters

safety = EarlyStopping(monitor='loss', patience=1)

# Fit Model and Apply Early Stopping
model.fit(
    X_train, y_train,
    epochs=1000,
    validation_data=(X_test, y_test),
    batch_size=15,
    callbacks=[tb, safety]
    )

# make predictions on the test set
y_pred = model.predict(X_test)
# convert the predictions to binary (0 or 1)
y_pred = np.round(y_pred).astype(int)

# Define the class labels as a list
class_labels = ['0', '1']

# Predict the classes on test data
preds = model.predict(X_test)
preds = np.round(y_pred).astype(int)

# Compute the confusion matrix
cm = confusion_matrix(y_test, preds)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot()

print(classification_report(y_test, preds, zero_division=0))

# define your Keras model

# define the number of folds
num_folds = 5

# define the cross-validation strategy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Loop over the folds
fold_scores = []
for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"Fold {fold+1}")
    # Split the data for this fold
    X_train_fold, y_train_fold = X[train_idx], y[train_idx]
    X_test_fold, y_test_fold = X[test_idx], y[test_idx]
    # Fit the model on the training data for this fold
    model.fit(X_train_fold, y_train_fold, epochs=1000, validation_data=(
        X_test_fold, y_test_fold), batch_size=15, callbacks=[tb, safety])
    # Evaluate the model on the test data for this fold
    scores = model.evaluate(X_test_fold, y_test_fold)
fold_scores.append(scores)
# Calculate the mean and standard deviation of the fold scores
fold_scores = np.array(fold_scores)

# define the number of folds
num_folds = 5

# define the cross-validation strategy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# create an empty list to store the confusion matrices
conf_matrices = []

# loop over the folds
for fold, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
    # get the training and validation data for this fold
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    # fit the model on the training data for this fold
model.fit(X_train, y_train,
          epochs=1000,
          batch_size=15,
          callbacks=[tb, safety],
          class_weight=class_weight_dict)
# make predictions on the validation data for this fold
y_pred = model.predict(X_val)
y_pred_classes = np.round(y_pred)
# calculate and print the confusion matrix for this fold
conf_matrix = confusion_matrix(y_val, y_pred_classes)
print(f"Confusion matrix for fold {fold+1}:")
print(conf_matrix)
# add the confusion matrix to the list
conf_matrices.append(conf_matrix)

conf_matrices

false_positives = [arr[0, 1:] for arr in conf_matrices]
sum_false_positives = sum(false_positives)
dollar_value_false_positives = sum_false_positives * 35
print("Sum of False Positives: " + str(sum_false_positives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_positives[0]))

false_negatives = [arr[1, :1] for arr in conf_matrices]
sum_false_negatives = sum(false_negatives)
dollar_value_false_negatives = sum_false_negatives * 15
print("Sum of False Negatives: " + str(sum_false_negatives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_negatives[0]))

Total = dollar_value_false_positives + dollar_value_false_negatives
print("Dollar Value of Incorrect Predictions: $" + str(Total[0]))

# Reduce False Positives by Increasing Threshold to 0.6
# False Positives have a higher dollar value than False Negatives

# define the number of folds
num_folds = 5

# define the cross-validation strategy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# create an empty list to store the confusion matrices
conf_matrices = []

# loop over the folds
for fold, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
    # get the training and validation data for this fold
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    # fit the model on the training data for this fold
    model.fit(X_train, y_train,
              epochs=1000,
              batch_size=15,
              callbacks=[tb, safety],
              class_weight=class_weight_dict)

    # make predictions on the validation data for this fold
    y_pred = model.predict(X_val)
    y_pred_classes = (y_pred > 0.6).astype(int)
    # calculate and print the confusion matrix for this fold
    conf_matrix = confusion_matrix(y_val, y_pred_classes)
    print(f"Confusion matrix for fold {fold+1}:")
    print(conf_matrix)
    # add the confusion matrix to the list
    conf_matrices.append(conf_matrix)

false_positives = [arr[0, 1:] for arr in conf_matrices]
sum_false_positives = sum(false_positives)
dollar_value_false_positives = sum_false_positives * 35
print("Sum of False Positives: " + str(sum_false_positives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_positives[0]))

false_negatives = [arr[1, :1] for arr in conf_matrices]
sum_false_negatives = sum(false_negatives)
dollar_value_false_negatives = sum_false_negatives * 15
print("Sum of False Negatives: " + str(sum_false_negatives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_negatives[0]))

Total = dollar_value_false_positives + dollar_value_false_negatives
print("Dollar Value of Incorrect Predictions: $" + str(Total[0]))

# define the number of folds
num_folds = 5

# define the cross-validation strategy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# create an empty list to store the confusion matrices
conf_matrices = []

# loop over the folds
for fold, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
    # get the training and validation data for this fold
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    # fit the model on the training data for this fold
    model.fit(X_train, y_train,
              epochs=1000,
              batch_size=15,
              callbacks=[tb, safety],
              class_weight=class_weight_dict)
    # make predictions on the validation data for this fold
    y_pred = model.predict(X_val)
    y_pred_classes = (y_pred > 0.3).astype(int)
    # calculate and print the confusion matrix for this fold
    conf_matrix = confusion_matrix(y_val, y_pred_classes)
    print(f"Confusion matrix for fold {fold+1}:")
    print(conf_matrix)
    # add the confusion matrix to the list
    conf_matrices.append(conf_matrix)

false_positives = [arr[0, 1:] for arr in conf_matrices]
sum_false_positives = sum(false_positives)
dollar_value_false_positives = sum_false_positives * 35
print("Sum of False Positives: " + str(sum_false_positives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_positives[0]))

false_negatives = [arr[1, :1] for arr in conf_matrices]
sum_false_negatives = sum(false_negatives)
dollar_value_false_negatives = sum_false_negatives * 15
print("Sum of False Negatives: " + str(sum_false_negatives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_negatives[0]))

Total = dollar_value_false_positives + dollar_value_false_negatives
print("Dollar Value of Incorrect Predictions: $" + str(Total[0]))

# define the number of folds
num_folds = 5

# define the cross-validation strategy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# create an empty list to store the confusion matrices
conf_matrices = []

# loop over the folds
for fold, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
    # get the training and validation data for this fold
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    # fit the model on the training data for this fold
    model.fit(X_train, y_train,
              epochs=1000,
              batch_size=15,
              callbacks=[tb, safety],
              class_weight=class_weight_dict)
    # make predictions on the validation data for this fold
    y_pred = model.predict(X_val)
    y_pred_classes = (y_pred > 0.4).astype(int)
    # calculate and print the confusion matrix for this fold
    conf_matrix = confusion_matrix(y_val, y_pred_classes)
    print(f"Confusion matrix for fold {fold+1}:")
    print(conf_matrix)
    # add the confusion matrix to the list
    conf_matrices.append(conf_matrix)

conf_matrices

false_positives = [arr[0, 1:] for arr in conf_matrices]
sum_false_positives = sum(false_positives)
dollar_value_false_positives = sum_false_positives * 35
print("Sum of False Positives: " + str(sum_false_positives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_positives[0]))

false_negatives = [arr[1, :1] for arr in conf_matrices]
sum_false_negatives = sum(false_negatives)
dollar_value_false_negatives = sum_false_negatives * 15
print("Sum of False Negatives: " + str(sum_false_negatives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_negatives[0]))

Total = dollar_value_false_positives + dollar_value_false_negatives
print("Dollar Value of Incorrect Predictions: $" + str(Total[0]))

# Define Early Stopping Parameters

safety = EarlyStopping(monitor='accuracy', patience=1)

# define the number of folds
num_folds = 5

# define the cross-validation strategy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# create an empty list to store the confusion matrices
conf_matrices = []

# loop over the folds
for fold, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
    # get the training and validation data for this fold
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    # fit the model on the training data for this fold
    model.fit(X_train, y_train,
              epochs=1000,
              batch_size=15,
              callbacks=[tb, safety],
              class_weight=class_weight_dict)
    # make predictions on the validation data for this fold
    y_pred = model.predict(X_val)
    y_pred_classes = np.round(y_pred)
    # calculate and print the confusion matrix for this fold
    conf_matrix = confusion_matrix(y_val, y_pred_classes)
    print(f"Confusion matrix for fold {fold+1}:")
    print(conf_matrix)
    # add the confusion matrix to the list
    conf_matrices.append(conf_matrix)

conf_matrices

false_positives = [arr[0, 1:] for arr in conf_matrices]
sum_false_positives = sum(false_positives)
dollar_value_false_positives = sum_false_positives * 35
print("Sum of False Positives: " + str(sum_false_positives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_positives[0]))

false_negatives = [arr[1, :1] for arr in conf_matrices]
sum_false_negatives = sum(false_negatives)
dollar_value_false_negatives = sum_false_negatives * 15
print("Sum of False Negatives: " + str(sum_false_negatives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_negatives[0]))

Total = dollar_value_false_positives + dollar_value_false_negatives
print("Dollar Value of Incorrect Predictions: $" + str(Total[0]))

# define the number of folds
num_folds = 5

# define the cross-validation strategy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# create an empty list to store the confusion matrices
conf_matrices = []

# loop over the folds
for fold, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
    # get the training and validation data for this fold
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    # fit the model on the training data for this fold
    model.fit(X_train, y_train,
              epochs=1000,
              batch_size=15,
              callbacks=[tb, safety],
              class_weight=class_weight_dict)
    # make predictions on the validation data for this fold
    y_pred = model.predict(X_val)
    y_pred_classes = (y_pred > 0.7).astype(int)
    # calculate and print the confusion matrix for this fold
    conf_matrix = confusion_matrix(y_val, y_pred_classes)
    print(f"Confusion matrix for fold {fold+1}:")
    print(conf_matrix)
    # add the confusion matrix to the list
    conf_matrices.append(conf_matrix)

conf_matrices

# Sum the arrays element-wise to get a single 2 by 2 array
cm = np.sum(conf_matrices, axis=0)

# Define the class labels as a list
class_labels = ['0', '1']

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)

false_positives = [arr[0, 1:] for arr in conf_matrices]
sum_false_positives = sum(false_positives)
dollar_value_false_positives = sum_false_positives * 35
print("Sum of False Positives: " + str(sum_false_positives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_positives[0]))

false_negatives = [arr[1, :1] for arr in conf_matrices]
sum_false_negatives = sum(false_negatives)
dollar_value_false_negatives = sum_false_negatives * 15
print("Sum of False Negatives: " + str(sum_false_negatives[0]))
print("Dollar Value of False Positives: $"
      + str(dollar_value_false_negatives[0]))

Total = dollar_value_false_positives + dollar_value_false_negatives
print("Dollar Value of Incorrect Predictions: $" + str(Total[0]))
