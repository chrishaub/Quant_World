import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

data = pd.read_csv('D:\\SMUMSDS\\QuantWorld\\CS5\\log2.csv')
data

data['Action'].value_counts()

# Compute the frequency of each color value and divide by the total count
# to get percentages
counts = data['Action'].value_counts(normalize=True) * 100

# Set Fig Size
fig, ax = plt.subplots(figsize=(7, 8))

# Create a bar plot of the result
ax = counts.plot(kind='bar', rot=0)

# Add labels and a title to the plot
ax.set_xlabel('Action')
ax.set_ylabel('Percentage')
ax.set_title('Percentage of Actions')

# Display the plot
plt.show()
print(counts)

data.memory_usage().sum()

for i in data.columns:
    if i in ['Bytes', 'Bytes Sent', 'Bytes Received', 'Packets',
             'Elapsed Time (sec)', 'pkts_sent', 'pkts_received']:
        data[i] = data[i].astype('uint32')

data.memory_usage().sum()

# Create Dummies

for i in ['Source Port', 'Destination Port',
          'NAT Source Port', 'NAT Destination Port']:
    data = data.join(pd.get_dummies(data[i], prefix=i, dtype=bool))

data

y = pd.factorize(data['Action'])
y

data = data.drop(['Source Port', 'Destination Port', 'NAT Source Port',
                  'NAT Destination Port', 'Action'], axis=1)

data.memory_usage()

# Remove Duplicate Columns After Dummy Creation

# Destination Port== Nat Destination Port
# filter columns that have 'Destination Port' in their name
dest_port_cols = data.filter(regex='Destination Port')

# Identify duplicate columns
dup_cols = dest_port_cols.columns[dest_port_cols.T.duplicated()].tolist()
# Combine the duplicates
dest_port_cols = dest_port_cols.T.drop_duplicates().T

# Print the list of dropped columns
dropped_cols = list(set(dup_cols) - set(dest_port_cols.columns))
# drop columns from the DataFrame
data = data.drop(columns=dropped_cols)
data.shape

# Source Port== Destination Port
# filter columns that start with 'Source Port' or 'Destination Port'
filtered_cols = data.filter(regex='^(Source Port|Destination Port)')
# Identify duplicate columns
dup_cols = filtered_cols.columns[filtered_cols.T.duplicated()].tolist()
# Combine the duplicates
filtered_cols = filtered_cols.T.drop_duplicates().T
# Print the list of dropped columns
dropped_cols = list(set(dup_cols) - set(filtered_cols.columns))
# drop columns from the DataFrame
data = data.drop(columns=dropped_cols)
data.shape

# NAT Source== NAT Destination
# filter columns that start with 'NAT Destination' or 'NAT Source'
filtered_cols = data.filter(regex='^(NAT Destination|NAT Source)')
# Identify duplicate columns
dup_cols = filtered_cols.columns[filtered_cols.T.duplicated()].tolist()
# Combine the duplicates
filtered_cols = filtered_cols.T.drop_duplicates().T
# Print the list of dropped columns
dropped_cols = list(set(dup_cols) - set(filtered_cols.columns))
# drop columns from the DataFrame
data = data.drop(columns=dropped_cols)
data.shape

# Save Transformed Data

# Save the DataFrame to a file
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)

# Save the Y to a file
with open('y.pickle', 'wb') as f:
    pickle.dump(y, f)

# Logistic Regression

# Create Sparse Data Frame
sparse_float_dtype = pd.SparseDtype("float", fill_value=0)
sparse_df = data.astype(sparse_float_dtype)
sparse_df.shape

# Save the sparse_df to a file
with open('sparse_df.pickle', 'wb') as f:
    pickle.dump(sparse_df, f)

sparse_df.memory_usage().sum()

# sparse_df.memory_usage().sum()

# Make sure that the same proportion of labels are present in each of the
#  train and test sets as the full df (Stratify)
x_train, x_test, y_train, y_test = train_test_split(
    sparse_df, y[0], stratify=y[0], test_size=0.2)

lr = LogisticRegression(n_jobs=4)
lr.fit(x_train, y_train)

preds = lr.predict(x_test)

# get the confusion matrix
cm = confusion_matrix(y_test, preds)

# display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y[1])
disp.plot()

print(classification_report(y_test, preds, target_names=y[1], zero_division=0))

# Linear SVC
# Train/Test Split Original Data 80/20

# Make sure that the same proportion of labels are present in each of the
# train and test sets as the full df (Stratify)
x_train, x_test, y_train, y_test = train_test_split(
    data, y[0], stratify=y[0], test_size=0.3)

linear = LinearSVC(dual=False, class_weight='balanced')
linear.fit(x_train, y_train)

preds = linear.predict(x_test)

# get the confusion matrix
cm = confusion_matrix(y_test, preds)

# display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y[1])
disp.plot()

print(classification_report(y_test, preds, target_names=y[1], zero_division=0))

# Load the saved DataFrame from the file
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
# Load the saved DataFrame from the file
with open('y.pickle', 'rb') as f:
    y = pickle.load(f)
# Load the saved DataFrame from the file
with open('sparse_df.pickle', 'rb') as f:
    sparse_df = pickle.load(f)

data

y

print(data.dtypes)

sparse_df

# SVC

# select the boolean columns for training
bool_cols = sparse_df.iloc[:, 7:].astype(int)
# select the non-boolean columns for training
non_bool_cols = sparse_df.iloc[:, :7].astype(float)

# split data into training and testing sets 60/40
X_train_bool, X_test_bool, y_train, y_test = train_test_split(
    bool_cols, y[0], test_size=0.4, stratify=y[0], random_state=42)
X_train_non_bool, X_test_non_bool, _, _ = train_test_split(
    non_bool_cols, y[0], test_size=0.4, stratify=y[0], random_state=42)

# create a StandardScaler object and fit on the non-boolean training set only
scaler = StandardScaler(with_mean=False)
scaler.fit(X_train_non_bool)

# transform the non-boolean training and testing sets using the scaler
X_train_non_bool_scaled = scaler.transform(X_train_non_bool)
X_test_non_bool_scaled = scaler.transform(X_test_non_bool)

X_train = hstack([X_train_non_bool_scaled, X_train_bool])

X_test = hstack([X_test_non_bool_scaled, X_test_bool])

X_test

svm = SVC(C=1.0, kernel='rbf', degree=2, gamma='scale', cache_size=500)

# train the classifier using the sample data and show a progress bar
for i in tqdm(range(100)):
    svm.fit(X_train, y_train)

preds = svm.predict(X_test)

# get the confusion matrix
cm = confusion_matrix(y_test, preds)

# display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y[1])
disp.plot()

print(classification_report(y_test, preds, target_names=y[1], zero_division=0))

svm = SVC(C=1., kernel='rbf', degree=2, gamma='scale',
          class_weight='balanced', cache_size=500)

# train the classifier using the sample data and show a progress bar
for i in tqdm(range(100)):
    svm.fit(X_train, y_train)

preds = svm.predict(X_test)

# get the confusion matrix
cm = confusion_matrix(y_test, preds)

# display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y[1])
disp.plot()

print(classification_report(y_test, preds, target_names=y[1], zero_division=0))


svm = SVC(C=.5, kernel='rbf', degree=2, gamma='scale',
          class_weight='balanced', cache_size=500)

# train the classifier using the sample data and show a progress bar
for i in tqdm(range(100)):
    svm.fit(X_train, y_train)

preds = svm.predict(X_test)

# get the confusion matrix
cm = confusion_matrix(y_test, preds)

# display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y[1])
disp.plot()

print(classification_report(y_test, preds, target_names=y[1], zero_division=0))

# Use Best Parameters on 80/20 Split

# split data into training and testing sets 80/20
X_train_bool, X_test_bool, y_train, y_test = train_test_split(
    bool_cols, y[0], test_size=0.2, stratify=y[0], random_state=42)
X_train_non_bool, X_test_non_bool, _, _ = train_test_split(
    non_bool_cols, y[0], test_size=0.2, stratify=y[0], random_state=42)

# create a StandardScaler object and fit on the non-boolean training set only
scaler = StandardScaler(with_mean=False)
scaler.fit(X_train_non_bool)

# transform the non-boolean training and testing sets using the scaler
X_train_non_bool_scaled = scaler.transform(X_train_non_bool)
X_test_non_bool_scaled = scaler.transform(X_test_non_bool)

# Join Scaled and Non Scaled Columns X_train
X_train = hstack([X_train_non_bool_scaled, csr_matrix(X_train_bool.values)])

# Join Scaled and Non Scaled Columns X_test
X_test = hstack([X_test_non_bool_scaled, X_test_bool])

svm = SVC(C=1., kernel='rbf', degree=2, gamma='scale',
          class_weight='balanced', cache_size=500)

# train the classifier using the sample data and show a progress bar
for i in tqdm(range(100)):
    svm.fit(X_train, y_train)

preds = svm.predict(X_test)

# get the confusion matrix
cm = confusion_matrix(y_test, preds)

# display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y[1])
disp.plot()

print(classification_report(y_test, preds, target_names=y[1], zero_division=0))

# Load the saved DataFrame from the file
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
# Load the saved DataFrame from the file
with open('y.pickle', 'rb') as f:
    y = pickle.load(f)
# Load the saved DataFrame from the file
with open('sparse_df.pickle', 'rb') as f:
    sparse_df = pickle.load(f)

# select the boolean columns for training
bool_cols = sparse_df.iloc[:, 7:].astype(int)
# select the non-boolean columns for training
non_bool_cols = sparse_df.iloc[:, :7].astype(float)

# split data into training and testing sets 60/40
X_train_bool, X_test_bool, y_train, y_test = train_test_split(
    bool_cols, y[0], test_size=0.2, stratify=y[0], random_state=42)
X_train_non_bool, X_test_non_bool, _, _ = train_test_split(
    non_bool_cols, y[0], test_size=0.2, stratify=y[0], random_state=42)

# create a StandardScaler object and fit on the non-boolean training set only
scaler = StandardScaler(with_mean=False)
scaler.fit(X_train_non_bool)

# transform the non-boolean training and testing sets using the scaler
X_train_non_bool_scaled = scaler.transform(X_train_non_bool)
X_test_non_bool_scaled = scaler.transform(X_test_non_bool)

X_train = hstack([X_train_non_bool_scaled, X_train_bool])

X_test = hstack([X_test_non_bool_scaled, X_test_bool])
X_test

# Save the Scaled DataFrame to a file
with open('X_train_scaled.80.pickle', 'wb') as f:
    pickle.dump(X_train, f)
with open('X_test_scaled.80.pickle', 'wb') as f:
    pickle.dump(X_test, f)
with open('y_train_scaled.80.pickle', 'wb') as f:
    pickle.dump(y_train, f)
with open('y_test_scaled.80.pickle', 'wb') as f:
    pickle.dump(y_test, f)

# SDG Classifier
# Create SGDClassifier object with partial_fit method
clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=42,
                    alpha=0.01)

# Compute class weights
classes = [0, 1, 2, 3]
weights = compute_class_weight('balanced', classes=classes, y=y_train)

# Train the model with multiple epochs using partial_fit
clf.partial_fit(X_train, y_train, classes=classes,
                sample_weight=weights[y_train])

preds = clf.predict(X_test)
# get the confusion matrix
cm = confusion_matrix(y_test, preds)

# display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y[1])
disp.plot()

print(classification_report(y_test, preds, target_names=y[1], zero_division=0))

# SDG Classifier Using Multiple Epochs and Data Shuffle
# Create SGDClassifier object with partial_fit method
clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=42,
                    alpha=.0001)

# Set number of epochs and shuffle data before each epoch
num_epochs = 10
for epoch in range(num_epochs):
    X_train, y_train = shuffle(X_train, y_train, random_state=epoch)
    clf.partial_fit(X_train, y_train, classes=classes,
                    sample_weight=weights[y_train])

preds = clf.predict(X_test)
# get the confusion matrix
cm = confusion_matrix(y_test, preds)

# display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y[1])
disp.plot()

print(classification_report(y_test, preds, target_names=y[1], zero_division=0))
