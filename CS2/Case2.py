import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# Import CSV
df = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS2\\diabetic_data.csv")

# View the head
df.head(20)

# Change Ids to Objects
df['admission_type_id'] = df['admission_type_id'].astype(object)
df['discharge_disposition_id'] = df['discharge_disposition_id'].astype(object)
df['admission_source_id'] = df['admission_source_id'].astype(object)

# Find Columns with one unique value
cols_with_one_value = df.columns[df.nunique() <= 1]
cols_with_one_value

# Drop columns that don't have more than 1 value
df = df.drop('examide', axis=1)
df = df.drop('citoglipton', axis=1)

# 'readmitted' is the dependent variable
df['readmitted'].unique()

# Stats on dependent variable
readmitted_counts = df['readmitted'].value_counts().to_frame(name='readmitted_counts')
readmitted_counts['percent'] = readmitted_counts['readmitted_counts']/df.shape[0]*100
readmitted_counts['percent'].plot.bar()
plt.title("Readmitted Percentages")
plt.show()

df['race'].unique()

# Replace question marks to NULL values
df.replace('?', np.nan, inplace=True)

null_counts = df.isnull().sum().to_frame(name='null_counts')
null_counts['percent_missing'] = null_counts['null_counts']/df.shape[0]*100
null_counts

# Drop variables with high percentage of null values
df = df.drop('weight', axis=1)
df = df.drop('medical_specialty', axis=1)
df = df.drop('payer_code', axis=1)

# Group by Patient # to Get Race Where Missing Race Can be Found on Different Records of the Same Patient
df1 = df[['patient_nbr', 'race']]
grouped = df1.groupby('patient_nbr').apply(lambda x: x.fillna(method='ffill'))
df['race'] = grouped['race']

# Distribution of race
race_counts = df['race'].value_counts()
race_counts.plot.bar()
plt.show()

# Stats on dependent variable
race_counts = df['race'].value_counts().to_frame(name='race_counts')
race_counts['percent'] = race_counts['race_counts']/df.shape[0]*100
race_counts['percent'].plot.bar()
plt.title("Race Percentages")
plt.show()

race_counts

# Impute missing values in race using mode
mode = df['race'].mode()[0]
df['race'] = df['race'].fillna(value=mode)

# Impute missing values in number_diagnoses where 2nd or 3rd diagnose was not made
df.loc[df['number_diagnoses'] == 1, 'diag_2'] = 0
df.loc[df['number_diagnoses'] == 1, 'diag_3'] = 0
df.loc[df['number_diagnoses'] == 2, 'diag_3'] = 0

# Find unique values under diag_1
df['diag_1'].unique()

# Use 3 digit ICD9
df['diag_1'] = df['diag_1'].str[:3]
df['diag_2'] = df['diag_2'].str[:3]
df['diag_3'] = df['diag_3'].str[:3]

# Stats on Dignosis 1
diag_1_counts = df['diag_1'].value_counts().to_frame(name='diag_1')
diag_1_counts['percent'] = diag_1_counts['diag_1']/df.shape[0]*100
diag_1_counts['percent']

# Distribution of Diagnosis_1
diag_1_counts['diag_1'].plot(kind='hist', edgecolor='black')
plt.title("Diagnosis_1 Percentages")

# Impute missing values in race using mode
mode = df['diag_1'].mode()[0]
df['diag_1'] = df['diag_1'].fillna(value=mode)

mode2 = df['diag_2'].mode()[0]
df['diag_2'] = df['diag_2'].fillna(value=mode2)

mode3 = df['diag_3'].mode()[0]
df['diag_3'] = df['diag_3'].fillna(value=mode3)

# One hot encoding
# Make dependent variable Binary
# Readmissions over 30 days are not counted as readmission
df['readmitted'] = np.where((df['readmitted'] == '>30') | (df['readmitted'] == 'NO'), 0 , df['readmitted'])
df['readmitted'] = df['readmitted'].replace('<30', 1)
readmitted_counts = df['readmitted'].value_counts().to_frame(name='readmitted')
readmitted_counts

categ_features = pd.concat([df.select_dtypes(exclude='int64')])
categ_features.dtypes

OneHotDF = pd.concat([pd.get_dummies(df[col], prefix=col, drop_first=True) for col in categ_features], axis=1)
OneHotDF.head()

# Code obtained from: https://github.com/jakemdrew/DataMiningNotebooks/blob/master/01.%20Pandas.ipynb
# Merged OneHotDF and df
mergedDF = pd.concat([df.select_dtypes(exclude='object'),OneHotDF],axis=1)

X = mergedDF.drop(columns=['encounter_id', 'patient_nbr', 'readmitted'], inplace=False)
y = mergedDF['readmitted']

# Scaler
scale = StandardScaler()
X = scale.fit_transform(X)

# Logistic regression
model = LogisticRegression()
modelCV = LogisticRegressionCV(cv=5)


def find_best(params, model, X, y):
    splitter = KFold(n_splits=5, shuffle=True, random_state=1776)
    best_score = -1E9
    for a in params:
        model.C = a
        score = cross_val_score(model, X, y, cv=splitter,
                                scoring='accuracy').mean()
        if score > best_score:
            best_score = score
            best_alpha = a
    return best_alpha, best_score


find_best(np.logspace(-4, 4, 10), model, X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression model
modelCV.fit(X_scaled, y)

# Predict class probabilities
# Fit logistic regression model
clf = modelCV
clf.fit(X_scaled, y)

probs = clf.predict_proba(X_scaled)

# Make predictions on the test set
y_pred = model.predict(X)

# Calculate accuracy of the model
acc = accuracy_score(y, y_pred)
