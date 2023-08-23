import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBClassifier
from matplotlib.pyplot import figure
import numpy as np
# import graphviz


loc = '/Users/alonsosalcido/Desktop/Quantifying the World/case 4/data'

year1 = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS4\\1year.arff",
                    skiprows=69, header=None, na_values='?')
year2 = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS4\\2year.arff",
                    skiprows=69, header=None, na_values='?')
year3 = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS4\\3year.arff",
                    skiprows=69, header=None, na_values='?')
year4 = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS4\\4year.arff",
                    skiprows=69, header=None, na_values='?')
year5 = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS4\\5year.arff",
                    skiprows=69, header=None, na_values='?')

data_labels = pd.read_csv(
    'https://raw.githubusercontent.com/AlonsoSalcido/Quantifying-the-World/main/Data_Labels.csv'
)
count = 1
for i in [year1, year2, year3, year4, year5]:
    i['year'] = count
    count += 1

# Concatenate Tables
data = pd.concat(
    [year1, year2, year3, year4, year5],
    axis=0,
    ignore_index=True
)
data.columns = data.columns.astype(str)

# Column 64 should be Boolean
data['64'] = data['64'].astype('bool')

# Missing Values
# Calculate the percentage of null values in each column
null_percentages = data.isnull().sum() / len(data) * 100

# Plot the null value percentages
plt.figure(figsize=(20, 10))
plt.bar(null_percentages.index, null_percentages.values)
plt.xticks(rotation=90)
plt.xlabel('Column')
plt.ylabel('Percentage of null values')
plt.title('Percentage of null values in the dataset')
plt.show()

print(data_labels.head(60))

# Print Data Label

print(data_labels.iloc[36]['Description'])

# Print summary statistics for column 36
print(data['36'].describe())

train_data = data[data['36'].notnull()]
train_data_x = train_data.iloc[:, :64]
train_data_x = train_data_x.drop(train_data_x.columns[36], axis=1)

# Impute missing data in column '36

# Create KNN imputer object
imputer = KNNImputer(n_neighbors=4)

# Impute missing values in column '36'
imputed_y = imputer.fit_transform(data[['36']])

# Merge imputed '36' values with original dataframe
merged_df = pd.concat(
    [train_data_x, pd.DataFrame(imputed_y, columns=['36'])],
    axis=1
)
data['36'] = merged_df['36']


# Print summary statistics for column 36
print(data['36'].describe())

dcols = data.iloc[:, :64]
dcols = dcols.drop(dcols.columns[36], axis=1)
cols_to_use = dcols.columns

# create KNN imputer object
imputer = KNNImputer(n_neighbors=4)

# impute missing values using KNN imputer
data[cols_to_use] = imputer.fit_transform(data[cols_to_use])

# Check For Null Values After Imputation
# Count the total null values in the dataset
total_null_count = data.isnull().sum().sum()

print("Data Shape:", data.shape)
print("Total null values in the dataset:", total_null_count)

rf = RandomForestClassifier(n_jobs=10)
splits = KFold(n_splits=5, shuffle=True)
cross_val_scores = cross_val_score(
    rf,
    data.iloc[:, 0:64],
    data.iloc[:, 64],
    cv=splits,
    scoring='roc_auc'
)

print(cross_val_scores)

# Grid Search

# time

parameters = {
    'class_weight': ['balanced'],
    'criterion': ['entropy'],
    'max_depth': [15, 20, 30],
    'min_samples_split': [4, 8, 16],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['log2', 'sqrt'],
    'n_estimators': [100, 200]
}
search = RandomizedSearchCV(
    rf,
    parameters,
    scoring='roc_auc',
    cv=splits,
    n_iter=20
)

outcomes = search.fit(data.iloc[:, 0:64], data.iloc[:, 64],)

outcomes.cv_results_

outcomes.best_score_

outcomes.best_params_

# XGBoost
Xtrain, Xtest, ytrain, ytest = train_test_split(
    data.iloc[:, 0:64],
    data.iloc[:, 64],
    test_size=0.20
)


train_dm = xgb.DMatrix(Xtrain, label=ytrain)
eval_dm = xgb.DMatrix(Xtest, label=ytest)

params = {
          'max_depth': 30,
          'eval_metric': 'auc',
          'objective': 'binary:logistic',
          'num_boost_round': 100
         }

xgb_scores = xgb.train(
    params,
    evals=[(train_dm, 'train'), (eval_dm, 'test')],
    dtrain=train_dm,
    verbose_eval=True,
    num_boost_round=params['num_boost_round']
)


# time
stuff = xgb.cv(params, dtrain=train_dm, nfold=5,
               stratified=False, metrics='auc',
               num_boost_round=params['num_boost_round'], verbose_eval=True)

stuff

plt.rcParams["figure.figsize"] = (10, 10)
plt.plot(stuff['train-auc-mean'], label='Training AUC')
plt.plot(stuff['test-auc-mean'], label='Test AUC')
plt.legend()

xgb.plot_importance(xgb_scores)

plt.rcParams["figure.figsize"] = (80, 10)

figure(figsize=(8, 6), dpi=80)
xgb.plot_tree(xgb_scores, num_trees=75)

my_model = XGBClassifier(
    subsample=0.8,
    max_depth=30,
    eval_metric='auc',
    objective='binary:logistic',
    n_estimators=1000,
    verbosity=1,
    early_stopping_rounds=10)

my_model.fit(Xtrain, ytrain, eval_set=[(Xtest, ytest)])

x = my_model.get_booster()

x.best_iteration

# Cross Validation Scores
# time
cv_scores = cross_val_score(
    my_model,
    data.iloc[:, 0:64],
    data.iloc[:, 64],
    cv=5,
    scoring='roc_auc'
)

cv_scores

mean = np.mean(cv_scores)
std_dev = np.std(cv_scores)
print(mean)
print(std_dev)
