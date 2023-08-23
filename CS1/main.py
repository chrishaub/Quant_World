import pandas as pd
import matplotlib.pyplot as plt
# import argparse
import numpy as np
from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# # Read in CSV's
# df1 = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS1\\train.csv")
# df2 = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS1\\unique_m.csv").drop(columns=['critical_temp', 'material'], axis=1 )

# # Joining the data frames
# merged = df1.join(df2, lsuffix='_left', rsuffix='_right')

# # View the head
# print(merged.head())

# # Null values
# print(merged.isnull().sum())

# # Print first item in df
# print(merged.loc[0])

def find_alpha(alphas, model_, X, y):
    Cs = []
    scores = []
    for i in range(100):
        seed = np.random.randint(0, 100000)
        split = KFold(n_splits=3, shuffle=True, random_state=seed)
        best = -1E9
        best_a = -1
        for a in alphas:
            model_.alpha = a
            tmp = cross_val_score(model_, X, y, cv=split,
                                  scoring='neg_mean_absolute_error')
            scores.append(-tmp.mean())
            Cs.append(a)
            if tmp.mean() > best:
                best = tmp.mean()
                best_a = a
    plt.scatter(Cs, scores)
    plt.xscale("log")
    plt.title("Regularization Strength Effect On Mean Squared Error")
    plt.xlabel("Regularization Strength, $\lambda$")
    plt.ylabel("Mean Squared Error")
    plt.show()
    return best, best_a


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("datafile",
    #                     help="path to data file")
    # args = parser.parse_args()
    # data = pd.read_csv(args.datafile)
    # Read in CSV's
    df1 = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS1\\train.csv")
    df2 = pd.read_csv("D:\\SMUMSDS\\QuantWorld\\CS1\\unique_m.csv").drop(columns=['critical_temp', 'material'], axis=1)

    # Joining the data frames
    data = df1.join(df2, lsuffix='_left', rsuffix='_right')
    scale = StandardScaler()
    scaled_data = scale.fit_transform(data)
    model = Ridge()
    candidates = np.logspace(-8, 1, num=20)
    good_a = find_alpha(candidates, model, scaled_data[:, :-1],
                        scaled_data[:, -1])
    _, model.alpha = good_a
    all_scores = []
    for i in range(100):
        splits = KFold(n_splits=5, shuffle=True, random_state=i)
        scores = cross_val_score(model,
                                 scaled_data[:, :-1],
                                 scaled_data[:, -1],
                                 cv=splits,
                                 scoring='neg_mean_absolute_error')
        all_scores.append(scores.mean())
    plt.hist(all_scores)
    plt.xlabel("Mean Squared Error")
    plt.title("Range of Scores of $\lambda=10^{-8}$")
    plt.ylabel("Count of Runs")
    plt.show()
    print(np.percentile(all_scores, 5),
          np.mean(all_scores),
          np.percentile(all_scores, 95),
          model.alpha)
