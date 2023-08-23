import numpy as np
import os
import pandas as pd
import email
from collections import Counter
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score


loc = 'D:\\SMUMSDS\\QuantWorld\\CS3\\SpamAssassinMessages'
os.listdir(loc)

spam = os.listdir(os.path.join(loc, 'spam'))
spam_2 = os.listdir(os.path.join(loc, 'spam_2'))

hard_ham = os.listdir(os.path.join(loc, "./hard_ham"))
easy_ham = os.listdir(os.path.join(loc, "./easy_ham"))
easy_ham_2 = os.listdir(os.path.join(loc, "./easy_ham_2"))

# Count the number of files using specific decoder
x = os.listdir(os.path.join(loc, "./easy_ham"))

count = 0
for msg in x:
    path_ = os.path.join(os.path.join(loc, "./easy_ham", msg))
    with open(path_, "r", encoding="latin-1") as file_handler:
        msg = file_handler.read()
        if count % 20 == 0:
            print(count)
        count += 1

y = os.listdir(os.path.join(loc, "./hard_ham"))

count = 0
for msg in y:
    path_ = os.path.join(os.path.join(loc, "./hard_ham", msg))
    with open(path_, "r", encoding="latin-1") as file_handler:
        msg = file_handler.read()
        if count % 20 == 0:
            print(count)
        count += 1

# Get the labels for all our emails
"""
Exclude Extensions that are not relevant
"""
file_name = []
label = []
extensions_to_exclude = ['.ipynb', '.DS_Store', 'cmds']

for root, dirs, files in os.walk(os.path.join(loc, ".")):
    for f in files:
        if any(f.endswith(ext) for ext in extensions_to_exclude):
            continue
        if "spam" in root:
            label.append(1)
        else:
            label.append(0)
        file_name.append(os.path.join(root, f))

# Create a data frame
data = pd.DataFrame({"Message": file_name, "Target": label})

# Read in text messages

"""
Lets count the types of messages we have first
"""
types = Counter()
msgs = []
trigger = True
for root, dirs, files in os.walk(os.path.join(loc, "./")):
    for f in files:
        with open(os.path.join(root, f), 'r',
                  encoding='latin-1') as file_point:
            msg = email.message_from_file(file_point, )
            type_ = msg.get_content_type()
            types[type_] += 1
            if type_ == 'multipart/mixed' and trigger:
                print(msg.get_payload())
                print("________")
                trigger = False

# print(types)


extensions_to_exclude = ['.ipynb', '.DS_Store', 'cmds']
msgs = []
for root, dirs, files in os.walk(os.path.join(loc, "./")):
    for f in files:
        if any(f.endswith(ext) for ext in extensions_to_exclude):
            continue
        file_path = os.path.join(root, f)
       
        with open(file_path, "r", encoding='latin-1') as file:
            message = email.message_from_file(file)
        for part in message.walk():
            if part.get_content_type() in ["text/plain", "text/html"]:
                if part.get_content_type() == "text/html":
                    soup = BeautifulSoup(part.get_payload(decode=True)
                                         .decode("latin-1"), "html.parser")
                    body = soup.get_text()
                else:
                    body = part.get_payload(decode=True).decode("latin-1")
        msgs.append(body)

data['messages'] = msgs

# Clean Text
# start by removing \n
data['messages'] = data['messages'].str.replace("\n", " ")

data['messages'][9345]


def replace_urls(text):

    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|\
    [!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return re.sub(url_pattern, "URL", text)


data['messages'] = data['messages'].apply(replace_urls)

data['messages'][9345]

# Remove unwanted characters using regex
data['messages'] = data['messages'].str.replace('[^\w\s]', ' ', regex=True)
data['messages'] = data['messages'].str.replace('\d+', '', regex=True)
data['messages'] = data['messages'].str.replace('_', ' ', regex=True)
data['messages'] = data['messages'].str.replace('\s+', ' ', regex=True)

data['messages'][9345]

# CountVectorizer Usage
vectorizer = CountVectorizer()
out = vectorizer.fit_transform(data['messages'].astype('str'))

# Sparse Matrix
# Out is a sparse matrix (mostly zeroes) To view it, convert to an array
out[0].toarray()

# Vocab
# There is a dictionary of word to column Here let's find where the
# column 'Martin' is counted Since the default in CountVectorizer is
# to lower case everything Martin, martin, MaRtiN are all counted as 'martin'
vectorizer.vocabulary_

vectorizer.vocabulary_['handheld']

# How to reverse a dictionary
# and translate back and forth between vector and text
reverse = {value: key for key, value in vectorizer.vocabulary_.items()}
out[0].toarray()[:, 32941]

# Compare these two, Notice what CountVectorizer did and did not include
for i in range(57736):
    if out[0].toarray()[:, i] != 0:
        print(reverse[i], out[0].toarray()[:, i])

# Add Target
targets = data['Target']

model = MultinomialNB(alpha=0.1)

# Evaluate the model using cross-validation
scores = cross_val_score(model, out, targets, cv=10)

# Calculate precision score using cross-validation
precision_scorer = make_scorer(precision_score, pos_label=1)
precision_cv = cross_val_score(
    model, out, targets, cv=10,
    scoring=precision_scorer
)
# Take the mean of the scores to obtain a single estimate of the precision
precision = np.mean(precision_cv)

# Print the mean and standard deviation of the cross-validation scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Precision: %.2f" % precision)

# Evaluate the model using cross-validation
scores = cross_val_score(model, out, targets, cv=10)

# Calculate precision score using cross-validation
precision_scorer = make_scorer(precision_score, pos_label=1)
precision_cv = cross_val_score(
    model, out, targets, cv=10,
    scoring=precision_scorer)
# Take the mean of the scores to obtain a single estimate of the precision
precision = np.mean(precision_cv)

# Calculate recall score using cross-validation
recall_scorer = make_scorer(recall_score, pos_label=1)
recall_cv = cross_val_score(model, out, targets, cv=10, scoring=recall_scorer)
# Take the mean of the scores to obtain a single estimate of the recall
recall = np.mean(recall_cv)

# Print the mean and standard deviation of the cross-validation scores
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
print("Precision: %.4f" % precision)
print("Specificity: %.4f" % recall)
