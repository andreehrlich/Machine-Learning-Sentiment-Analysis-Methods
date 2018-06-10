import numpy as np
import pandas as pd
import sys  # for command line tool
import os  # to locate data sets relative to this path

# Data Preprocessing modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

# Model Selection modules
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Decomposition
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Get data from file
import sys

datafile = sys.argv[1]
algo = sys.argv[2]
feature_title = sys.argv[3]
target_title = sys.argv[4]

data = pd.read_csv(datafile)

targets = data[target_title].values.astype(str)
features = np.array(data[feature_title])

# Split data for x-validation
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, random_state=0)

if algo == "MultinomialNB" or algo == "multinomialNB":
    SGD_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 4))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),
                    ])
elif algo == "LO" or algo == "lo" or algo == "Logistic Regression" or algo == "logistic regression":
    SGD_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 4))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression()),
                    ])
elif algo == "Perceptron" or algo == "perceptron":
    SGD_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 4))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', Perceptron()),
                    ])
elif algo == "BernoulliNB" or algo == "bernoulliNB":
     SGD_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 4))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', BernoulliNB()),
                    ])
elif algo == "SGDClassifier" or algo == "sgdClassifier":
     SGD_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 4))),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier()),
                    ])


# Fit model to training set
SGD_clf.fit(X_train, y_train)

# Predict on test set
SVM_pred = SGD_clf.predict(X_test)



# Calculate the accuracy of model predictions on a test set by comparing with correct labels.
def accuracy(experiment, control):
    correct = 0
    for i in range(0, len(experiment)):
        if experiment[i] == np.string_(control[i][0]):
            correct += 1

    total = len(experiment)
    return correct / float(total)


def main():
    # SVM
    svm_acc = accuracy(SVM_pred, y_test) * 100
    print("Accuracy: {0}%".format(svm_acc))

    # trunc + SVC
    # trunc_svc_acc = accuracy(trunc_svc_pred, y_test) * 100
    # print("Accuracy: {0}%".format(trunc_svc_acc))

    # Parameter tuning
    # gs_acc = accuracy(gs_pred, y_test) * 100
    # print("Accuracy: {0}%".format(gs_acc))


main()
