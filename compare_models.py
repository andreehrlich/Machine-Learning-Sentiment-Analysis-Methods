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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# # Decomposition
# from sklearn.decomposition import TruncatedSVD
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Plot
import matplotlib.pyplot as plt


"""
Sources:
https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

"""
# Get data from file
datafile = os.getcwd() + "/datasets/sentiments_mod.csv"
data = pd.read_csv(datafile)

# Get a list of target values
classifs = []
for i in data.values:
    classification = i[len(i) - 1]
    classifs.append(classification)

targets = np.array(classifs).astype(str)
features = np.array(data['SentimentText'])

# Transform text documents to TF-IDF word count feature vectors
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(features)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

models = [	('MultinomialNB', MultinomialNB()),
			("Perceptron", Perceptron()),
			("BernoulliNB", BernoulliNB()),
			("SGDClassifier", SGDClassifier()),
			("LogisticRegression", LogisticRegression()),
			# ("SVC", SVC()) # SVC is not terminating!!!!
]

# Evaluate Each Model
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	# Add exhaustive parameter tuning for each model here. #TODO
	kfold = KFold(10, random_state=808)
	cv_results = cross_val_score(model, X_train_tfidf, targets, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	print "{0}: avg:{1}, std:{2}".format(name, cv_results.mean(), cv_results.std())

# Plot results to compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

