###this contains GaussianNB and MultinomialNB

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn import preprocessing
import numpy as np

data = pd.read_csv("/Users/AnkurG/Desktop/EECS349-ML/Projects/datasets/sentiments_mod.csv")

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

target = np.array(['Good', 'Bad'])
classifs = []
for i in data.values:
    classification = i[len(i) - 1]
    classifs.append(classification)

data['train'] = np.random.uniform(0, 1, len(data)) <= 0.75
data['type'] = pd.Categorical(classifs, target)
data['targets'] = classifs

train = data[data['train'] == 1]
test = data[data['train'] == 0]

trainTargets = np.array(train['targets']).astype(str)
testTargets = np.array(test['targets']).astype(int)

features = data.columns[0:1]
label = data.columns[1:2]


gnb = GaussianNB()
gnb_res = gnb.fit(train[features], trainTargets).predict(test[label])

mnb = MultinomialNB()
mnb_res = mnb.fit(train[features], trainTargets).predict(test[label])


control = test[label].values
gnb_experiment = gnb_res
mnb_experiment = mnb_res
    
gnb_correct = 0
for i in range(0, len(gnb_experiment)):
    if gnb_experiment[i] == np.string_(control[i][0]):
        gnb_correct += 1

mnb_correct = 0
for i in range(0, len(mnb_experiment)):
    if mnb_experiment[i] == np.string_(control[i][0]):
        mnb_correct += 1

gnb_total = len(gnb_experiment)
gnb_accuracy = gnb_correct / float(gnb_total)

mnb_total = len(mnb_experiment)
mnb_accuracy = mnb_correct / float(mnb_total)


