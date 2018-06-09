###here is all the stuff for kNN
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn import preprocessing



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

classifier = KNeighborsClassifier(n_neighbors=1)
pred = classifier.fit(train[features], trainTargets).predict(test[label])


control = test[label].values
    
correct = 0
for i in range(0, len(pred)):
    if pred[i] == np.string_(control[i][0]):
        correct += 1

total = len(pred)

results = correct / float(total)
