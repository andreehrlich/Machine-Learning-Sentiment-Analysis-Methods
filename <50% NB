import nltk
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from pandas import DataFrame
import numpy as np
from nltk.classify import NaiveBayesClassifier as nbc
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
gnb = GaussianNB()
y_gnb = gnb.fit(train[features], trainTargets).predict(test[label])


def accuracy(experiment, control):
    correct = 0
    for i in range(0, len(experiment)):
        print(experiment[i])
        print(control[i][0])
        print(type(experiment[i]))
        print(type(np.string_(control[i][0])))
        if experiment[i] == np.string_(control[i][0]):
            correct += 1

    total = len(experiment)
    return correct / float(total)
###############################################################################
def preprocess(text):
    spaced = ''
    for i in text:
        if i == ',':
            spaced += ' '
        elif "\"" not in i:
            spaced += i

    return spaced

def remove_commas(text):
    for i in text:
        if i == ',':
            text.remove(',')
        elif i == "''":
            text.remove("''")
        elif i == "\"":
            text.remove("\"")

    return text


def remove_stop_words(text, sw):
    new = []


def main():
    goodness = accuracy(y_gnb, test[label].values)
    print(goodness)

#####################################################################
def past_main():
    sentiments_data = open("/Users/AnkurG/Desktop/EECS349-ML/Projects/datasets/sentiments.csv")
    training_data1 = open("/Users/AnkurG/Desktop/EECS349-ML/Projects/datasets/trainingandtestdata/training.1600000.processed.noemoticon.csv")
    stop_words = open("stop_words.txt")

    ########################################sentiments information
    sentiments = sentiments_data.readlines()
    headers = sentiments[0]

    for i in range(1, len(sentiments)):
        line = sentiments[i]
        line = preprocess(line)
        line_array = nltk.word_tokenize(line)
        classification = line_array[1]
    #########################################

    #######################################training_data
    trains = training_data1.readlines()

    #######################################
    ex = sentiments_data.readlines()[2]
    ex2 = training_data1.readlines()[2]
    list_of_stop_words = stop_words.readlines()[0]

    pre_processed = preprocess(ex)
    pre_processed2 = preprocess(ex2)
    stop_processed = preprocess(list_of_stop_words)

    token = nltk.word_tokenize(pre_processed)
    token = remove_commas(token)
    token2 = nltk.word_tokenize(pre_processed2)
    token2 = remove_commas(token2)
    stop_token = nltk.word_tokenize(list_of_stop_words)
    stop_token = remove_commas(stop_token)
    ##################

    print(token)

    print(token2)


if __name__ == "__main__":
    main()
