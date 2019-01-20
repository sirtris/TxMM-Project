# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:25:28 2019

@author: tpayer
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn import decomposition, ensemble

#import pandas as pd
from pathlib import Path
#from nltk.corpus import stopwords
#%%
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')
#%%
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
#from sklearn.preprocessing import Normalizer
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.svm import SVC
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import re
import string
#import numpy as np
from tqdm import tqdm

#%%
#load the data:
train = pd.read_csv(Path("../input/train.csv"))
test = pd.read_csv(Path('../input/test.csv'))
sub = pd.read_csv(Path('../input/sample_submission.csv'))
y = train.target.values
print("done reading input files")
print(test.head())

#%%
def extract_features(text):
    bag_of_words = [x for x in wordpunct_tokenize(text)]

    features = []
    #**************************************************************************
    ## Countable/statistical features
    #**************************************************************************
    # Example feature 1: count the number of words
    num_words = len(bag_of_words)
    features.append(num_words)

    # Example feature 2: count the number of words, excluded the stopwords
    #features.append(len([x for x in bag_of_words if x.lower() not in stop_words]))

    # words without vowels
    features.append(len([word for word in bag_of_words if not re.search('[aeiou]', word.lower(), re.I)]))

    # number of sentences
    sentences  = sent_tokenize(text)
    num_sentences = len(sentences)
    features.append(num_sentences)

    # avg number of words per sentence
    words_per_sentence = np.array([len(wordpunct_tokenize(s)) for s in sentences])
    features.append(np.average(words_per_sentence))

    # standard deviation number of words per sentence
    features.append(np.std(words_per_sentence))

    # number of characters including whitespace
    features.append(len(text))

    # number of characters without whitespace
    features.append(len(text.replace(" ", "")))

    # punctuation count
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    features.append(count(text, set(string.punctuation)))

    # number of numbers
    features.append(sum(c.isdigit() for c in text))

    # number of alpha chars
    features.append(sum(c.isalpha() for c in text))

    # number of spaces
    features.append(sum(c.isspace() for c in text))

    # Commas per sentence
    features.append(bag_of_words.count(',') / float(len(sentences)))    # TODO alterative text.count(",") <- see what is faster

    # Semicolons per sentence
    features.append(bag_of_words.count(';') / float(len(sentences)))

    # Two/three continuous punctuation count
    features.append(len(re.findall('(\!|\?){2,}', text)))

    # number of all caps words
    # TODO use per sentence?
    features.append(sum(1 for word in bag_of_words if word.isupper()))

    # number of sentences starting with a lower case letter
    # TODO divide by number of sentences?
    features.append(sum(1 for sentence in sentences if not sentence[0].isupper()))

    # number of selfe reference
    features.append(bag_of_words.count("I") + bag_of_words.count("me") + bag_of_words.count("myslef") + bag_of_words.count("my") / num_words)

    # number of small letters 'i' instead of 'I'
    features.append(bag_of_words.count("i"))

    # number of sentences that have no space after a full stop
    features.append(len(re.findall('((\.|\?|\!|\:)\w+)', text)))

    # number of questions
    features.append(len(re.findall('\?', text)))

    # number of exclamation marks
    features.append(len(re.findall('\!', text)))

    # number of sentences that start with a number (only works if there is a space between sentences)
    features.append(sum(1 for sentence in sentences if sentence[0].isdigit()))

    # number of he
    features.append(bag_of_words.count("he") + bag_of_words.count("He"))

    # number of she
    features.append(bag_of_words.count("she") + bag_of_words.count("She"))

    # number of he/she
    features.append(bag_of_words.count("he/she") + bag_of_words.count("He/she"))
    
    #**************************************************************************
    # POS based features
    #**************************************************************************
    pos_tags = [pos_tag[1] for pos_tag in nltk.pos_tag(bag_of_words)]
    # count frequencies for common POS types
    pos_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
                'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN',
                'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

    counted_pos = []
    for part_of_speech in pos_list:
        counted_pos.append(pos_tags.count(part_of_speech))

    [features.append(i) for i in counted_pos]
    
    return features


def classify(train_features, train_labels, test_features):
    # Averaged total f-score 0.61038, F1-score: 0.7059
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,
    min_samples_split=2, random_state=0)
    clf.fit(train_features, train_labels)
    return clf.predict(test_features)

def evaluate(y_true, y_pred):
    # TODO: What is being evaluated here and what does it say about the performance? Include or change the evaluation
    # TODO: if necessary.
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    print("F1-score: %f" % f1_score)

    return recall, precision, f1_score


def main():
    features = list(map(extract_features, tqdm(list(train['question_text']))))
    
    # Classify and evaluate
    skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
    scores = []
    train_target = list(train['target'])
    for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(train['question_text'], train_target)):
        # Print the fold number
        print("Fold %d" % (fold_id + 1))

        # Collect the data for this train/validation split
        print('train features...')
        train_features = [features[x] for x in train_indexes]
        print('train labels...')
        train_labels = [train_target[x] for x in train_indexes]
        print('validation...')
        validation_features = [features[x] for x in validation_indexes]
        validation_labels = [train_target[x] for x in validation_indexes]

        # Classify and add the scores to be able to average later
        print('classify...')
        y_pred = classify(train_features, train_labels, validation_features)
        scores.append(evaluate(validation_labels, y_pred))

        # Print a newline
        print("")
        
    # Print the averaged score
    recall = sum([x[0] for x in scores]) / len(scores)
    print("Averaged total recall", recall)
    precision = sum([x[1] for x in scores]) / len(scores)
    print("Averaged total precision", precision)
    f_score = sum([x[2] for x in scores]) / len(scores)
    print("Averaged total f-score", f_score)
    print("")

    '''
    # TODO: Once you are done crafting your features and tuning your model, also test on the test set and report your
    # TODO: findings. How does the score differ from the validation score? And why do you think this is?
    test_data = load_test_data()
    test_features = list(map(extract_features, test_data.data))
    #
    y_pred = classify(features, train_data.target, test_features)
    evaluate(test_data.target, y_pred)
    '''
    
    test_features = list(map(extract_features, tqdm(list(test['question_text']))))
    y_pred = classify(features, train['target'], test_features)
    
    
    n = sub.columns[1]
    sub.drop(n, axis = 1, inplace = True)
    sub[n] = y_pred
    sub.to_csv("submission.csv", index=False)


#%%
if __name__ == '__main__':
    main()
