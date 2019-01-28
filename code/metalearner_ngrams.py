import numpy as np # linear algebra
import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA

import pandas as pd
from pathlib import Path

#load the data:
train = pd.read_csv(Path("../input/train.csv"))
test = pd.read_csv(Path('../input/test.csv'))
sub = pd.read_csv(Path('../input/sample_submission.csv'))
print("done reading input files")
print(test.head())

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train['question_text'], train['target'])

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('engeneering features...')
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train['question_text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
xtest_tfidf  = tfidf_vect.transform(test['question_text'])  

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(train['question_text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(test['question_text'])

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(train['question_text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test['question_text'])

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from sklearn.decomposition import TruncatedSVD
print('classification...')
classifier = make_pipeline(Normalizer(), TruncatedSVD(n_components=500, n_iter=7, random_state=42), naive_bayes.GaussianNB()) #naive_bayes.MultinomialNB()
classifier.fit(xtrain_tfidf, train_y)
wordlvl_predictions_valid = classifier.predict(xvalid_tfidf)
wordlvl_predictions = classifier.predict(xtest_tfidf)

classifier.fit(xtrain_tfidf_ngram, train_y)
ngramlvl_predictions_valid = classifier.predict(xvalid_tfidf_ngram)
ngramlvl_predictions = classifier.predict(xtest_tfidf_ngram)

classifier.fit(xtrain_tfidf_ngram_chars, train_y)
charlvl_predictions_valid = classifier.predict(xvalid_tfidf_ngram_chars)
charlvl_predictions = classifier.predict(xtest_tfidf_ngram_chars)

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import sklearn.metrics

f1_score = sklearn.metrics.f1_score(valid_y, wordlvl_predictions_valid)
print("F1-score word lvl: %f" % f1_score)

f1_score = sklearn.metrics.f1_score(valid_y, ngramlvl_predictions_valid)
print("F1-score ngram lvl: %f" % f1_score)

f1_score = sklearn.metrics.f1_score(valid_y, charlvl_predictions_valid)
print("F1-score char lvl: %f" % f1_score)
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('conc matrices')
conc_matrix_valid = np.vstack((wordlvl_predictions_valid, ngramlvl_predictions_valid, charlvl_predictions_valid)).T
conc_matrix_test = np.vstack((wordlvl_predictions, ngramlvl_predictions, charlvl_predictions)).T

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('train meta learner...')
from sklearn.svm import SVC

metaLearner = clf = SVC(kernel='poly')
metaLearner.fit(conc_matrix_valid, valid_y)

super_predictions = metaLearner.predict(conc_matrix_test)
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('wring to file...')
# write to file
n = sub.columns[1]
sub.drop(n, axis = 1, inplace = True)
sub[n] = super_predictions
sub.to_csv("submission.csv", index=False)

