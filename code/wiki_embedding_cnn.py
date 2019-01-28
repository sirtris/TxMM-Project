import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings"))

#load the data:
train = pd.read_csv(Path("../input/train.csv"))
test = pd.read_csv(Path('../input/test.csv'))
sub = pd.read_csv(Path('../input/sample_submission.csv'))
y = train.target.values
print("done reading input files")
print(test.head())


# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train['question_text'], train['target'])

#train_x = train['question_text']
#train_y = train['target']

test_x = test['question_text']

# load the pre-trained word-embedding vectors 
# needs about 999995it to finish
embeddings_index = {}
for i, line in enumerate(tqdm(open('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec', encoding="utf8"))):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')


# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(train['question_text'])
word_index = token.word_index


# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)
test_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# train cnn
cnn_model = create_cnn()
cnn_model.fit(train_seq_x, train_y)

#make predictions
predictions = cnn_model.predict(test_seq_x, verbose=1)
# following line wrong?
predictions = predictions.argmax(axis=-1)

# save results to file
n = sub.columns[1]
sub.drop(n, axis = 1, inplace = True)
sub[n] = predictions
sub.to_csv("submission.csv", index=False)