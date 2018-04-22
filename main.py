#  PART 1 - load

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt




from nltk.tokenize import WordPunctTokenizer
from collections import Counter
# from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm



from gensim.models import Word2Vec



from keras.preprocessing.sequence import pad_sequences



from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

plt.style.use("ggplot")

path = 'data/'

TRAIN_DATA_FILE = path + 'train.csv'
TEST_DATA_FILE = path + 'test.csv'

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

# train_df.head(10)

# print(train_df.head(10))
# print(test_df.head(10))

# PART 2 - process

print('Processing text dataset')


# replace urls
re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
                    .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
                    re.MULTILINE | re.UNICODE)
# replace ips
re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

# setup tokenizer
tokenizer = WordPunctTokenizer()

vocab = Counter()


def text_to_wordlist(text, lower=False):
    # replace URLs
    text = re_url.sub("URL", text)

    # replace IPs
    text = re_ip.sub("IPADDRESS", text)

    # Tokenize
    text = tokenizer.tokenize(text)

    # optional: lower case
    if lower:
        text = [t.lower() for t in text]

    # Return a list of words
    vocab.update(text)
    return text


def process_comments(list_sentences, lower=False):
    comments = []
    for text in tqdm(list_sentences):
        txt = text_to_wordlist(text, lower=lower)
        comments.append(txt)
    return comments


list_sentences_train = list(train_df["comment_text"].fillna("NAN_WORD").values)
list_sentences_test = list(test_df["comment_text"].fillna("NAN_WORD").values)

comments = process_comments(list_sentences_train + list_sentences_test, lower=True)

print("The vocabulary contains {} unique tokens".format(len(vocab)))

#  PART 3 - teach


model = Word2Vec(comments, size=100, window=5, min_count=5, workers=16, sg=0, negative=5)
word_vectors = model.wv
print("Number of word vectors: {}".format(len(word_vectors.vocab)))
print(model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man']))

#  PART 4 - initialize embedding in keras

MAX_NB_WORDS = len(word_vectors.vocab)
MAX_SEQUENCE_LENGTH = 200


word_index = {t[0]: i + 1 for i, t in enumerate(vocab.most_common(MAX_NB_WORDS))}
sequences = [[word_index.get(t, 0) for t in comment]
             for comment in comments[:len(list_sentences_train)]]
test_sequences = [[word_index.get(t, 0) for t in comment]
                  for comment in comments[len(list_sentences_train):]]

# pad
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,
                     padding="pre", truncating="post")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_df[list_classes].values
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre",
                          truncating="post")
print('Shape of test_data tensor:', test_data.shape)

WV_DIM = 100
nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
# we initialize the matrix with random numbers
wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass


#  PART 6 - setup comment clasifier



wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)

# Inputs
comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = wv_layer(comment_input)

# biGRU
embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(embedded_sequences)

# Output
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
preds = Dense(6, activation='sigmoid')(x)

# build the model
model = Model(inputs=[comment_input], outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
              metrics=[])

hist = model.fit([data], y, validation_split=0.1,
                 epochs=10, batch_size=256, shuffle=True)

history = pd.DataFrame(hist.history)
plt.figure(figsize=(12,12));
plt.plot(history["loss"]);
plt.plot(history["val_loss"]);
plt.title("Loss with pretrained word vectors");
plt.show();

# wv_layer = Embedding(nb_words,
#                      WV_DIM,
#                      mask_zero=False,
#                      # weights=[wv_matrix],
#                      input_length=MAX_SEQUENCE_LENGTH,
#                      trainable=False)
#
# # Inputs
# comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = wv_layer(comment_input)
#
# # biGRU
# embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
# x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(embedded_sequences)
#
# # Output
# x = Dropout(0.2)(x)
# x = BatchNormalization()(x)
# preds = Dense(6, activation='sigmoid')(x)
#
# # build the model
# model = Model(inputs=[comment_input], outputs=preds)
# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
#               metrics=[])
#
# hist = model.fit([data], y, validation_split=0.1,
#                  epochs=10, batch_size=256, shuffle=True)
#
#
# history = pd.DataFrame(hist.history)
# plt.figure(figsize=(12,12));
# plt.plot(history["loss"]);
# plt.plot(history["val_loss"]);
# plt.title("Loss with random word vectors");
# plt.show();
