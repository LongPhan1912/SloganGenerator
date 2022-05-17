# Dataset collected from: https://www.kaggle.com/datasets/chaibapat/slogan-dataset?select=sloganlist.csv

# keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
# citation: https://stackoverflow.com/questions/70240387/attributeerror-module-keras-utils-has-no-attribute-to-categorical
from keras.utils import np_utils 

# set seeds for reproducability
import tensorflow as tf
from numpy.random import seed

# set random number generator
tf.random.set_seed(2) 
seed(1)

import pandas as pd
import numpy as np
import string, os 

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# tokenizer global variable
tokenizer = Tokenizer()


def clean_text(text):
    # remove punctation and make the text lowercase
    text = "".join(v for v in text if v not in string.punctuation).lower()
    # encode string as utf-8 bytes, then decode the bytes into an 'ascii' string
    text = text.encode("utf8").decode("ascii",'ignore')
    # remove any whitespaces in between words
    text = " ".join(text.split())
    return text 

def get_sequence_of_tokens(corpus):
    # creates the vocabulary index (think of a dictionary / map data structure) based on word frequency
    # the lower the integer, the more frequent a word appears in the corpus
    tokenizer.fit_on_texts(corpus)
    total_num_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    for line in corpus:
        # assign integers to words based on the tokenizer's word_index dictionary
        # basically a lookup function done by the keras library
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]  # bigram, trigrams, 4-grams, and so on
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_num_words

# as neural networks require inputs to have the same shape and size,
# we need to pad the text sequences so they all have the exact same length
def generate_padded_sequences(input_sequences, total_num_words):
    # get the max length as our benchmark
    max_sequence_len = max([len(x) for x in input_sequences])
    # use keras' pad_sequences function to pad the n-gram sequences
    # padding = 'pre' means pad zeros at the beginning of the sequence
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    # predictors are the n-gram sequences, whereas the label is the next word succeeding the n-gram sequence
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    # create a matrix of binary values with the number of classes equal to the number of columns / total number of words in the corpus
    # basically we mark the word label with the boolean 1, else for any word in the corpus that's not the label, we mark it as 0
    label = np_utils.to_categorical(label, num_classes=total_num_words)
    return predictors, label, max_sequence_len

# now we can train the data!
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    # embedding is a form of word representation that allows words to have the same meaning to obtain the same representation
    # This is where we represent individual words as real-valued vectors in a predefined vector space. 
    # We then map each word to a vector so that the the vector values are learned in a way that resembles a neural network.
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    # In explanation, LSTM is a recurrent neural network capable of handling long-term dependencies.
    # In other words, it can retain / remember information for a long period of time
    # so to process, predict, and classify sequenced data
    model.add(LSTM(50))  # 50 LTSM blocks or neurons in our network
    model.add(Dropout(0.1))  # randomly sets input units to zero at a frequency of 0.1 at each step of the training time

    # Add Output Layer
    # Basically triggers the implementation for the activation function to produce the output
    model.add(Dense(total_words, activation='softsign'))

    # Add configurations to the model with the loss function and the optimizer in place
    model.compile(loss='categorical_crossentropy', optimizer='Adam')
    
    return model

# finally, we can generate text for slogans from our model architecture
def generate_text(seed_text, next_words, model, max_seq_len):
    for _ in range(next_words):
        # tokenize the seed text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # pad sequences so all input have the same size
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        # predict the model

        # ran into trouble with this line as TensorFlow version 2.6 does not support this function call anymore
        # predicted_idx = model.predict_classes(token_list, verbose=0)
        # fix: https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes
        predicted_idx = model.predict(token_list, verbose=0) 
        predicted_idx = np.argmax(predicted_idx, axis=1)
        
        word_to_append = ""
        for word, idx in tokenizer.word_index.items():
            # basically we find the word with a matching index to our prediction
            if idx == predicted_idx:
                word_to_append = word
                break
        # append the word with a space delimiter in between
        seed_text += " " + word_to_append
    # get our predicted sequence
    return seed_text.title()

def create_slogans():
    # load in the dataset
    slogan_file = "sloganlist.csv"
    df = pd.read_csv(slogan_file)
    # get the collection of 1,163 slogans and ignore the name of the companies with the particular slogan
    list_of_slogans = df["Slogan"].tolist()
    # get the corpus for our project
    corpus = [clean_text(x) for x in list_of_slogans]
    # generate sequence of tokens
    input_seqs, total_words = get_sequence_of_tokens(corpus)
    # pad sequences of text
    predictors, label, max_sequence_len = generate_padded_sequences(input_seqs, total_words)
    # create our model used for training
    model = create_model(max_sequence_len, total_words)
    # now we train the model with 100 epochs, using our predictors and label numpy arrays
    model.fit(predictors, label, epochs=150, verbose=5)
    # ahh the fun part -- generating our slogans
    print(generate_text("It's kind of", next_words=3, model=model, max_seq_len=max_sequence_len))
    print(generate_text("Be", next_words=3, model=model, max_seq_len=max_sequence_len))
    print(generate_text("A diamond is", next_words=3, model=model, max_seq_len=max_sequence_len))
    print(generate_text("There are some", next_words=3, model=model, max_seq_len=max_sequence_len))
    print(generate_text("It's yours", next_words=3, model=model, max_seq_len=max_sequence_len))
    print(generate_text("The ultimate", next_words=3, model=model, max_seq_len=max_sequence_len))
    print(generate_text("You are", next_words=3, model=model, max_seq_len=max_sequence_len))
    print(generate_text("What's in", next_words=3, model=model, max_seq_len=max_sequence_len))

create_slogans()
