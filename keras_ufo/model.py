## SAMUEL CARDOSO
## 31 Avril 2019
###################
## DOCUMENTATION ##
###################
## Env. ---- 

## Install Python >=3
## homebrew is your friend

## "brew install python3"
## In your terminal go to the project folder
## python3 -m venv .venv
## source .venv/bin/activate
## pip install -r requirements.txt

## Usage ----

##  !! You must be in the project folder with your terminal !!

## - Training
## python3 model.py --train --dataset <data.txt>

## - Example
## python3 model.py --train --dataset data.txt
## the trained model and a binary file will be saved in the script folder

## - Prediction
## python3 model.py --predict --next <first_word> --model <filename.h5> --nbr <number of words to predict>

## - Example
## python3 model.py --predict --next bird --model test.h5 --nbr 3

## ENJOY

from argparse import ArgumentParser

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model

import keras.utils as ku 
import numpy as np 
import pickle

# *20th century fox theme, playing*

tokenizer = Tokenizer()

class Data:
    def __init__(self, predictors, label, max_sequence_len, total_words, tokenizer):
        self.predictors = predictors
        self.label = label
        self.max_sequence_len = max_sequence_len
        self.total_words = total_words
        self.tokenizer = tokenizer

    def save(self, filename):
        bin_file = open(filename + '.bin', mode='wb')
        pickle.dump(self, bin_file)
        bin_file.close()

def dataset_preparation(data):

    corpus = data.lower().split("\n")

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words

def create_model(predictors, label, max_sequence_len, total_words):
	
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(LSTM(150, return_sequences = True))

    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(predictors, label, epochs=100, verbose=1, callbacks=[earlystop])
    print(model.summary())
    
    return model 

def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
		
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# PARSER
parser = ArgumentParser()
parser.add_argument("--train",
                    action='store_true',
                    help="TRAIN MODE")
parser.add_argument("--dataset",
                    type=str,
                    action='store',
                    help="dataset to learn from")
parser.add_argument("--output", 
                    type=str,
                    action='store',
                    help="filename to save the h5 from the learning process")
parser.add_argument("--predict", 
                    action='store_true',
                    help="PREDICT MODE")
parser.add_argument("--next", 
                    type=str,
                    action='store',
                    help="first word to complete")
parser.add_argument("--model",
                    type=str,
                    action='store',
                    help="trained model to use")
parser.add_argument("--number", 
                    type=int,
                    action='store',
                    help="number of words to predict")
args = parser.parse_args()

# MAIN
if args.train:
    if args.dataset is not None:
        data = open(args.dataset).read()
        predictors, label, max_sequence_len, total_words = dataset_preparation(data)
        d = Data(predictors=predictors, label=label, 
                max_sequence_len=max_sequence_len, total_words=total_words,
                tokenizer=tokenizer)

        d.save(args.dataset.split('.')[0])
        model = create_model(predictors, label, max_sequence_len, total_words)
        model.save(args.dataset.split('.')[0] + '.h5')
    else:
        parser.print_help()
elif args.predict:
    if args.next is not None and args.model is not None and args.number is not None:
        model = load_model(args.model)
        with open(args.model.split('.')[0] + '.bin', 'rb') as f:
            data = pickle.load(f)
        tokenizer = data.tokenizer
        print(generate_text(args.next, args.number, data.max_sequence_len))
else:
    parser.print_help()
