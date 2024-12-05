import json
import pickle5 as pickle
import nltk
import random
import numpy as np
import unicodedata

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def remove_accents(input_str):
    """
    Remove acentos de uma string
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

words = []
documents = []
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

classes = [i['tag'] for i in intents['intents']]
ignore_words = ["!", "@", "#", "$", "%", "*", "?"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = remove_accents(pattern)
        
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    bag = [1 if word in pattern_words else 0 for word in words]

    if len(bag) != len(words):
        print(f"Tamanho da bag inconsistente: {len(bag)}")

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

for i, (bag, output) in enumerate(training):
    if len(bag) != len(words) or len(output) != len(classes):
        print(f"Erro na posição {i}: bag tem tamanho {len(bag)}, output tem tamanho {len(output)}")

random.shuffle(training)
x = np.array([item[0] for item in training], dtype=np.float32)
y = np.array([item[1] for item in training], dtype=np.float32)

model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x, y, epochs=200, batch_size=5, verbose=1)
model.save('model.keras')

print("Fim")

