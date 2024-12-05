import random
import numpy as np
import pickle5 as pickle
import nltk
from nltk.stem import WordNetLemmatizer
import unicodedata

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def remove_accents(input_str):
    """
    Remove acentos de uma string
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def clear_writing(writing):
    """
    Limpa todas as sentenças inseridas, removendo acentos e lematizando.
    """
    writing = remove_accents(writing)
    
    sentence_words = nltk.word_tokenize(writing)
    
    lemmatized_words = []
    for word in sentence_words:
        lemma_masc = lemmatizer.lemmatize(word.lower(), pos='v')
        lemma_fem = lemmatizer.lemmatize(lemma_masc, pos='n')
        lemmatized_words.append(lemma_fem)
    
    return lemmatized_words

def bag_of_words(writing, words):
    """
    Pega as sentenças que são limpas e cria um pacote de palavras 
    para classes de previsão.
    """
    sentence_words = clear_writing(writing)

    bag = [0]*len(words)
    for setence in sentence_words:
        for i, word in enumerate(words):
            if word == setence:
                bag[i] = 1

    return(np.array(bag))

def class_prediction(writing, model):
    """
    Faz a previsao do pacote de palavras, usamos como limite de erro 0.25 
    para evitarmos overfitting e classificamos esses resultados por força da probabilidade.
    """
    prevision = bag_of_words(writing, words)
    response_prediction = model.predict(np.array([prevision]))[0]
    results = [[index, response] for index, response in enumerate(response_prediction) if response > 0.25]    

    if "1" not in str(prevision) or len(results) == 0 :
        results = [[0, response_prediction[0]]]

    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents, intents_json):
    """
    Pega a lista gerada e verifica o arquivo json e produz a maior parte das 
    respostas com a maior probabilidade.
    """
    tag = intents[0]['intent']
    list_of_intents = intents_json['intents']
    for idx in list_of_intents:
        if idx['tag'] == tag:
            result = random.choice(idx['responses'])
            break

    return result