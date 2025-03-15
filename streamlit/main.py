import streamlit as st
import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow import keras
import os

# Load necessary data
stemmer = LancasterStemmer()

dir_model = "./model"

with open(os.path.join(dir_model, "training_data"), "rb") as file:
    data = pickle.load(file)

words = data['words']
classes = data['classes']
train_x = np.array(data['train_x'])
train_y = np.array(data['train_y'])

with open(os.path.join(dir_model, 'intents.json')) as data_file:
    intents = json.load(data_file)

# Load the trained model
model = keras.models.load_model(os.path.join(dir_model, 'model.h5'))

# Function to clean and tokenize user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert sentence into bag-of-words representation
def bow(sentence, words, show_details=False):
    bag = [0] * len(words)
    sentence_words = clean_up_sentence(sentence)
    for s in sentence_words:
        for i, w in enumerate(words):
            if s == w:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

# Function to classify user input
def classify(sentence):
    ERROR_THRESHOLD = 0.30
    results = model.predict(np.array([bow(sentence, words)]))[0]
    results = [[i, r] for i, r in enumerate(results) if r >= ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

# Function to generate chatbot response
def response(sentence):
    results = classify(sentence)
    if results:
        for intent in intents['intents']:
            if intent['tag'] == results[0][0]:
                return random.choice(intent['responses'])
    return "Maaf, saya tidak mengerti apa yang Anda katakan."

# Streamlit UI
st.title("Chatbot AI")
st.write("Interaksi dengan chatbot berbasis AI")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Input pengguna
user_input = st.text_input("Anda: ", "")
if st.button("Kirim") and user_input:
    bot_response = response(user_input)
    st.session_state.chat_history.append(("Anda", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))

# Tampilkan riwayat percakapan dalam text area
chat_text = "\n".join([f"{sender}: {message}" for sender, message in st.session_state.chat_history])
st.text_area("Percakapan", chat_text, height=300, disabled=True)
