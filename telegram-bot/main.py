import telebot
import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow import keras
import os

# Replace with your Telegram Bot API token
API_TOKEN = 'YOUR_TELEGRAM_BOT_API'
bot = telebot.TeleBot(API_TOKEN)

# Load necessary data
stemmer = LancasterStemmer()

dir_model = "model"

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

# Telegram bot message handler
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    bot_response = response(user_input)
    bot.send_message(message.chat.id, bot_response)

bot.polling()
