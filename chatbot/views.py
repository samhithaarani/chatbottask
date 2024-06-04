from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import random
from keras.models import load_model
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

# Load model and data files
model = load_model('chatbot/models/chatassistant_model.h5')
intents = json.loads(open('chatbot/data/intents.json').read())
words = pickle.load(open('chatbot/models/words.pkl', 'rb'))
classes = pickle.load(open('chatbot/models/classes.pkl', 'rb'))

from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    print(f"BOW for sentence '{sentence}': {p}")  # Debug BOW output
    res = model.predict(np.array([p]))[0]
    print(f"Model prediction raw result: {res}")  # Debug model prediction
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print(f"Predicted classes: {return_list}")  # Debug predicted classes
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "Sorry, I am having trouble understanding you."
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            print(f"Response for intent '{tag}': {result}")  # Debug response
            return result
    return "Sorry, I am having trouble understanding you."

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message')
        print(f"User message: {message}")  # Debug user message
        ints = predict_class(message, model)
        res = get_response(ints, intents)
        return JsonResponse({'response': res})

    return JsonResponse({'response': 'Invalid request'}, status=400)
