from flask import Flask, render_template, url_for, request
import pickle
import re
import os
import pandas as pd

file1 = open('lang_vectors.pkl', 'rb')
eng_v = pickle.load(file1)
rmu_v = pickle.load(file1)
acu_v = pickle.load(file1)
file1.close()

english_model = pickle.load(open('english_model', "rb"))
english_features = pickle.load(open('english_features', "rb"))
roman_urdu_model = pickle.load(open('roman_urdu_model', "rb"))
roman_urdu_features = pickle.load(open('roman_urdu_features', "rb"))
actual_urdu_model = pickle.load(open('actual_urdu_model', "rb"))
actual_urdu_features = pickle.load(open('actual_urdu_features', "rb"))


def getCountVectorizer(featureSet):
    vect = CountVectorizer()
    features = vect.fit_transform(featureSet)
    return vect


english_vectorizer = getCountVectorizer(english_features)
actual_urdu_vectorizer = getCountVectorizer(actual_urdu_features)
roman_urdu_vectorizer = getCountVectorizer(roman_urdu_features)


def findClass(text, modelName, vect):
    lab = ['Ham', 'spam']
    x = vect.transform([text]).toarray()
    p = modelName.predict(x)
    a = int(p[0])
    return lab[a]


def preprocess(text):
    text = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'http', text)
    text = re.sub('£|\$', 'money', text)
    text = re.sub(
        '\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'number', text)
    text = re.sub('\d+(\.\d+)?', 'numbr', text)
    text = re.sub('[^\w\d\s]', ' ', text)
    text = text.lower()
    return text


def calculateProbability(text_message, language_set):
    splitted_message = text_message.split()
    found = 0
    for i in range(len(splitted_message)):
        if splitted_message[i] in language_set:
        found += 1
    probability = float(found) / len(splitted_message)
    print(probability)
    return probability


def language_detection(text):
    processed_text = preprocess(text)
    english_probability = calculateProbability(processed_text, eng_v)
    roman_urdu_probability = calculateProbability(processed_text, rmu_v)
    actual_urdu_probability = calculateProbability(processed_text, acu_v)
    max_prob = max(english_probability, roman_urdu_probability,
                   actual_urdu_probability)
    if max_prob == english_probability:
        return 'Eng'
    elif max_prob == roman_urdu_probability:
        return 'RmUr'
    elif max_prob == actual_urdu_probability:
        return 'Ur'


def checking(text):
    language_detected = language_detection(text)
    msg_class = ""
    if(language_detected == 'Eng'):
        # use English model
        pretext = preprocess(text)
        msg_class = findClass(pretext, english_model, english_vectorizer)
    elif (language_detected == 'Ur'):
        # use urdu model
        msg_class = findClass(text, actual_urdu_model, actual_urdu_vectorizer)
    elif (language_detected == 'RmUr'):
        # use roman urdu model
        pretext = preprocess(text)
        msg_class = findClass(pretext, roman_urdu_model, roman_urdu_vectorizer)
    return msg_class


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = checking(data)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
