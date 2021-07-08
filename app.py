from flask import Flask, render_template, url_for, request
import pickle
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

file1 = open('lang_vectors_2.pkl', 'rb')
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
    x = vect.transform([text]).toarray()
    p = modelName.predict(x)
    a = int(p[0])
    return a


def preprocess(text):
    text = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'http', text)
    text = re.sub('£|\$', 'money', text)
    text = re.sub(
        '\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'number', text)
    text = re.sub('\d+(\.\d+)?', 'numbr', text)
    text = re.sub('[^\w\d\s]', ' ', text)
    text = text.lower()
    return text


def calculateProbability(text_message,english_text,roman_urdu_text,actual_urdu_text):
  splitted_message = text_message.split()
  found_eng = found_rur = found_acu = found_oth = 0
  for i in range(len(splitted_message)):
    ttext = splitted_message[i]
    if ttext in english_text:
      found_eng +=1
    elif ttext in roman_urdu_text:
      found_rur +=1
    elif ttext in actual_urdu_text:
      found_acu +=1
    else:
      found_oth +=1
  prob_eng = float(found_eng)/ len(splitted_message)
  prob_rur = float(found_rur)/ len(splitted_message)
  prob_acu = float(found_acu)/ len(splitted_message)
  prob_oth = float(found_oth)/ len(splitted_message)
  return prob_eng, prob_rur, prob_acu, prob_oth


def language_detection(text):
    processed_text =  preprocess(text)
    ep,rup,aup,olp = calculateProbability(processed_text,eng_v,rmu_v,acu_v)
    max_prob = max(ep,rup,aup,olp) 
    if max_prob == ep:
      return 'Eng'
    elif max_prob == rup:
      return 'RmUr'
    elif max_prob == aup:
      return 'Ur'
    elif max_prob == olp:
      return 'Other'


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
    elif (language_detected == 'Other'):
        msg_class = 2
    return msg_class


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/tell', methods=['POST'])
def tell():
    if request.method == 'POST':
        message = request.form['message']
        my_prediction = checking(message)
    return str(my_prediction)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        my_prediction = checking(message)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
