#front-end section

from flask import Flask, render_template, request

app = Flask(__name__)


# data science section

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

def wordopt (text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W", " ", text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text


from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()

@app.route("/")
def home():
    data_fake = pd.read_csv('Fake.csv')
    data_true = pd.read_csv('True.csv')

    data_fake["class"]= 0
    data_true["class"] = 1

    for i in range (23480,23470, -1):
        data_fake.drop([i], axis = 0, inplace = True)
    
    for i in range (21416,21406 -1):
        data_true.drop([i], axis = 0, inplace = True)
    

    data_merge = pd.concat([data_fake, data_true], axis = 0)
    data = data_merge.drop(['title', 'subject', 'date'], axis = 1)
    data = data.sample(frac =1)
    data.reset_index(inplace = True)
    data.drop(['index'], axis = 1, inplace = True)
    
    
    data['text'] = data['text'].apply(wordopt)
    x = data['text']
    y = data['class']
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)


    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)
    
    LR.fit(xv_train, y_train)

    return render_template("fake-news.html")

@app.route("/predict", methods = ["GET","POST"])
def predict():
    
    def output_label(n):
        if n == 0:
            return "FAKE NEWS!"
        elif n == 1:
            return "THIS NEWS IS LEGIT"

    def manual_testing(news):
        testing_news = {"text":[news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        pred_LR = LR.predict(new_xv_test)
        return output_label(pred_LR[0])

    news = str(request.form['news-input'])
    result = manual_testing(news)
    return render_template("result.html", result = result)

if __name__ == "__main__":
    app.run(debug=True)