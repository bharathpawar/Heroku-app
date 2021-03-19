import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB



app = Flask(__name__)



tfidf = pickle.load(open('tfidf_model.sav', 'rb'))
model = pickle.load(open('finalized_model.sav','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    current_title= list(request.form['title'])
    
    text_feats = tfidf.transform(current_title)
    prediction = model.predict(text_feats)[0]
    
    

    if prediction == 1:
        return render_template('Cxo.html')
    else:
        return render_template('Not_Cxo.html')

    

if __name__ == "__main__":
    app.run(debug=True)

    