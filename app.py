import numpy as np
import pickle
from flask import Flask, request, render_template

app=Flask(__name__)

model=pickle.load(open("model.pkl","rb"))
tv=pickle.load(open("tfidf_vectorizer.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message=request.form['message']
    data=[message]
    features = tv.transform(data).toarray()
    prediction=model.predict(features)
    output = 'spam' if prediction[0] == 1 else 'not spam'
    return render_template("index.html",prediction_text="The message is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)