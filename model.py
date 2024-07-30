import pandas as pd
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df=pd.read_csv('spam.csv',encoding='latin-1')

x=df["v2"]
y=df["v1"]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer(max_features=2500)
x=tv.fit_transform(x).toarray()

pickle.dump(tv,open("tfidf_vectorizer.pkl","wb"))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=1)

from sklearn.svm import SVC
model=SVC(kernel='linear')
model.fit(x_train,y_train)

pickle.dump(model,open("model.pkl", "wb"))