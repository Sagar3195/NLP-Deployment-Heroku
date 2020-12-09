##SMS Spam Classification Using Lemmatization

import pandas as pd
import numpy as np

#loading spam classifier datasets
data = pd.read_csv("spam.csv", encoding='latin-1')

print(data.head())

print(data.columns)

print(data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace=True))

print(data.shape)

print(data.head())

data['class'].value_counts(normalize = True)

#Checking missing values in dataset
print(data.isnull().sum())

#Data cleanning and preprocesssing :

import nltk
import re

from nltk.corpus import stopwords #removing words which is not required
from nltk.stem   import WordNetLemmatizer #lemmatization


#create objecct for lemmatization
lemmatizer = WordNetLemmatizer()


corpus = []
for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)
#corpus

print(len(corpus))

#Now we create model for Bag of words

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features= 5000)

#create independent variable
X = vectorizer.fit_transform(corpus)


X = X.toarray() #independent variable
print(X.shape)
#Now dependent variable

y = pd.get_dummies(data['class'])

print(y.head())

y = y.iloc[:, 1].values

print(y.shape)

#Now splitting dataset into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

print(x_train.shape, x_train.shape)

#now create classification model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

#fit the model and then predict the model
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#y_pred


#Now checking accuracy of the model
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model: ", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix of the model: \n",cm)

import joblib
#joblib.dump(model, 'spam_message.pkl')

#joblib.dump(vectorizer, 'vector_transform.pkl')

