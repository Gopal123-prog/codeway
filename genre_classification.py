import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



description = []
genre = []
test_description = []

with open('C:\Users\gopal\Downloads\codeway\archive\Genre Classification Dataset\train_data.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        line = line.split(":::")
        description.append(line[3])
        genre.append(line[2])

with open('/content/test_data.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        line = line.split(":::")
        test_description.append(line[2])


df = pd.DataFrame({"Descriptions": description, "Genres": genre})

le = LabelEncoder()
df['Genres'] = le.fit_transform(df['Genres'])
X = df['Descriptions']
y = df['Genres']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)
test_transform = vectorizer.transform(test_description)

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_transformed, y_train)
naive_bayes_predictions = naive_bayes_classifier.predict(X_test_transformed)


logistic_regression_classifier = LogisticRegression()
logistic_regression_classifier.fit(X_train_transformed, y_train)
logistic_regression_predictions = logistic_regression_classifier.predict(X_test_transformed)

svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_transformed, y_train)
svm_predictions = svm_classifier.predict(X_test_transformed)

print("Naive Bayes Accuracy:", accuracy_score(y_test, naive_bayes_predictions))
print(classification_report(y_test, naive_bayes_predictions))

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, logistic_regression_predictions))
print(classification_report(y_test, logistic_regression_predictions))

print("\nSVM Accuracy:", accuracy_score(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions))

test_predictions_nb = naive_bayes_classifier.predict(test_transform)
test_predictions_lr = logistic_regression_classifier.predict(test_transform)
test_predictions_svm = svm_classifier.predict(test_transform)

print("Test Predictions (Naive Bayes):", le.inverse_transform(test_predictions_nb))
print("Test Predictions (Logistic Regression):", le.inverse_transform(test_predictions_lr))
print("Test Predictions (SVM):", le.inverse_transform(test_predictions_svm))

vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)
test_transform = vectorizer.fit_transform(test_description)

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_transformed, y)

predictions = naive_bayes_classifier.predict(test_transform)

accuracy = accuracy_score(df['Genres'], predictions)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(df['Genres'], predictions))

