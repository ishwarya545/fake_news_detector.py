import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv('dataset.csv')
df = df[['text', 'label']]

# Preprocessing
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['text'] = df['text'].apply(clean_text)

# Split data
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

# Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Prediction
y_pred = model.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f'Accuracy: {round(score * 100, 2)}%')
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
