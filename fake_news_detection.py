import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")  # Ensure this file is in the same directory

# Preprocessing function
def preprocess(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(preprocess)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# ROC AUC Score
proba = model.predict_proba(X_test_vec)[:, 1]
y_true_binary = y_test.map({ "FAKE": 0, "REAL": 1 })
print("ROC AUC Score:", roc_auc_score(y_true_binary, proba))

# Real-time predictions
def predict_news(news_text):
    processed = preprocess(news_text)
    vect = vectorizer.transform([processed])
    return model.predict(vect)[0]

# Example predictions
example_news = [
    "Aliens landed in New York and took over the city!",
    "The government passed a new climate change bill to reduce emissions."
]

for article in example_news:
    prediction = predict_news(article)
    print(f"News: {article}\nPrediction: {prediction}\n")
