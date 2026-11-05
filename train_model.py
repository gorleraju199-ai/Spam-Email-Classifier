import pandas as pd
import re
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Identify text and label columns
if {"v1", "v2"}.issubset(data.columns):
    label_col, text_col = "v1", "v2"
elif {"Category", "Message"}.issubset(data.columns):
    label_col, text_col = "Category", "Message"
elif {"label", "text"}.issubset(data.columns):
    label_col, text_col = "label", "text"
elif {"category", "message"}.issubset(data.columns):
    label_col, text_col = "category", "message"
else:
    raise ValueError(
        "Dataset columns not recognized. "
        "Please check your CSV column names."
    )

data = data[[label_col, text_col]].dropna()
data.columns = ["label", "message"]

# Convert labels to spam and ham
data["label"] = data["label"].astype(str).str.strip().str.lower()

label_map = {
    "spam": "spam",
    "ham": "ham",
    "1": "spam",
    "0": "ham"
}

data["label"] = data["label"].map(label_map)
data = data.dropna()

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["message"] = data["message"].apply(clean_text)
data = data[data["message"] != ""]

# Split dataset
X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
print("MODEL EVALUATION")
print("-------------------------")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(
    y_test, y_pred, pos_label="spam", zero_division=0
))
print("Recall:", recall_score(
    y_test, y_pred, pos_label="spam", zero_division=0
))
print("F1 Score:", f1_score(
    y_test, y_pred, pos_label="spam", zero_division=0
))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save trained model
Path("model").mkdir(exist_ok=True)

with open("model/spam_classifier.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel trained and saved successfully!")
