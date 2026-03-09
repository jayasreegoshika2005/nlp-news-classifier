import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import preprocess_text

# Load dataset
data = pd.read_csv("dataset/news.csv")

# Combine Title and Description into one text column
data["text"] = data["Title"] + " " + data["Description"]

# Target labels
data["category"] = data["Class Index"]

# Preprocess text
data["clean_text"] = data["text"].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"],
    data["category"],
    test_size=0.2,
    random_state=42
)

# Convert text to numerical vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Training Complete ✅")