import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from utils import preprocess

# -----------------------------
# Resolve paths safely
# -----------------------------
HERE = os.path.dirname(__file__)  # directory of this script (src/)
DATA_PATH = os.path.join(HERE, "..", "data", "language_dataset.csv")
MODELS_DIR = os.path.join(HERE, "..", "models")

# Make sure models/ folder exists
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Drop rows with missing text or language
df = df.dropna(subset=["text", "language"])

# Preprocess text
df["text"] = df["text"].apply(preprocess)

# -----------------------------
# Features and labels
# -----------------------------
X = df["text"]
y = df["language"]

# Convert text to numerical features using character n-grams
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train classifier
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Evaluate accuracy
# -----------------------------
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Save model and vectorizer
# -----------------------------
joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))

print("✅ Model and vectorizer saved in models/ folder")
