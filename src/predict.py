import joblib
from utils import preprocess

# Load model and vectorizer
model = joblib.load('../model.pkl')
vectorizer = joblib.load('../vectorizer.pkl')

# Sample inputs
texts = [
    "Bonjour, je suis heureux",
    "Hello, I am learning Python",
    "नमस्ते, मेरा नाम राम है",
    "Ciao, come va?"
]

for text in texts:
    text_proc = preprocess(text)
    text_vec = vectorizer.transform([text_proc])
    prediction = model.predict(text_vec)
    print(f"Text: {text} -> Predicted language: {prediction[0]}")
