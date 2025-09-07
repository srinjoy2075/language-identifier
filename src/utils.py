import re

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation (optional)
    text = re.sub(r'[^\w\s]', '', text)
    return text
