import os
import streamlit as st
import joblib
import pandas as pd
from utils import preprocess

# -----------------------------
# Paths
# -----------------------------
HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "..", "models")

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Language Identifier", page_icon="🌍")

st.title("🌍 Language Identifier")
st.write("Enter a sentence and I’ll predict the language!")

# --- Sample sentences for quick testing ---
samples = {
    "English 🇬🇧": "Hello, how are you?",
    "French 🇫🇷": "Bonjour, comment ça va?",
    "Spanish 🇪🇸": "Hola amigo, ¿cómo estás?",
    "German 🇩🇪": "Wie geht es dir?",
    "Hindi 🇮🇳": "नमस्ते, आप कैसे हैं?",
    "Italian 🇮🇹": "Ciao, come stai?"
}

option = st.selectbox("Or pick a sample sentence:", ["-- Select --"] + list(samples.keys()))
if option != "-- Select --":
    user_input = samples[option]
else:
    user_input = st.text_area("Type your text here:", "")

# Initialize history in session_state
if "history" not in st.session_state:
    st.session_state.history = []

# Predict button
if st.button("Predict Language"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text or pick a sample.")
    else:
        # Preprocess and predict
        text_proc = preprocess(user_input)
        text_vec = vectorizer.transform([text_proc])

        prediction = model.predict(text_vec)[0]
        probs = model.predict_proba(text_vec)[0]  # probabilities for all classes

        # Display result
        st.success(f"✅ Predicted Language: **{prediction}**")

        # Create probability table
        lang_probs = pd.DataFrame({
            "Language": model.classes_,
            "Probability": probs
        }).sort_values(by="Probability", ascending=False)

        # Show chart
        st.write("### Confidence Scores")
        st.bar_chart(lang_probs.set_index("Language"))

        # Save to history
        st.session_state.history.append({
            "Input": user_input,
            "Prediction": prediction,
            "Top Probability": round(lang_probs.iloc[0]["Probability"], 3)
        })

# Show history
if st.session_state.history:
    st.write("### 📜 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))
