import streamlit as st
import langid

# ISO codes to full names
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "hi": "Hindi",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic",
    "cy": "Welsh"
    # add more if needed
}

def get_language_name(lang_code):
    return LANGUAGE_NAMES.get(lang_code, f"Unknown ({lang_code})")

def predict_language(text):
    lang_code, _ = langid.classify(text)
    return get_language_name(lang_code)

# --- Streamlit UI ---
st.set_page_config(page_title="Language Identifier", layout="centered")
st.title("🌐 Language Identifier")

user_input = st.text_area("Enter text here:")

if st.button("Detect Language"):
    if user_input.strip():
        lang = predict_language(user_input)
        st.success(f"**Detected Language:** {lang}")
    else:
        st.warning("Please enter some text!")
