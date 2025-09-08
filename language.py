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
    lang_name = get_language_name(lang_code)
    return lang_name

if __name__ == "__main__":
    while True:
        text = input("Enter a sentence (or type 'exit' to quit): ")
        if text.lower() == "exit":
            break
        lang = predict_language(text)
        print(f"{text} → {lang}")
