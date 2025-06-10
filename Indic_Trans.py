import os
import time
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
from streamlit.components.v1 import html

# Environment optimizations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration - using the correct language codes expected by the model (with script information)
TRANSLATION_TAGS = {
    "English": "eng_Latn",
    "Hindi (हिन्दी)": "hin_Deva",
    "Bengali (বাংলা)": "ben_Beng",
    "Tamil (தமிழ்)": "tam_Taml",
    "Telugu (తెలుగు)": "tel_Telu",
    "Gujarati (ગુજરાતી)": "guj_Gujr",
    "Kannada (ಕನ್ನಡ)": "kan_Knda",
    "Malayalam (മലയാളം)": "mal_Mlym",
    "Marathi (मराठी)": "mar_Deva",
    "Punjabi (ਪੰਜਾਬੀ)": "pan_Guru",
    "Urdu (اردو)": "urd_Arab",
    "Odia (ଓଡ଼ିଆ)": "ory_Orya",
    "Assamese (অসমীয়া)": "asm_Beng",
    "Nepali (नेपाली)": "npi_Deva",
    "Sanskrit (संस्कृतम्)": "san_Deva",
    "Kashmiri (کٲشُر)": "kas_Arab",
    "Sindhi (سنڌي)": "snd_Arab",
    "Maithili (मैथिली)": "mai_Deva",
    "Manipuri (মণিপুরী)": "mni_Beng",
    "Bodo (बर')": "brx_Deva",
    "Santali (ᱥᱟᱱᱛᱟᱲᱤ)": "sat_Olck",
    "Konkani (कोंकणी)": "gom_Deva",
    "Dogri (डोगरी)": "doi_Deva"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_name(src_lang, tgt_lang):
    if src_lang == tgt_lang:
        st.warning("Please select different source and target languages.")
        st.stop()

    if src_lang == "English":
        return "ai4bharat/indictrans2-en-indic-dist-200M"
    elif tgt_lang == "English":
        return "ai4bharat/indictrans2-indic-en-dist-200M"
    else:
        st.error("Only English ↔ Indic translation is supported with current models.")
        st.stop()


@st.cache_resource(show_spinner="Loading translation models...")
def load_models(src_lang, tgt_lang):
    try:
        model_name = get_model_name(src_lang, tgt_lang)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
        processor = IndicProcessor(inference=True)
        return tokenizer, model, processor
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()


def translate_text(text, src_lang, tgt_lang, tokenizer, model, processor):
    if not text.strip():
        return ""

    try:
        if src_lang not in TRANSLATION_TAGS or tgt_lang not in TRANSLATION_TAGS:
            return f"Error: Invalid language tag(s): src='{src_lang}', tgt='{tgt_lang}'"

        src_code = TRANSLATION_TAGS[src_lang]
        tgt_code = TRANSLATION_TAGS[tgt_lang]

        batch = processor.preprocess_batch([text.strip()], src_lang=src_code, tgt_lang=tgt_code)
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        with torch.no_grad():
            output = model.generate(**inputs, num_beams=4, max_length=256, early_stopping=True)

        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        final_output = processor.postprocess_batch(decoded, lang=tgt_code)[0]
        return final_output

    except Exception as e:
        return f"Error: {str(e)}"

    
LANGUAGES = [
    "English",
    "Hindi (हिन्दी)",
    "Bengali (বাংলা)",
    "Tamil (தமிழ்)",
    "Telugu (తెలుగు)",
    "Gujarati (ગુજરાતી)",
    "Kannada (ಕನ್ನಡ)",
    "Malayalam (മലയാളം)",
    "Marathi (मराठी)",
    "Punjabi (ਪੰਜਾਬੀ)",
    "Urdu (اردو)",
    "Odia (ଓଡ଼ିଆ)",
    "Assamese (অসমীয়া)",
    "Nepali (नेपाली)",
    "Sanskrit (संस्कृतम्)",
    "Kashmiri (کٲشُر)",
    "Sindhi (سنڌي)",
    "Maithili (मैथिली)",
    "Manipuri (মণিপুরী)",
    "Bodo (बर')",
    "Santali (ᱥᱟᱱᱛᱟᱲᱤ)",
    "Konkani (कोंकणी)",
    "Dogri (डोगरी)"
]

def main():
    st.set_page_config(
        page_title="Indic Translator",
        page_icon="🇮🇳",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .title {color: #4b8bbe; text-align: center; font-size: 2.5rem}
    .subtitle {color: #666; text-align: center}
    .stButton>button {background-color: #4b8bbe; color: white}
    .translation-box {
        background: #f5f9ff;
        padding: 15px;
        border-radius: 10px;
        color: #000000;  /* <-- Ensures visible text */
        font-size: 1.1rem;
    }
    .error-box {
        background: #fff0f0;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff4b4b;
        color: #990000;  /* Better visibility for errors */
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title">🇮🇳 Indic Language Translator</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Bidirectional translation between English and 22 Indian languages</p>', unsafe_allow_html=True)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Language")
        src_lang = st.selectbox(
            "Select source language",
            options=LANGUAGES,
            index=0,
            key="source_language_select"
        )

        src_text = st.text_area(
            "Enter text to translate",
            height=150,
            placeholder="Type or paste your text here..."
        )

    with col2:
        st.subheader("Target Language")
        tgt_lang = st.selectbox(
            "Select target language",
            options=LANGUAGES,
            index=1,
            key="target_language_select"
        )

        if st.button("Translate", type="primary"):
            if src_text.strip():
                with st.spinner("Loading models..."):
                    tokenizer, model, processor = load_models(src_lang, tgt_lang)

                with st.spinner("Translating..."):
                    start_time = time.time()
                    translation = translate_text(
                        text=src_text,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        tokenizer=tokenizer,
                        model=model,
                        processor=processor
                    )
                    duration = time.time() - start_time

                st.subheader("Translation Result")
                if translation.startswith("Error:"):
                    st.markdown(f'<div class="error-box">{translation}</div>', unsafe_allow_html=True)
                else:
                    # Escape translation text for safe HTML rendering
                    safe_translation = translation.replace('"', '&quot;').replace("'", "&#39;")

                    # Display the result with a copy button
                    html(f"""
                        <div id="translation-container" style="
                            background: #f5f9ff;
                            padding: 15px;
                            border-radius: 10px;
                            color: #000000;
                            font-size: 1.1rem;
                            margin-bottom: 10px;
                        ">{safe_translation}</div>
                        <button onclick="copyTranslation()" style="
                            background-color:#4CAF50;
                            color:white;
                            padding:8px 12px;
                            border:none;
                            border-radius:5px;
                            cursor:pointer;
                        ">📋 Copy Translation</button>

                        <script>
                        function copyTranslation() {{
                            const text = document.getElementById("translation-container").innerText;
                            navigator.clipboard.writeText(text).then(() => {{
                                alert("✅ Translation copied to clipboard!");
                            }});
                        }}
                        </script>
                    """, height=200)

                    st.caption(f"Translated in {duration:.2f} seconds")

            else:
                st.warning("Please enter text to translate")

    # Add info section
    with st.expander("ℹ️ About this translator"):
        st.markdown("""
        - **Model**: AI4Bharat IndicTrans2 (200M parameter distilled version)
        - **Languages**: 22 Indian languages + English
        - **Bidirectional**: Supports both English→Indian and Indian→English
        - **Optimized**: Runs on both CPU and GPU

        For best results:
        - Keep sentences under 100 words
        - Use clear, grammatical text
        - Avoid slang and idioms
        """)
    if st.button("🧹 Clear cache & session"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    main()
