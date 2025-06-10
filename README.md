🇮🇳 Indic Language Translator
A Streamlit-based language translation app for bi-directional translation between English and 22 Indian languages using AI4Bharat's IndicTrans2 models.

🌐 Live Demo
Coming soon – host using Streamlit Sharing or Hugging Face Spaces.

🚀 Features
🔁 Bidirectional Translation: English ↔ Indic (22 Indian languages)

🤖 Transformer-based Models: Powered by AI4Bharat’s IndicTrans2 distilled models (200M parameters)

⚡ Optimized: Automatically utilizes GPU (if available) or CPU

🖥️ Interactive UI: Built with Streamlit for a smooth experience

📚 Language Tags Included: Uses correct script and language codes

🧠 Context-Aware Translation: Processes and cleans text pre/post translation

🧪 Supported Languages
English	Hindi	Bengali	Tamil	Telugu	Gujarati
Kannada	Malayalam	Marathi	Punjabi	Urdu	Odia
Assamese	Nepali	Sanskrit	Kashmiri	Sindhi	Maithili
Manipuri	Bodo	Santali	Konkani	Dogri	

Script support included for Devanagari, Tamil, Telugu, Bengali, etc.

📦 Installation
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/indic-language-translator.git
cd indic-language-translator
2. Create a virtual environment (recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
📥 Download Required Models
The model will be automatically downloaded from Hugging Face when first used:

ai4bharat/indictrans2-en-indic-dist-200M

ai4bharat/indictrans2-indic-en-dist-200M

Make sure you have internet access for the first run.

To download them manually (optional):

python
Copy
Edit
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
🖥️ Run the App Locally
bash
Copy
Edit
streamlit run app.py
You should now see the app running at http://localhost:8501

🛠 Project Structure
bash
Copy
Edit
indic-language-translator/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependency list
└── README.md              # Project documentation
🧼 Optional Cache & Session Reset
Click the "🧹 Clear cache & session" button inside the app if you face stale model/data issues.

🧠 Model & Toolkit Info
Models: IndicTrans2 Distilled (200M)

Toolkit: IndicTransToolkit

✅ Tips for Best Results
Limit input to ~100 words

Avoid informal, slang-heavy text

Ensure correct script/language pair is selected

Uses beam search decoding (num_beams=4)

📄 License
This project uses models provided by AI4Bharat, which are open for research and non-commercial use under their license terms.

🙏 Acknowledgements
AI4Bharat for IndicTrans2

Hugging Face Transformers

Streamlit for the UI

