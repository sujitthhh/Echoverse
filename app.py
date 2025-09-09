import os
import time
import streamlit as st
from dotenv import load_dotenv
import PyPDF2   # 📘 for reading PDFs
import docx     # 📘 for reading Word files
from docx import Document
from deep_translator import GoogleTranslator

# IBM TTS
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# IBM watsonx.ai
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model


# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="EchoVerse", page_icon="🎧", layout="centered")

st.title("🎧 EchoVerse — AI Audiobook Creator (By TechElite)")
st.caption("Paste or upload text → choose tone → choose voice → listen or download.")

# ---------- Credentials ----------
WX_API_KEY = os.getenv("WATSONX_API_KEY")
WX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

TTS_API_KEY = os.getenv("TTS_API_KEY")
TTS_URL = os.getenv("TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")


# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def get_watsonx_model():
    if not (WX_API_KEY and WX_URL and WX_PROJECT_ID):
        return None
    creds = Credentials(api_key=WX_API_KEY, url=WX_URL)
    return Model(
        model_id="ibm/granite-13b-instruct-v2",
        params={
            "max_new_tokens": 300,
            "temperature": 0.7,
            "decoding_method": "sample",
        },
        credentials=creds,
        project_id=WX_PROJECT_ID,
    )


@st.cache_resource(show_spinner=False)
def get_tts_client():
    if not (TTS_API_KEY and TTS_URL):
        return None
    auth = IAMAuthenticator(TTS_API_KEY)
    client = TextToSpeechV1(authenticator=auth)
    client.set_service_url(TTS_URL)
    return client


def rewrite_with_tone(text: str, tone: str) -> str:
    model = get_watsonx_model()
    if model is None:
        return text  # fallback

    system = (
        "You rewrite user text in a specified tone while keeping the original meaning. "
        "Keep the output concise and suitable for narration. Do not add new facts."
    )
    tone_instructions = {
        "Neutral": "Use a neutral, clear, informative tone with smooth flow.",
        "Suspenseful": "Increase tension and anticipation; vary sentence length; end some lines with subtle hooks.",
        "Inspiring": "Make it uplifting and motivational; use positive, energetic language and forward momentum.",
    }

    prompt = f"""{system}

TONE: {tone}
TONE NOTES: {tone_instructions[tone]}

Rewrite the following text faithfully to the meaning while adapting the tone:

<<<TEXT>>>
{text}
<<<END>>>"""

    try:
        result = model.generate_text(prompt=prompt)
        rewritten = ""
        if isinstance(result, dict):
            rewritten = result.get("generated_text", "").strip()
        elif isinstance(result, str):
            rewritten = result.strip()
        return rewritten if rewritten else text
    except Exception:
        return text


def speak_ibm_tts(text: str, voice: str = "en-US_AllisonV3Voice", audio_format="mp3") -> bytes:
    """Synthesizes speech using IBM TTS and returns audio bytes."""
    tts = get_tts_client()
    if tts is None or not text.strip():
        st.error("❌ TTS client not initialized or empty text.")
        return b""

    try:
        mime_type = f"audio/{audio_format}"
        file_ext = audio_format
        res = tts.synthesize(
            text=text.strip(),
            voice=voice,
            accept=mime_type
        ).get_result()
        return res.content
    except Exception as e:
        st.error(f"❌ TTS error: {str(e)}")
        return b""


# ---------- Input Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["✍️ Paste text", "📂 Upload .txt", "📄 Upload .pdf", "📘 Upload .docx"])

user_text = ""

with tab1:
    user_text = st.text_area("📖 Enter your text", height=200, placeholder="Type or paste your story/article here...")

with tab2:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded is not None:
        try:
            file_text = uploaded.read().decode("utf-8")
        except UnicodeDecodeError:
            file_text = uploaded.read().decode("latin-1")
        user_text = file_text

with tab3:
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        reader = PyPDF2.PdfReader(pdf_file)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text() + "\n"
        user_text = pdf_text

with tab4:
    docx_file = st.file_uploader("Upload a Word file", type=["docx"])
    if docx_file is not None:
        doc = docx.Document(docx_file)
        doc_text = "\n".join([para.text for para in doc.paragraphs])
        user_text = doc_text


# ---------- Options ----------
tone = st.selectbox("🎚️ Choose tone", ["Neutral", "Suspenseful", "Inspiring"])

voice = st.selectbox(
    "🗣️ Choose voice",
    [
        "en-US_AllisonV3Voice",
        "en-US_LisaV3Voice",
        "en-US_MichaelV3Voice",
        "en-GB_CharlotteV3Voice",
        "en-GB_JamesV3Voice",
        "en-GB_KateV3Voice",
    ],
    index=0,
    help="Select voices (more can be added later)."
)

gen = st.button("✨ Rewrite & Generate Audio", type="primary", disabled=not user_text)

# ---------- Processing ----------
if gen and user_text:
    with st.spinner("Rewriting with selected tone..."):
        progress_bar = st.progress(0)
        rewritten = rewrite_with_tone(user_text, tone)
        progress_bar.progress(50)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.markdown(user_text)
    with col2:
        st.subheader(f"{tone} Rewrite")
        st.markdown(rewritten)

    with st.spinner("Generating narration..."):
        audio_bytes = speak_ibm_tts(rewritten, voice=voice, audio_format="mp3")
        progress_bar.progress(100)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "⬇️ Download MP3",
            data=audio_bytes,
            file_name="echoverse_narration.mp3",
            mime="audio/mp3",
        )
        st.success("✅ Audio ready!")
    else:
        st.warning("⚠️ No audio generated. Check your TTS setup.")
