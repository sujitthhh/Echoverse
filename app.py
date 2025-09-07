import os
import streamlit as st
from dotenv import load_dotenv

# IBM TTS
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# IBM watsonx.ai
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model

# ---------- Setup ----------
load_dotenv()

# watsonx.ai creds (from .env file)
WX_API_KEY = os.getenv("WATSONX_API_KEY")
WX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# TTS creds (from .env file)
TTS_API_KEY = os.getenv("TTS_API_KEY")
TTS_URL = os.getenv("TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")

st.set_page_config(page_title="EchoVerse", page_icon="🎧", layout="centered")
st.title("🎧 EchoVerse — AI Audiobook Creator (By TechElite)")
st.caption("Paste or upload text → choose tone → rewrite with IBM watsonx.ai → speak with IBM Text-to-Speech → listen or download MP3.")

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
    """Uses watsonx.ai to rewrite text in a chosen tone while preserving meaning."""
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

def speak_ibm_tts(text: str, voice: str = "Allison") -> bytes:
    """Synthesizes speech using IBM Text to Speech and returns MP3 bytes."""
    tts = get_tts_client()
    if tts is None or not text.strip():
        return b""

    try:
        res = tts.synthesize(
            text=text.strip(),
            voice=voice,
            accept="audio/mp3"
        ).get_result()
        return res.content
    except Exception:
        return b""

# ---------- UI ----------
tab1, tab2 = st.tabs(["Paste text", "Upload .txt"])

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

if gen and user_text:
    with st.spinner("Rewriting..."):
        rewritten = rewrite_with_tone(user_text, tone)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.markdown(user_text)
    with col2:
        st.subheader(f"{tone} rewrite")
        st.markdown(rewritten)

    with st.spinner("Creating narration ..."):
        audio_bytes = speak_ibm_tts(rewritten, voice=voice)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "⬇️ Download MP3",
            data=audio_bytes,
            file_name="echoverse_narration.mp3",
            mime="audio/mpeg",
        )
    else:
        st.warning("⚠️ No audio generated. Check your TTS setup.")
