import os
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
from docx import Document

# IBM TTS
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# IBM watsonx.ai
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model

# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="EchoVerse", page_icon="🎧", layout="wide")

st.title("🎧 EchoVerse - AI Audiobook Creator")
st.caption("Enter text in English → choose tone → select output language → listen or download.")

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
        params={"max_new_tokens": 400, "temperature": 0.7, "decoding_method": "sample"},
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
        return text + " (⚠️ Watsonx not configured, showing original)"

    try:
        result = model.generate_text(prompt=text)
        st.write("Raw Watsonx Response:", result)  # 👈 Show full output

        if isinstance(result, dict):
            rewritten = result.get("generated_text", "").strip()
            if not rewritten:
                raise ValueError("Watsonx returned empty text.")
            return rewritten
        elif isinstance(result, str):
            return result.strip()
        else:
            return str(result).strip()

    except Exception as e:
        st.error(f"❌ Watsonx Error: {e}")
        return text + " (⚠️ Rewrite failed)"


def translate_text(text: str, target_lang: str) -> str:
    """Translate rewritten English text into target language."""
    model = get_watsonx_model()
    if model is None:
        return text + f" (⚠️ Watsonx not configured, kept in English)"

    prompt = f"""
Translate this English text into {target_lang} for audiobook narration.
Do not explain or add comments, only give translated text.

<<<TEXT>>>
{text}
<<<END>>>
"""

    try:
        result = model.generate_text(prompt=prompt)

        # Debugging: show response in Streamlit
        st.write("🔎 Translation Response:", result)

        if isinstance(result, dict):
            return (result.get("generated_text") or "").strip()
        elif isinstance(result, str):
            return result.strip()
        else:
            return str(result).strip()

    except Exception as e:
        st.error(f"❌ Error translating text: {e}")
        return text + f" (⚠️ Translation failed, kept in English)"


def translate_text(text: str, target_lang: str) -> str:
    """Translate rewritten English text into target language."""
    model = get_watsonx_model()
    if model is None:
        return text
    prompt = f"""
You are a professional translator. Translate the following English text into {target_lang}.
Keep meaning faithful and natural for audiobook narration.
Do not explain or add comments, only give translated text.

<<<TEXT>>>
{text}
<<<END>>>
"""
    try:
        result = model.generate_text(prompt=prompt)
        return (result.get("generated_text") or "").strip() if isinstance(result, dict) else str(result).strip()
    except Exception:
        return text

def speak_ibm_tts(text: str, voice: str) -> bytes:
    """Generate speech in chosen voice."""
    tts = get_tts_client()
    if tts is None or not text.strip():
        return b""
    try:
        res = tts.synthesize(text=text.strip(), voice=voice, accept="audio/mp3").get_result()
        return res.content
    except Exception:
        return b""

# ---------- Input ----------
user_text = st.text_area("✍️ Enter text in English", height=200, placeholder="Type or paste your English text here...")

# ---------- Options ----------
tone = st.selectbox("🎚️ Choose tone", ["Neutral", "Suspenseful", "Inspiring"])

languages = {
    "English (US)": ["en-US_AllisonV3Voice", "en-US_LisaV3Voice", "en-US_MichaelV3Voice"],
    "English (UK)": ["en-GB_CharlotteV3Voice", "en-GB_JamesV3Voice", "en-GB_KateV3Voice"],
    "Spanish": ["es-ES_EnriqueV3Voice", "es-ES_LauraV3Voice", "es-LA_SofiaV3Voice"],
    "French": ["fr-FR_ReneeV3Voice"],
    "German": ["de-DE_DieterV3Voice", "de-DE_BirgitV3Voice"],
    "Italian": ["it-IT_FrancescaV3Voice"],
    "Portuguese (Brazil)": ["pt-BR_IsabelaV3Voice"],
    "Japanese": ["ja-JP_EmiV3Voice"],
    "Arabic": ["ar-MS_OmarVoice"]
}

lang = st.selectbox("🌍 Output language", list(languages.keys()))
voice = st.selectbox("🗣️ Voice", languages[lang])

gen = st.button("✨ Generate Audiobook", type="primary", disabled=not bool(user_text.strip()))

# ---------- History ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Processing ----------
if gen and user_text.strip():
    with st.spinner("Rewriting in chosen tone..."):
        progress_bar = st.progress(0)
        rewritten = rewrite_with_tone(user_text, tone)
        progress_bar.progress(30)

    final_text = rewritten
    if not lang.startswith("English"):
        with st.spinner(f"Translating English → {lang}..."):
            final_text = translate_text(rewritten, lang)
        progress_bar.progress(60)
    else:
        progress_bar.progress(60)

    with st.spinner("Generating narration..."):
        audio_bytes = speak_ibm_tts(final_text, voice)
        progress_bar.progress(100)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button("⬇️ Download MP3", data=audio_bytes, file_name="echoverse_narration.mp3", mime="audio/mp3")

        st.session_state.history.append({
            "original": user_text,
            "rewritten": rewritten,
            "translated": final_text,
            "language": lang,
            "tone": tone,
            "voice": voice,
            "audio": audio_bytes
        })
        st.success("✅ Audiobook ready!")

# ---------- History Display ----------
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 History")

    for i, item in enumerate(reversed(st.session_state.history), start=1):
        with st.expander(f"{i}. {item['language']} | {item['tone']} | {item['voice']}"):
            st.markdown("**Original (English)**")
            st.markdown(item["original"][:500] + ("..." if len(item["original"]) > 500 else ""))

            st.markdown("**Rewritten in English**")
            st.markdown(item["rewritten"][:500] + ("..." if len(item["rewritten"]) > 500 else ""))

            if not item["language"].startswith("English"):
                st.markdown(f"**Translated → {item['language']}**")
                st.markdown(item["translated"][:500] + ("..." if len(item["translated"]) > 500 else ""))

            st.audio(item["audio"], format="audio/mp3")
            st.download_button(
                f"⬇️ Download Narration {i}",
                data=item["audio"],
                file_name=f"echoverse_history_{i}.mp3",
                mime="audio/mp3"
            )

