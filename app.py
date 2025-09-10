import os
import streamlit as st
from dotenv import load_dotenv
import PyPDF2                     # PDF reading
from docx import Document         # DOCX reading

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
        params={"max_new_tokens": 300, "temperature": 0.7, "decoding_method": "sample"},
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
        return text  # fallback if creds missing

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
        if isinstance(result, dict):
            rewritten = (result.get("generated_text") or "").strip()
        else:
            rewritten = (str(result) or "").strip()
        return rewritten if rewritten else text
    except Exception:
        return text


def speak_ibm_tts(text: str, voice: str = "en-US_AllisonV3Voice") -> bytes:
    """Synthesizes speech using IBM TTS and returns MP3 bytes."""
    tts = get_tts_client()
    if tts is None or not text.strip():
        st.error("❌ TTS client not initialized or empty text.")
        return b""

    try:
        res = tts.synthesize(
            text=text.strip(),
            voice=voice,
            accept="audio/mp3",
        ).get_result()
        return res.content
    except Exception as e:
        st.error(f"❌ TTS error: {str(e)}")
        return b""


# ---------- Input (Tabs) ----------
tab1, tab2, tab3, tab4 = st.tabs(["Paste text", "Upload .txt", "Upload .pdf", "Upload .docx"])

user_text = ""

with tab1:
    user_text = st.text_area("Enter your text", height=200, placeholder="Type or paste your story/article here...")

with tab2:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded is not None:
        raw = uploaded.read()
        try:
            user_text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                user_text = raw.decode("latin-1")
            except Exception:
                user_text = raw.decode("utf-8", errors="ignore")

with tab3:
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            pages = [page.extract_text() or "" for page in reader.pages]
            user_text = "\n".join(pages).strip()
            if not user_text:
                st.warning("⚠️ No extractable text found in this PDF (it might be scanned).")
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")

with tab4:
    docx_file = st.file_uploader("Upload a Word file", type=["docx"])
    if docx_file is not None:
        try:
            doc = Document(docx_file)
            user_text = "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception as e:
            st.error(f"Failed to read DOCX: {e}")


# ---------- Options ----------
tone = st.selectbox("🎚️ Choose tone", ["Neutral", "Suspenseful", "Inspiring"])

# ---------- Translation Helper ----------
def translate_text(text: str, target_lang: str) -> str:
    """Translate English text into the target language using Watsonx model."""
    model = get_watsonx_model()
    if model is None:
        return text  # fallback if creds missing

    prompt = f"""
    You are a professional translator. Translate the following English text into {target_lang}.
    Keep the meaning faithful and suitable for audiobook narration. 
    Do not add explanations, just return the translated text.

    <<<TEXT>>>
    {text}
    <<<END>>>
    """

    try:
        result = model.generate_text(prompt=prompt)
        if isinstance(result, dict):
            translated = (result.get("generated_text") or "").strip()
        else:
            translated = (str(result) or "").strip()
        return translated if translated else text
    except Exception:
        return text


# Available languages & voices
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

lang = st.selectbox("🌍 Choose language", list(languages.keys()))

voice = st.selectbox(
    "🗣️ Choose voice",
    languages[lang],
    index=0,
    help="Select a voice available for this language."
)

gen = st.button("✨ Rewrite, Translate & Generate Audio", type="primary", disabled=not bool(user_text.strip()))


# Language dropdown
lang = st.selectbox("🌍 Choose language", list(languages.keys()))

# Voices update dynamically based on selected language
voice = st.selectbox(
    "🗣️ Choose voice",
    languages[lang],
    index=0,
    help="Select a voice available for this language."
)

gen = st.button("✨ Rewrite & Generate Audio", type="primary", disabled=not bool(user_text.strip()))


# ---------- History Storage ----------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------- Processing ----------
if gen and user_text.strip():
    with st.spinner("Rewriting with selected tone..."):
        progress_bar = st.progress(0)
        rewritten = rewrite_with_tone(user_text, tone)
        progress_bar.progress(30)

    # If language is not English → translate
    final_text = rewritten
    if not lang.startswith("English"):
        with st.spinner(f"Translating into {lang}..."):
            final_text = translate_text(rewritten, lang)
        progress_bar.progress(60)
    else:
        progress_bar.progress(60)

    with st.spinner("Generating narration..."):
        audio_bytes = speak_ibm_tts(final_text, voice=voice)
        progress_bar.progress(100)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "⬇️ Download MP3",
            data=audio_bytes,
            file_name="echoverse_narration.mp3",
            mime="audio/mp3",
        )

        # Save to history
        st.session_state.history.append({
            "original": user_text,
            "rewritten": rewritten,
            "translated": final_text,
            "language": lang,
            "tone": tone,
            "voice": voice,
            "audio": audio_bytes
        })

        st.success("✅ Your Audio is Ready!")

# ---------- Display History ----------
if st.session_state.history:
    st.subheader("📜 History")
    for i, item in enumerate(reversed(st.session_state.history), start=1):
        with st.expander(f"{i}. {item['tone']} | {item['voice']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.markdown(item["original"][:500] + ("..." if len(item["original"]) > 500 else ""))
            with col2:
                st.markdown(f"**{item['tone']} Rewrite**")
                st.markdown(item["rewritten"][:500] + ("..." if len(item["rewritten"]) > 500 else ""))

            st.audio(item["audio"], format="audio/mp3")
            st.download_button(
                f"⬇️ Download Narration {i}",
                data=item["audio"],
                file_name=f"echoverse_history_{i}.mp3",
                mime="audio/mp3",
            )


# --- Footer ---
st.markdown("""
    <div style="text-align:center; color:gray; font-size:13px; margin-top:30px;">
         | By <b>TechElite</b>
    </div>
""", unsafe_allow_html=True)  