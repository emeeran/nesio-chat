import streamlit as st
import os
import sys
import base64
from gtts import gTTS
import PyPDF2
import docx
from PIL import Image
import pytesseract
import tempfile
from deep_translator import GoogleTranslator
from openai import OpenAI
from datetime import datetime
from fpdf import FPDF

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set page config at the very beginning
st.set_page_config(page_title="OpenAI-Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Load configuration
def load_config():
    return {
        'api_key': os.environ.get('OPENAI_API_KEY'),
        'models': {
            'gpt-4': {"name": "GPT-4", "tokens": 8192},
            'gpt-3.5-turbo': {"name": "GPT-3.5 Turbo", "tokens": 4096}
        }
    }

# Load configuration
try:
    config = load_config()
except Exception as e:
    st.error(f"Failed to load configuration: {str(e)}")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=config['api_key'])

# System messages
system_messages = {
    "Default": "You are a helpful assistant.",
    "Creative Writer": "You are a creative writer, skilled in crafting engaging stories and prose.",
    "Code Expert": "You are an expert programmer, proficient in multiple programming languages and software development practices.",
    "Math Tutor": "You are a patient and knowledgeable math tutor, able to explain complex concepts in simple terms.",
    "History Buff": "You are a history enthusiast with extensive knowledge of world history and cultural developments.",
    "Science Educator": "You are a science educator, passionate about explaining scientific concepts and recent discoveries.",
    "Philosophy Guide": "You are a philosophy guide, well-versed in various philosophical traditions and able to discuss complex ideas.",
    "Language Tutor": "You are a language tutor, fluent in multiple languages and skilled in teaching language concepts.",
    "Career Advisor": "You are a career advisor, offering guidance on job searches, resume writing, and professional development.",
    "Fitness Coach": "You are a fitness coach, providing advice on exercise routines, nutrition, and overall wellness.",
    "Travel Planner": "You are a travel planner, knowledgeable about destinations worldwide and skilled in creating itineraries.",
    "Financial Advisor": "You are a financial advisor, offering guidance on personal finance, investing, and financial planning.",
    "Cooking Expert": "You are a cooking expert, sharing recipes, cooking techniques, and culinary knowledge.",
    "Tech Support": "You are a tech support specialist, helping users troubleshoot software and hardware issues.",
    "Movie Critic": "You are a movie critic, discussing films, directors, and cinematic techniques.",
    "Music Enthusiast": "You are a music enthusiast, knowledgeable about various genres, artists, and music history.",
    "Art Curator": "You are an art curator, discussing art history, techniques, and contemporary art movements.",
    "Environmental Scientist": "You are an environmental scientist, discussing climate change, conservation, and sustainability.",
    "Psychologist": "You are a psychologist, offering insights into human behavior, mental health, and personal growth.",
    "Legal Advisor": "You are a legal advisor, providing general information about laws and legal processes.",
    "Debate Moderator": "You are a debate moderator, facilitating discussions on various topics while maintaining neutrality.",
    "Ethics Consultant": "You are an ethics consultant, discussing moral dilemmas and ethical decision-making.",
    "Futurist": "You are a futurist, speculating on future trends and technological developments.",
    "Mythology Expert": "You are a mythology expert, well-versed in myths and legends from cultures around the world.",
    "Social Media Strategist": "You are a social media strategist, offering advice on online presence and digital marketing.",
    "Quantum Physics Explainer": "You are a quantum physics explainer, making complex physics concepts accessible to non-experts.",
    "Mindfulness Coach": "You are a mindfulness coach, guiding meditation practices and stress reduction techniques.",
    "Urban Planner": "You are an urban planner, discussing city development, infrastructure, and community design.",
    "Fashion Consultant": "You are a fashion consultant, offering style advice and discussing fashion trends.",
    "Game Designer": "You are a game designer, discussing game mechanics, storytelling in games, and game development."
}

# Voice options
voice_options = {
    "English (US)": "en",
    "English (UK)": "en-gb",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh-cn",
    "Hindi": "hi",
    "Arabic": "ar",
    "Turkish": "tr",
    "Dutch": "nl",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da",
    "Finnish": "fi",
    "Polish": "pl"
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_base64" not in st.session_state:
    st.session_state.audio_base64 = ""
if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = []
if "system_message" not in st.session_state:
    st.session_state.system_message = system_messages["Default"]
if "model_params" not in st.session_state:
    st.session_state.model_params = {}
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0
if "enable_audio" not in st.session_state:
    st.session_state.enable_audio = False
if "voice" not in st.session_state:
    st.session_state.voice = "English (US)"

def export_chat(format):
    chat_history = "\n\n".join([f"**{m['role'].capitalize()}:** {m['content']}" for m in st.session_state.messages])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_exports/chat_history_{timestamp}.{format}"
    os.makedirs("chat_exports", exist_ok=True)

    if format == "md":
        with open(filename, "w") as f:
            f.write(chat_history)
    elif format == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, chat_history)
        pdf.output(filename)
        with open(filename, "rb") as f:
            st.download_button("Download PDF", f, file_name=filename)

def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    elif uploaded_file.type in ["text/plain", "text/markdown"]:
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    else:
        raise ValueError("Unsupported file type")

def summarize_file(prompt):
    if st.session_state.file_content:
        full_prompt = f"{prompt}\n\nContent:\n{st.session_state.file_content[:4000]}..."
        st.session_state.messages.extend([
            {"role": "user", "content": f"Summarize the uploaded file: {prompt}"},
            {"role": "assistant", "content": "Certainly! I'll summarize the file content based on your prompt."}
        ])

        try:
            response = client.chat.completions.create(
                model=st.session_state.model_params["model"],
                messages=[{"role": "user", "content": full_prompt}],
                stream=True
            )
            summary = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    summary += chunk.choices[0].delta.content
                    st.session_state.messages[-1]["content"] = summary
                    st.experimental_rerun()
            update_token_count(len(summary.split()))
        except Exception as e:
            st.error(f"Error summarizing file: {str(e)}")
    else:
        st.warning("Please upload a file first.")

def save_chat_history():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.chat_histories.append({
        "name": f"Chat {timestamp}",
        "messages": st.session_state.messages.copy()
    })

def load_chat_history(selected_history):
    for history in st.session_state.chat_histories:
        if history["name"] == selected_history:
            st.session_state.messages = history["messages"].copy()
            break

def update_token_count(tokens):
    st.session_state.total_tokens += tokens
    # Assuming a cost of $0.0001 per token (adjust as needed)
    st.session_state.total_cost += tokens * 0.0001

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        tts.save(temp_file.name)
        with open(temp_file.name, "rb") as f:
            audio_bytes = f.read()
    os.unlink(temp_file.name)
    audio_base64 = base64.b64encode(audio_bytes).decode()
    st.session_state.audio_base64 = audio_base64

def reset_all():
    st.session_state.messages = []
    st.session_state.total_tokens = 0
    st.session_state.total_cost = 0
    st.session_state.file_content = ""
    st.session_state.audio_base64 = ""

def translate_text(text, target_lang):
    if target_lang == "english":
        return text
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

def main():
    st.markdown("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>OpenAI Fast Chat</i> 💬</h1>""", unsafe_allow_html=True)

    with st.sidebar:
        st.title("🔧 Settings")

        st.write(f"Total Tokens: {st.session_state.total_tokens}")
        st.write(f"Total Cost: ${st.session_state.total_cost:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.button("📄 Exp.to PDF", on_click=export_chat, args=("pdf",))
            if st.button("Reset All"):
                reset_all()
        with col2:
            st.button("📄 Exp.to md", on_click=export_chat, args=("md",))
            if st.button("Save Chat"):
                save_chat_history()

        selected_history = st.selectbox("Load Chat", options=[h["name"] for h in st.session_state.chat_histories])
        if st.button("Load"):
            load_chat_history(selected_history)

        st.session_state.enable_audio = st.checkbox("Enable Audio Response", value=False)

        if st.session_state.enable_audio:
            st.session_state.voice = st.selectbox("Select Voice:", options=list(voice_options.keys()))

        selected_system_message = st.selectbox("Select System Message:", options=list(system_messages.keys()))
        st.session_state.system_message = system_messages[selected_system_message]

        model_choice = st.selectbox("Choose Model:", options=list(config['models'].keys()), format_func=lambda x: config['models'][x]["name"])

        with st.expander("Advanced Model Parameters"):
            max_tokens = st.slider("Max Tokens:", min_value=512, max_value=config['models'][model_choice]["tokens"], value=4096, step=512)
            model_temp = st.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
            top_p = st.slider("Top-p:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

        st.session_state.model_params = {
            "model": model_choice,
            "temperature": model_temp,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        uploaded_file = st.file_uploader("Upload File", type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"])
        if uploaded_file:
            try:
                st.session_state.file_content = process_uploaded_file(uploaded_file)
                st.success("File uploaded successfully!")
                st.info("You can now ask questions about the uploaded file in the main chat.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Main chat interface
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Prompt input at the bottom
    prompt = st.chat_input("Message OpenAI Fast Chat...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state.model_params["model"],
                messages=[
                    {"role": "system", "content": st.session_state.system_message},
                    *st.session_state.messages
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        update_token_count(len(full_response.split()))

        if st.session_state.enable_audio:
            text_to_speech(full_response, voice_options[st.session_state.voice])
            st.audio(f"data:audio/mp3;base64,{st.session_state.audio_base64}", format="audio/mp3")

    # Scroll to the bottom of the chat
    st.markdown('<script>window.scrollTo(0,document.body.scrollHeight);</script>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()