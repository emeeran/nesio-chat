import streamlit as st
import openai
from dotenv import load_dotenv
import os
import json
import time
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# Load environment variables
load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to generate response using OpenAI API
def generate_response(prompt, system_message, model, max_tokens, temperature, top_p):
    response = openai.ChatCompletion.create(
        # client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return (
        response.choices[0].message.content.strip(),
        response.usage.total_tokens,
    )


# Function to convert text to speech
def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language)
    filename = f"response_{int(time.time())}.mp3"
    tts.save(filename)
    return filename


# Function to save chat to file
def save_chat(chat, filename, file_format):
    filepath = os.path.join("chat_exports", f"{filename}.{file_format}")
    with open(filepath, "w") as f:
        if file_format == "json":
            json.dump(chat, f, indent=4)
        elif file_format == "md":
            f.write("\n".join(chat))
    return filepath


# Initialize session state variables
if "chat" not in st.session_state:
    st.session_state.chat = []
if "tokens_spent" not in st.session_state:
    st.session_state.tokens_spent = 0

# Sidebar settings
st.sidebar.header("Settings")

system_message_options = [
    "You are a helpful assistant.",
    "You are a strict and precise assistant.",
    "You are a friendly and engaging assistant.",
]
selected_system_message = st.sidebar.selectbox(
    "Select System Message", system_message_options
)

model_options = [
    "gpt-4o-2024-05-13",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0125-preview",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
]
selected_model = st.sidebar.selectbox("Select Model", model_options)

language_options = {"English": "en", "Tamil": "ta", "Hindi": "hi"}
selected_language = st.sidebar.selectbox(
    "Select Language", list(language_options.keys())
)

openai_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
selected_voice = st.sidebar.selectbox("Select OpenAI Voice", openai_voices)

enable_audio = st.sidebar.checkbox("Enable Audio Response")

with st.sidebar.expander("Parameters"):
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=4000, value=150)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=1.0)

with st.sidebar.expander("Content Creation"):
    content_type = st.selectbox("Select Content Type", ["Letter", "Email", "Article"])
    content_tone = st.selectbox("Select Tone", ["Formal", "Informal", "Friendly"])
    content_format = st.selectbox("Select Format", ["Short", "Medium", "Long"])

    st.subheader("Programming")
    programming_task = st.selectbox("Select Task", ["Code Writing", "Debugging"])

    st.subheader("Document Summarizing")
    summarizing_length = st.selectbox(
        "Select Summarizing Length", ["Short", "Detailed"]
    )

    st.subheader("Translation")
    target_language = st.selectbox(
        "Select Target Language", ["French", "Spanish", "German"]
    )

st.sidebar.subheader("Actions")
col1, col2 = st.sidebar.columns(2)
if col1.button("Reset Chat"):
    st.session_state.chat = []
    st.session_state.tokens_spent = 0
if col2.button("Save Chat"):
    filename = st.text_input("Enter filename:", value="chat")
    file_format = st.selectbox("Select format:", ["json", "md"])
    if filename:
        save_chat(st.session_state.chat, filename, file_format)
        st.success(f"Chat saved as {filename}.{file_format}")

col3, col4 = st.sidebar.columns(2)
if col3.button("Export to MD"):
    save_chat(st.session_state.chat, "chat", "md")
if col4.button("Export to JSON"):
    save_chat(st.session_state.chat, "chat", "json")

st.sidebar.subheader("Load Chat")
uploaded_file = st.sidebar.file_uploader("Choose a JSON file", type="json")
if uploaded_file is not None:
    st.session_state.chat = json.load(uploaded_file)
    st.session_state.tokens_spent = sum(
        message["tokens"] for message in st.session_state.chat
    )

# Token counter
st.sidebar.subheader("Token Counter")
st.sidebar.write(f"Tokens Spent: {st.session_state.tokens_spent}")
st.sidebar.write(
    f"Approximate Cost: ${st.session_state.tokens_spent * 0.02 / 1000:.2f} USD"
)

# Main content
st.title("Multimodal Assistant")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Assistant:** {message['content']}")

# Input prompt at the bottom
prompt = st.text_area("Enter your prompt here", height=100)

if st.button("Submit"):
    if prompt:
        with st.spinner("Generating response..."):
            response, tokens = generate_response(
                prompt,
                selected_system_message,
                selected_model,
                max_tokens,
                temperature,
                top_p,
            )
            st.session_state.chat.append(
                {"role": "user", "content": prompt, "tokens": 0}
            )
            st.session_state.chat.append(
                {"role": "assistant", "content": response, "tokens": tokens}
            )
            st.session_state.tokens_spent += tokens

            # Update chat display
            chat_container.empty()
            with chat_container:
                for message in st.session_state.chat:
                    if message["role"] == "user":
                        st.write(f"**You:** {message['content']}")
                    else:
                        st.write(f"**Assistant:** {message['content']}")

            # Text-to-speech
            if enable_audio:
                audio_file = text_to_speech(
                    response, language_options[selected_language]
                )
                st.audio(audio_file)

            # OpenAI voice (placeholder - replace with actual OpenAI voice API call)
            st.write(f"OpenAI voice '{selected_voice}' would be used here.")

    else:
        st.warning("Please enter a prompt")

# Scroll to the bottom of the chat
st.markdown(
    "<script>window.scrollTo(0,document.body.scrollHeight);</script>",
    unsafe_allow_html=True,
)

# Custom CSS
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        padding-top: 20px;
        padding-right: 30px;
        padding-left: 30px;
        padding-bottom: 20px;
    }
    .reportview-container .main {
        border-top: 10px solid #f63366;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
