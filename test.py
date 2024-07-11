import os
import json
import openai
import streamlit as st
# from streamlit_chat import message
from dotenv import load_dotenv
from playsound import playsound  # For audio response
from gtts import gTTS             # For text-to-speech
import markdown                   # For Markdown export
import tempfile
from PIL import Image
import io
import datetime

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")  # Replace with your actual key
news_api_key = os.getenv("NEWS_API_KEY")                  # Replace with your actual key

# --- Global Variables ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "token_count" not in st.session_state:
    st.session_state.token_count = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0

# --- Constants ---
TOKEN_PRICE_PER_1K = 0.002  # Assuming a price of $0.002 / 1K tokens
CHAT_EXPORTS_DIR = "./chat_exports"
VOICE_LANGUAGE_CODES = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
}

# --- Utility Functions ---

def generate_response(messages, model="gpt-4o-2024-05-13",
, max_tokens=1200, temperature=0.7, top_p=1):
    """Generates a response from the OpenAI API."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    response_text = response.choices[0].message.content  # Access using dot notation
    tokens_used = response.usage.total_tokens      # Access using dot notation
    return response_text, tokens_used


def display_message(message_text, is_user=True, key=None):
    """Displays a message in the chat interface."""
    if is_user:
        st.text_area(label="", value=f"User: {message_text}", key=key, disabled=True)
    else:
        st.text_area(label="", value=f"Bot: {message_text}", key=key, disabled=True)
def update_token_info(tokens_used):
    """Updates the token count and estimated cost."""
    st.session_state.token_count += tokens_used
    st.session_state.total_cost = st.session_state.token_count * TOKEN_PRICE_PER_1K / 1000

def reset_chat():
    """Clears the chat history, token count, and cost."""
    st.session_state.messages = []
    st.session_state.token_count = 0
    st.session_state.total_cost = 0

def save_chat(format="json"):
    """Saves the chat history to a file."""
    os.makedirs(CHAT_EXPORTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "json":
        file_path = os.path.join(CHAT_EXPORTS_DIR, f"chat_history_{timestamp}.json")
        with open(file_path, 'w') as f:
            json.dump(st.session_state.messages, f, indent=4)
        st.success(f"Chat saved to {file_path}")

    elif format == "md":
        file_path = os.path.join(CHAT_EXPORTS_DIR, f"chat_history_{timestamp}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            for msg in st.session_state.messages:
                if msg["is_user"]:
                    f.write(f"**User:** {msg['content']}\n\n")
                else:
                    f.write(f"**Bot:** {msg['content']}\n\n")
        st.success(f"Chat saved to {file_path}")
    else:
        st.error("Invalid file format selected.")

def load_chat(file_path):
    """Loads chat history from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            loaded_messages = json.load(f)
        st.session_state.messages = loaded_messages
        st.success("Chat loaded successfully!")
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")

def play_audio_response(text, language="en"):
    """Plays the audio response using gTTS and playsound."""
    try:
        tts = gTTS(text=text, lang=language)
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            tts.save(fp.name)
            playsound(fp.name)
    except Exception as e:
        st.error(f"Error in audio playback: {e}")

def process_image(image_data):
    """Processes the uploaded image and returns a short description."""
    # Placeholder: Replace with actual image processing logic
    try:
        img = Image.open(io.BytesIO(image_data))
        # ... (Add image processing code here, e.g., using OpenCV) ...
        return "Image description (placeholder)"
    except Exception as e:
        return f"Error processing image: {e}"

# --- UI Components ---
def main():
    st.title("Multimodal Chatbot")

    # --- Sidebar ---
    st.sidebar.title("Settings")

    # Token Counter
    st.sidebar.subheader("Token Counter")
    st.sidebar.text(f"Tokens Spent: {st.session_state.token_count}")
    st.sidebar.text(f"Approximate Cost: ${st.session_state.total_cost:.5f}")

    # Buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button("Reset Chat", on_click=reset_chat)
    with col2:
        st.button("Save Chat", on_click=save_chat, args=("json",))

    col3, col4 = st.sidebar.columns(2)
    with col3:
        st.button("Export to md", on_click=save_chat, args=("md",))
    with col4:
        st.button("Export to JSON", on_click=save_chat, args=("json",))

    # Load Chat
    uploaded_file = st.sidebar.file_uploader("Load Chat", type=["json"])
    if uploaded_file is not None:
        load_chat(uploaded_file)

    # Enable Audio Response
    enable_audio = st.sidebar.checkbox("Enable Audio Response")
    selected_voice = st.sidebar.selectbox("Voice Language", list(VOICE_LANGUAGE_CODES.keys()))
    voice_language = VOICE_LANGUAGE_CODES.get(selected_voice, "en")

    # System Message
    system_message = st.sidebar.text_area("Select System Message", value="You are a helpful assistant.")

    # Select Model
    selected_model = st.sidebar.selectbox(
        "Select Model",
        [
            "gpt-4o-2024-05-13",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-0125-preview",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
        ],
    )

    # Parameters
    st.sidebar.subheader("Parameters")
    max_tokens = st.sidebar.slider("Max Tokens", 50, 4000, 150)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0)

    # File Drag & Drop
    uploaded_file = st.sidebar.file_uploader("Drag & Drop Files Here", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_description = process_image(uploaded_file.read())
        st.session_state.messages.append({"role": "user", "content": f"Image Description: {image_description}", "is_user": True})

    # --- Main Chat Display ---
    for message_data in st.session_state.messages:
        display_message(message_data["content"], message_data["is_user"])

    # User Input (at the bottom)
    user_input = st.text_area("You:", key="user_input")

    # Send Button
    if st.button("Send") and user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input, "is_user": True})
        display_message(user_input)

        # Generate response
        with st.spinner("Thinking..."):
            messages_for_api = [
                {"role": "system", "content": system_message},
                *st.session_state.messages,
            ]

            # Debugging: Print messages_for_api before sending to the API
            print("Messages for API:")
            for msg in messages_for_api:
                print(f"  {msg['role']}: {msg['content']}")

            response, tokens_used = generate_response(
                messages_for_api,
                selected_model,
                max_tokens,
                temperature,
                top_p
            )

        # Add bot message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response, "is_user": False})
        display_message(response, is_user=False)
        update_token_info(tokens_used)

        # Play audio response
        if enable_audio:
            play_audio_response(response, voice_language)

if __name__ == "__main__":
    main()
