import streamlit as st
import os
import openai
import json
from dotenv import load_dotenv
import pandas as pd
from gtts import gTTS
from io import BytesIO, StringIO
import base64
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from docx import Document

# Set page config
st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:", layout="wide")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "token_count" not in st.session_state:
    st.session_state.token_count = 0
if "cost" not in st.session_state:
    st.session_state.cost = 0
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# System messages
system_messages = {
    "default": "You are a helpful assistant.",
    "creative": "You are a creative assistant. Provide imaginative and original responses.",
    "analytical": "You are an analytical assistant. Provide detailed, logical analyses.",
    # Add more system messages as needed
}

# Models
models = [
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
]


# Function to process uploaded image
def process_image(file):
    try:
        image = Image.open(file)
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"I've uploaded an image: {file.name}",
                "image": img_base64,
            }
        )
        st.image(image, caption="Uploaded Image")

        extracted_text = pytesseract.image_to_string(image)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Extracted Text from Image: {extracted_text}",
            }
        )
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")


# Function to process uploaded text file
def process_text(file):
    try:
        file_contents = file.getvalue().decode("utf-8")
        st.session_state.messages.append(
            {"role": "user", "content": f"I've uploaded a text file: {file.name}"}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": f"File Contents:\n{file_contents}"}
        )
        summarize_content(file_contents, "text")
    except Exception as e:
        st.error(f"Error processing text file: {str(e)}")


# Function to process uploaded PDF file
def process_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        st.session_state.messages.append(
            {"role": "user", "content": f"I've uploaded a PDF file: {file.name}"}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": f"PDF Contents:\n{text}"}
        )
        summarize_content(text, "PDF")
    except Exception as e:
        st.error(f"Error processing PDF file: {str(e)}")


# Function to process uploaded DOCX file
def process_docx(file):
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        st.session_state.messages.append(
            {"role": "user", "content": f"I've uploaded a DOCX file: {file.name}"}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": f"DOCX Contents:\n{text}"}
        )
        summarize_content(text, "DOCX")
    except Exception as e:
        st.error(f"Error processing DOCX file: {str(e)}")


# Function to summarize content using OpenAI API
def summarize_content(content, content_type):
    try:
        summary = openai.Completion.create(
            model=selected_model,
            prompt=f"Summarize the following {content_type} content:\n{content}",
            max_tokens=150,
            temperature=temperature,
            top_p=top_p,
        )
        summarized_content = summary.choices[0].text.strip()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Summarized Content: {summarized_content}",
            }
        )
    except Exception as e:
        st.error(f"Error summarizing {content_type} content: {str(e)}")


# Function to generate AI response
def generate_response(user_input, system_message):
    messages = [{"role": "system", "content": system_message}]
    messages.extend(st.session_state.messages)

    try:
        response = openai.ChatCompletion.create(
            model=selected_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})

        if enable_audio:
            if voice not in ["en", "ta", "hi"]:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Error: Selected voice not supported.",
                    }
                )
            else:
                tts = gTTS(reply, lang=voice)
                audio_data = BytesIO()
                tts.write_to_fp(audio_data)
                st.audio(audio_data, format="audio/mp3")

        # Update token count and cost
        st.session_state.token_count += response.usage.total_tokens
        st.session_state.cost += (
            response.usage.total_tokens / 1000
        ) * 0.02  # Assuming $0.02 per 1k tokens

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


# Sidebar configuration
st.sidebar.title("Settings")

# Token Counter
st.sidebar.subheader("Token Counter")
st.sidebar.text(f"Tokens Spent: {st.session_state.token_count}")
st.sidebar.text(f"Approximate Cost: ${st.session_state.cost:.4f}")

# Buttons for resetting and saving chat
col1, col2 = st.sidebar.columns(2)
if col1.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.token_count = 0
    st.session_state.cost = 0

if col2.button("Save Chat"):
    chat_exports_dir = "./chat_exports"
    os.makedirs(chat_exports_dir, exist_ok=True)
    filename = f"chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(chat_exports_dir, filename), "w") as f:
        json.dump(st.session_state.messages, f)
    st.sidebar.success(f"Chat saved as {filename}")

# Buttons for exporting chat to MD or JSON format
col3, col4 = st.sidebar.columns(2)
if col3.button("Export to MD"):
    chat_exports_dir = "./chat_exports"
    os.makedirs(chat_exports_dir, exist_ok=True)
    filename = f"chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(os.path.join(chat_exports_dir, filename), "w") as f:
        for message in st.session_state.messages:
            f.write(f"**{message['role']}**: {message['content']}\n\n")
    st.sidebar.success(f"Chat exported as {filename}")

if col4.button("Export to JSON"):
    chat_exports_dir = "./chat_exports"
    os.makedirs(chat_exports_dir, exist_ok=True)
    filename = f"chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(chat_exports_dir, filename), "w") as f:
        json.dump(st.session_state.messages, f)
    st.sidebar.success(f"Chat exported as {filename}")

# Dropdown to load saved chats
chat_exports_dir = "./chat_exports"
os.makedirs(chat_exports_dir, exist_ok=True)
try:
    saved_chats = [f for f in os.listdir(chat_exports_dir) if f.endswith(".json")]
except Exception as e:
    st.sidebar.error(f"Error loading saved chats: {str(e)}")
    saved_chats = []

selected_chat = st.sidebar.selectbox("Load Chat", [""] + saved_chats)
if selected_chat:
    with open(os.path.join(chat_exports_dir, selected_chat), "r") as f:
        st.session_state.messages = json.load(f)
    st.sidebar.success(f"Loaded chat: {selected_chat}")

# Checkbox and dropdown to enable audio response and select voice
enable_audio = st.sidebar.checkbox("Enable Audio Response")
if enable_audio:
    voice = st.sidebar.selectbox("Select Voice", ["en", "ta", "hi"])

# Dropdown to select system message and model
selected_system_message = st.sidebar.selectbox(
    "Select System Message", list(system_messages.keys())
)
selected_model = st.sidebar.selectbox("Select Model", models, index=0)

# Sliders to adjust parameters
max_tokens = st.sidebar.slider("Max Tokens", 50, 4000, 150)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0)

# Main chat interface
st.title("AI Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], caption="Uploaded Image")

# User input field
user_input = st.text_input("Type your message here...")

# Process user input and uploaded file
if user_input or st.session_state.uploaded_file:
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        generate_response(user_input, system_messages[selected_system_message])

    if st.session_state.uploaded_file:
        file_processor = {
            "image": process_image,
            "text/plain": process_text,
            "application/pdf": process_pdf,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": process_docx,
        }

        file_type = st.session_state.uploaded_file.type
        if file_type.startswith("image"):
            file_type = "image"

        if file_type in file_processor:
            file_processor[file_type](st.session_state.uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_type}")

# File uploader widget
uploaded_file = st.sidebar.file_uploader(
    "Upload a file", type=["txt", "pdf", "png", "jpg", "jpeg", "docx"]
)

# Update uploaded file in session state
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
