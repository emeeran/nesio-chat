import streamlit as st
import os
from openai import AsyncOpenAI
import json
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO
import base64
from PIL import Image
import pytesseract
import fitz
import docx2txt
import tempfile
from persona import system_messages
import asyncio

# Set page config
st.set_page_config(page_title="NesiO-Chat", layout="wide")

# Load environment variables
load_dotenv()

@st.cache_resource
def load_openai_client():
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = load_openai_client()

# Initialize session state
def init_session_state():
    default_values = {
        "messages": [],
        "token_count": 0,
        "cost": 0,
        "model": "gpt-4o-mini",
        "max_tokens": 950,
        "temperature": 0.6,
        "top_p": 1.0,
        "enable_audio": False,
        "voice": "alloy",
        "language": "English",
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Constants
MAX_MESSAGES = 450
MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

def add_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.messages.append({"role": role, "content": content})
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

def process_image(file):
    """Process an uploaded image file and extract text."""
    try:
        with Image.open(file) as image:
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
            extracted_text = pytesseract.image_to_string(image)
            return img_base64, extracted_text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return None, None

def process_pdf(file):
    """Process an uploaded PDF file and extract text."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        with fitz.open(temp_file_path) as doc:
            text = "".join(page.get_text() for page in doc)
        os.remove(temp_file_path)
        return text
    except Exception as e:
        st.error(f"Error processing PDF file: {str(e)}")
        return None

def process_docx(file):
    """Process an uploaded DOCX file and extract text."""
    try:
        return docx2txt.process(file)
    except Exception as e:
        st.error(f"Error processing DOCX file: {str(e)}")
        return None

def process_file(file):
    """Process the uploaded file based on its type."""
    file_type = file.type
    try:
        if file_type.startswith("image"):
            img_base64, extracted_text = process_image(file)
            if img_base64 and extracted_text:
                add_message("user", f"I've uploaded an image: {file.name}")
                st.image(file, caption="Uploaded Image")
                if st.checkbox("Extract Text from Image?", key=f"ocr_{file.name}"):
                    add_message("assistant", f"Extracted Text from Image: {extracted_text}")
        elif file_type == "text/plain":
            file_contents = file.getvalue().decode("utf-8")
            add_message("user", f"I've uploaded a text file: {file.name}")
            add_message("assistant", f"File Contents:\n{file_contents}")
            asyncio.run(summarize_content(file_contents, "text"))
        elif file_type == "application/pdf":
            text = process_pdf(file)
            if text:
                add_message("user", f"I've uploaded a PDF file: {file.name}")
                add_message("assistant", f"PDF Contents:\n{text}")
                asyncio.run(summarize_content(text, "PDF"))
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = process_docx(file)
            if text:
                add_message("user", f"I've uploaded a DOCX file: {file.name}")
                add_message("assistant", f"DOCX Contents:\n{text}")
                asyncio.run(summarize_content(text, "DOCX"))
        else:
            st.error(f"Unsupported file type: {file_type}")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

async def summarize_content(content: str, content_type: str):
    """Summarize the provided content using OpenAI's API."""
    try:
        response = await client.chat.completions.create(
            model=st.session_state.model,
            messages=[
                {"role": "system", "content": f"You are a summarization assistant. Summarize the following {content_type} content in {st.session_state.language}."},
                {"role": "user", "content": content},
            ],
            max_tokens=450,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
        )
        summarized_content = response.choices[0].message.content.strip()
        add_message("assistant", f"Summarized Content: {summarized_content}")
    except Exception as e:
        st.error(f"Error summarizing {content_type} content: {str(e)}")

def get_current_weather(location, unit="celsius"):
    """Fetch current weather data (placeholder implementation)."""
    return {"temperature": 22, "unit": unit, "description": "Sunny"}

functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

async def generate_response(user_input: str, system_message: str):
    """Generate a response from the model based on user input."""
    messages = [{"role": "system", "content": f"{system_message} Respond in {st.session_state.language}."}, {"role": "user", "content": user_input}]

    try:
        response_container = st.empty()
        full_response = ""
        async for response in await client.chat.completions.create(
            model=st.session_state.model,
            messages=messages,
            functions=functions,
            function_call="auto",
            max_tokens=st.session_state.max_tokens,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            stream=True,
        ):
            if response.choices[0].delta.function_call:
                function_name = response.choices[0].delta.function_call.name
                function_args = json.loads(response.choices[0].delta.function_call.arguments)
                if function_name == "get_current_weather":
                    function_response = get_current_weather(**function_args)
                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": json.dumps(function_response),
                        }
                    )
                    second_response = await client.chat.completions.create(
                        model=st.session_state.model,
                        messages=messages,
                        max_tokens=st.session_state.max_tokens,
                        temperature=st.session_state.temperature,
                        top_p=st.session_state.top_p,
                    )
                    full_response = second_response.choices[0].message.content
                    break
            else:
                full_response += response.choices[0].delta.content or ""
                response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)
        add_message("assistant", full_response)

        if st.session_state.enable_audio:
            try:
                audio_response = await client.audio.speech.create(
                    model="tts-1", voice=st.session_state.voice, input=full_response
                )
                audio_data = audio_response.content
                st.audio(audio_data, format="audio/mp3")
            except Exception as e:
                st.error(f"Error generating audio: {str(e)}")

        # Update token count and cost (rough estimate)
        st.session_state.token_count += len(full_response.split())
        st.session_state.cost += (len(full_response.split()) / 1000) * 0.02

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def reset_all():
    """Reset all session state variables."""
    for key in st.session_state.keys():
        del st.session_state[key]
    init_session_state()

def save_chat():
    chat_exports_dir = "./chat_exports"
    os.makedirs(chat_exports_dir, exist_ok=True)
    filename = f"chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(chat_exports_dir, filename), "w") as f:
        json.dump(st.session_state.messages, f)
    st.sidebar.success(f"Chat saved as {filename}")

def export_chat(format):
    chat_exports_dir = "./chat_exports"
    os.makedirs(chat_exports_dir, exist_ok=True)
    filename = f"chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{format}"
    with open(os.path.join(chat_exports_dir, filename), "w") as f:
        if format == "md":
            for message in st.session_state.messages:
                f.write(f"**{message['role']}**: {message['content']}\n\n")
        elif format == "json":
            json.dump(st.session_state.messages, f)
    st.sidebar.success(f"Chat exported as {filename}")

def main():
    # Sidebar configuration
    st.sidebar.title("Settings")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Reset Chat"):
        reset_all()
    if col2.button("Save Chat"):
        save_chat()

    col3, col4 = st.sidebar.columns(2)
    if col3.button("Export to MD"):
        export_chat("md")
    if col4.button("Export to JSON"):
        export_chat("json")

    # Load saved chats
    chat_exports_dir = "./chat_exports"
    os.makedirs(chat_exports_dir, exist_ok=True)
    saved_chats = [f for f in os.listdir(chat_exports_dir) if f.endswith(".json")]
    selected_chat = st.sidebar.selectbox("Load Chat", [""] + saved_chats)
    if selected_chat:
        with open(os.path.join(chat_exports_dir, selected_chat), "r") as f:
            st.session_state.messages = json.load(f)
        st.sidebar.success(f"Loaded chat: {selected_chat}")

    st.sidebar.checkbox("Enable Audio Response", key="enable_audio")
    if st.session_state.enable_audio:
        st.sidebar.selectbox("Select Voice", OPENAI_VOICES, key="voice")

    st.sidebar.subheader("Select Model")
    st.sidebar.selectbox("Model", MODELS, key="model")

    # Persona selection
    system_messages_with_custom = ["Default"] + list(system_messages.keys()) + ["Custom"]
    st.sidebar.subheader("Select Persona")
    selected_system_message = st.sidebar.selectbox(
        "Select Persona",
        system_messages_with_custom,
        index=0,
        key="system_message",
    )

    if selected_system_message == "Custom":
        custom_system_message = st.sidebar.text_input(
            "Enter Custom System Message",
            key="custom_system_message",
            placeholder="Enter Custom System Message",
        )
    elif selected_system_message == "Default":
        st.sidebar.text_area(
            "System Message",
            "You are a helpful AI assistant.",
            key="selected_system_message_content",
            height=100,
            disabled=True,
        )
    else:
        st.sidebar.text_area(
            "System Message",
            system_messages[selected_system_message],
            key="selected_system_message_content",
            height=100,
            disabled=True,
        )

    with st.sidebar.expander("Params"):
        st.slider("Max Tokens", 50, 8000, 1000, key="max_tokens")
        st.slider("Temperature", 0.0, 1.0, 0.6, key="temperature")
        st.slider("Top P", 0.0, 1.0, 1.0, key="top_p")

    with st.sidebar.expander("Content Creation Options"):
        st.selectbox(
            "Select Content Type",
            ["Letter", "Email", "Article", "Blog", "Essay"],
            key="content_type",
        )
        st.selectbox(
            "Select Tone",
            ["Formal", "Informal", "Friendly", "Assertive", "Persuasive"],
            key="tone",
        )
        st.selectbox("Select Format", ["Short", "Medium", "Long"], key="format")

    st.sidebar.subheader("Select Language")
    st.sidebar.selectbox("Select Language", ["English", "Tamil", "Hindi"], key="language")

    st.sidebar.subheader("Programming")
    st.sidebar.selectbox(
        "Select Programming Task", ["Coding", "Debugging"], key="programming_task"
    )

    st.sidebar.subheader("Summarization")
    st.sidebar.selectbox(
        "Select Summary Type",
        [
            "as described in prompt",
            "Key Takeaways",
            "Main Points bulleted",
            "Overview",
            "Comprehensive overview",
            "Detailed Summary",
            "In-depth Summary",
            "Executive Summary",
        ],
        key="summary_type",
    )

    # Upload File
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file", type=["txt", "pdf", "png", "jpg", "jpeg", "docx"]
    )

    # Process uploaded file
    if uploaded_file:
        process_file(uploaded_file)

    # Display token count and cost
    st.sidebar.markdown(f"**Token Count:** {st.session_state.token_count}")
    st.sidebar.markdown(f"**Estimated Cost:** ${st.session_state.cost:.4f}")

    # Main chat interface
    st.title("NesiO-Chat")

    # Create a container for the chat history
    chat_container = st.container()

    # Display chat messages in the chat history container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "image" in message:
                    st.image(message["image"], caption="Uploaded Image")

    # User input at the bottom
    user_input = st.chat_input("Type your message here...")

    # Process user input
    if user_input:
        add_message("user", user_input)
        if selected_system_message == "Custom":
            system_message = custom_system_message
        elif selected_system_message == "Default":
            system_message = "You are a helpful AI assistant."
        else:
            system_message = system_messages[selected_system_message]
        asyncio.run(generate_response(user_input, system_message))

        # Rerun the app to update the chat history
        st.rerun()

# Custom CSS to make the chat container scrollable and improve UI
st.markdown("""
    <style>
    .stApp {
        max-height: 100vh;
        overflow: hidden;
    }
    .main .block-container {
        max-height: calc(100vh - 150px);
        overflow-y: auto;
        display: flex;
        flex-direction: column-reverse;
    }
    .chat-message {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .chat-message.user {
        background-color: #e0f7fa;
        align-self: flex-end;
    }
    .chat-message.assistant {
        background-color: #ffe0b2;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()