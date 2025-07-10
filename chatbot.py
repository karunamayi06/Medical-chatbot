import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from gtts import gTTS

# --- Load env vars (for local dev)
load_dotenv()
groq_api_key = st.sidebar.text_input("🔑 Enter your GROQ API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

# --- Streamlit UI Setup
st.set_page_config(page_title="🩺 Medical Chatbot", layout="wide")
st.title("🩺 Multilingual Medical Consultant Chatbot")

# --- Sidebar
st.sidebar.header("🌐 Settings")
model_name = st.sidebar.selectbox("Choose Model", ["Gemma2-9b-It", "llama3-8b-8192"])
language = st.sidebar.selectbox("Select Language", ["English", "Tamil", "Hindi", "Malayalam", "Kannada", "Telugu", "Marathi"])

# --- Session State for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Language Code Map
language_codes = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Telugu": "te",
    "Marathi": "mr"
}

# --- Prompt Setup
system_template = """
You are a highly knowledgeable medical consultant. Based on the symptoms provided, suggest possible conditions, recommended treatments or medications, and the type of specialist doctor the patient should consult.
Symptom: {problem}
"""
prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{text}")])
parser = StrOutputParser()

# --- Initialize LLM
if groq_api_key:
    model = ChatGroq(model=model_name, groq_api_key=groq_api_key)

# --- Chat History Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input Section
user_input = st.chat_input("Describe your symptoms (e.g., 'I have a sore throat and fever')")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not groq_api_key:
        st.error("❗ Please enter your GROQ API key to continue.")
        st.stop()

    # Run the prompt and get model output
    formatted_prompt = prompt_template.format(problem=user_input, text=user_input)
    try:
        model_response = model.predict(formatted_prompt)
        parsed_response = parser.parse(model_response)

        # Translate if needed
        if language != "English":
            translator = GoogleTranslator(source='auto', target=language_codes[language])
            parsed_response = translator.translate(parsed_response)

        # Save and show bot response
        st.session_state.messages.append({"role": "assistant", "content": parsed_response})
        with st.chat_message("assistant"):
            st.markdown(parsed_response)

    except Exception as e:
        st.error(f"❌ Error: {e}")
