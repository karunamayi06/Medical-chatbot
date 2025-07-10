import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from gtts import gTTS
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("ai")
# Initialize the model (make sure it's a medically fine-tuned model or use a prompt specifically for medical advice)
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Define the prompt template with medical context
system_template = """
You are a highly knowledgeable medical consultant. Based on the symptoms provided, suggest possible conditions, recommended treatments or medications, and the type of specialist doctor the patient should consult.
Symptom: {problem}
"""

# Adjust the input/output
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Define the output parser
parser = StrOutputParser()

# Streamlit interface
st.title("Medical Consultant Chatbot")
st.write("Ask me anything about your symptoms. The bot will suggest possible medications and recommend which specialist to consult.")
st.write(groq_api_key)
# Add language selection in the sidebar
language = st.sidebar.selectbox(
    "Select Language",
    ("English", "Tamil", "Hindi", "Malayalam", "Kannada", "Telugu", "Marathi")
)

# Language codes for Google Translator
language_codes = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Telugu": "te",
    "Marathi": "mr"
}

# Initialize chat history in session state if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Input from the user
user_input = st.text_input("Enter your symptoms or medical problem:", key="input_key")

# Process the input if it's not empty
if user_input:
    # Prepare the input for the model
    formatted_prompt = prompt_template.format(problem=user_input, text=user_input)
    
    # Get model response (the model should be trained to give medical advice, or the prompt should direct it accordingly)
    model_response = model.predict(formatted_prompt)
    
    # Parse the model response
    response = parser.parse(model_response)
    
    # Translate the response if the language is not English
    if language != "English":
        target_language_code = language_codes[language]
        try:
            translator = GoogleTranslator(source='auto', target=target_language_code)
            translated_response = translator.translate(response)
        except Exception as e:
            translated_response = f"Error in translation: {str(e)}"
    else:
        translated_response = response
    
    # Save the user input and model response to the chat history
    st.session_state.chat_history.append({"user": user_input, "bot": translated_response})
    
    # Display the text response
    st.write(f"**Consultant:** {translated_response}")
    
    # Button to generate and play audio
    if st.button("Get Audio of Response"):
        try:
            # Convert response to audio using gTTS
            tts = gTTS(translated_response, lang=language_codes[language])
            tts.save("response_audio.mp3")
            
            # Confirm the file is saved
            st.write("Audio file saved successfully.")
            
            # Read and play the audio response
            with open("response_audio.mp3", "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                
                # Provide download option for testing
                st.download_button(
                    label="Download Audio",
                    data=audio_bytes,
                    file_name="response_audio.mp3",
                    mime="audio/mp3"
                )
        except Exception as e:
            st.write(f"Error generating audio: {str(e)}")

# Display the chat history with the latest response on top
for i, chat in reversed(list(enumerate(st.session_state.chat_history))):
    st.write(f"**User:** {chat['user']}")
    st.write(f"**Consultant:** {chat['bot']}")
    st.write("---")
