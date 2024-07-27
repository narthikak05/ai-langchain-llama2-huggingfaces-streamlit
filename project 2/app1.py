import streamlit as st
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def load_answer(question):
    # Retrieve the API token from environment variable
    api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not api_token:
        return "Error: Hugging Face Hub API token not found."

    # Initialize HuggingFaceHub with the API token
    llm = HuggingFaceHub(
        repo_id="huggingfaceh4/zephyr-7b-alpha",
        model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512},
        huggingfacehub_api_token=api_token
    )

    # Get the response from the model
    result = llm.predict(question)
    return result

# Streamlit UI setup
st.title("Langchain Demo Using Chat Model")

def get_text():
    input_text = st.text_input(label="Enter A Chat:", placeholder='How are you?', key='new_question')
    return input_text

user_input = get_text()

submit = st.button("Chat")

if submit and user_input:
    st.subheader("Reply:")
    response = load_answer(user_input)
    st.write(response)