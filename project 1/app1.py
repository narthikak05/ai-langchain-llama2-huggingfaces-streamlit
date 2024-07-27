import os
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint

# Set the Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
print(f"API Token Set: {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}")  # Verify the token is set

def load_answer(question):
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3"
        )
        result = llm.invoke(question)
        return result
    except Exception as e:
        return f"An error occurred: {e}"

st.title("Langchain Demo")

def get_text():
    input = st.text_input(label="Enter a question:", placeholder='example: how to make lime juice?', key='new_question')
    return input

user_input = get_text()

submit = st.button("Generate")

if submit and user_input:
    st.subheader("Answer:")
    response = load_answer(user_input)
    st.write(response)