import streamlit as st
from streamlit_chat import message
from transformers import pipeline

# Initialize the Hugging Face pipeline
def initialize_pipeline():
    return pipeline("text-generation", model="gpt2")

chat_pipeline = initialize_pipeline()

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] = ''

# Setting page title and header
st.set_page_config(page_title="Chat GPT Clone", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>How can I assist you? </h1>", unsafe_allow_html=True)

st.sidebar.title("😎")
st.session_state['API_Key'] = st.sidebar.text_input("What's your API key?", type="password")
summarise_button = st.sidebar.button("Summarise the conversation", key="summarise")
if summarise_button:
    summarise_placeholder = st.sidebar.write("Nice chatting with you my friend ❤️:\n\n" + "\n".join(st.session_state['messages']))

def getresponse(userInput):
    if st.session_state['conversation'] is None:
        st.session_state['conversation'] = []

    response = chat_pipeline(userInput, max_length=50, num_return_sequences=1)
    response_text = response[0]['generated_text']
    return response_text

response_container = st.container()
# Here we will have a container for user input text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button:
            st.session_state['messages'].append(user_input)
            model_response = getresponse(user_input)
            st.session_state['messages'].append(model_response)

            with response_container:
                for i in range(len(st.session_state['messages'])):
                    if (i % 2) == 0:
                        message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
                    else:
                        message(st.session_state['messages'][i], key=str(i) + '_AI')
