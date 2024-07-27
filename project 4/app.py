import streamlit as st
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from dotenv import load_dotenv

load_dotenv()

def initialize_pipeline():
    return pipeline("text-generation", model="gpt2")  # or another suitable model

chat_pipeline = initialize_pipeline()

def getLLMResponse(query, age_option, tasktype_option):
    if age_option == "Kid":  # Silly and Sweet Kid
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile is a electronic device]"},
            {"query": "What are your dreams?", "answer": "My dreams are like colorful..."},
            {"query": " What are your ambitions?", "answer": "I want to be a super funny..."},
            # Add more examples as needed
        ]
    
    elif age_option == "Adult":  # Curious and Intelligent Adult
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile is a portable device"},
            {"query": "What are your dreams?", "answer": "My dream is to become a doctor"},
            {"query": " What are your ambitions?", "answer": "My ambition is to become a dentist"},
            # Add more examples as needed
        ]

    elif age_option == "Senior Citizen":  # A 90 years old guy
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile used for communication and message transfer"},
            {"query": "What are your dreams?", "answer": "My dreams for my grandsons..."},
            {"query": "What happens when you get sick?", "answer": "When I get sick, I will go to hospital"},
            # Add more examples as needed
        ]

    example_template = """
    Question: {query}
    Response: {answer}
    """

    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    prefix = """You are a {template_ageoption}, and {template_tasktype_option}: 
    Here are some examples: 
    """

    suffix = """
    Question: {template_userInput}
    Response: """

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=200
    )

    new_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["template_userInput", "template_ageoption", "template_tasktype_option"],
        example_separator="\n"
    )

    prompt_text = new_prompt_template.format(
        template_userInput=query,
        template_ageoption=age_option,
        template_tasktype_option=tasktype_option
    )

    response = chat_pipeline(prompt_text, max_length=150, temperature=0.7)[0]['generated_text']
    return response

# UI Starts here
st.set_page_config(
    page_title="Marketing Tool",
    page_icon='✅',
    layout='centered',
    initial_sidebar_state='collapsed'
)
st.header("Hey, How can I help you?")

form_input = st.text_area('Enter text', height=275)

tasktype_option = st.selectbox(
    'Please select the action to be performed?',
    ('Write a sales copy', 'Create a tweet', 'Write a product description'),
    key=1
)

age_option = st.selectbox(
    'For which age group?',
    ('Kid', 'Adult', 'Senior Citizen'),
    key=2
)

numberOfWords = st.slider('Words limit', 1, 200, 25)

submit = st.button("Generate")

if submit:
    st.write(getLLMResponse(form_input, age_option, tasktype_option))
