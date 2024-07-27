import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

load_dotenv()

# Set the Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Configure the Streamlit page
st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things")

# Initialize the Hugging Face Embeddings object
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load CSV data
loader = CSVLoader(file_path='myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})
data = loader.load()

# Display the data
print(data)

# Create a FAISS index from the documents
db = FAISS.from_documents(data, embeddings)

# Function to receive input from user and store it in a variable
def get_text():
    input_text = st.text_input("You: ", key='input')
    return input_text

user_input = get_text()
submit = st.button('Find similar Things')

if submit:
    # If the button is clicked, fetch similar text
    docs = db.similarity_search(user_input)
    print(docs)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content if docs else "No matches found.")
    if len(docs) > 1:
        st.text(docs[1].page_content)