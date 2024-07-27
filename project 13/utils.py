from pypdf import PdfReader
import pandas as pd
import re,ast
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



#Function to extract data from text...
def extracted_data(pages_data):

    template = """Please Extract all the following values : invoice no., Description, Quantity, date, 
        Unit price , Amount, Total, email, phone number and address from this data: {pages}

        Expected output: remove any dollar symbols {{'Invoice no.': '1001329','Description': 'Office Chair','Quantity': '2','Date': '5/4/2023','Unit price': '1100.00$','Amount': '2200.00$','Total': '2200.00$','Email': 'Santoshvarma0988@gmail.com','Phone number': '9999999999','Address': 'Mumbai, India'}}
        """
    prompt_template = PromptTemplate(input_variables=["pages"], template=template)

    llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.3', temperature=0.7, huggingfacehub_api_token='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')
    full_response=llm(prompt_template.format(pages=pages_data))

    print(full_response)
    return full_response


# iterate over files in
# that user uploaded PDF files, one by one
def preprocess_text(text):
    text = text.strip('{}')  # Remove outer curly brace quotes to double quotes
    return text


def create_docs(user_pdf_list):
    df = pd.DataFrame({'Invoice no.': pd.Series(dtype='str'),
                       'Description': pd.Series(dtype='str'),
                       'Quantity': pd.Series(dtype='str'),
                       'Date': pd.Series(dtype='str'),
                       'Unit price': pd.Series(dtype='str'),
                       'Amount': pd.Series(dtype='str'),
                       'Total': pd.Series(dtype='str'),
                       'Email': pd.Series(dtype='str'),
                       'Phone number': pd.Series(dtype='str'),
                       'Address': pd.Series(dtype='str')
                       })

    for filename in user_pdf_list:
        print(f"Processing file: {filename}")
        raw_data = get_pdf_text(filename)
        print("Extracted raw data")

        llm_extracted_data = extracted_data(raw_data)

        pattern = r'{.*}'
        match = re.search(pattern, llm_extracted_data, re.DOTALL)

        if match:
            extracted_text = match.group(0)
            extracted_text = preprocess_text(extracted_text)
            extracted_text = '{' + extracted_text + '}'
            print("Extracted text before evaluation:", extracted_text)
            try:
                data_dict = ast.literal_eval(extracted_text)
                print(data_dict)
            except Exception as e:
                print(f"Error parsing extracted text: {e}")
                data_dict = {}
        else:
            print("No match found.")
            data_dict = {}

        df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
        print("********************DONE***************")

    print(df.head())
    return df