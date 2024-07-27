#from langchain.llms import OpenAI
#The above is no longer avialable, so replaced it with the below import :)
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEndpoint

# Function to generate video script
def generate_script(prompt,video_length,creativity,api_key):
    
    # Template for generating 'Title'
    title_template = PromptTemplate(
        input_variables = ['subject'], 
        template='Please come up with a title for a YouTube video on the  {subject}.'
        )

    # Template for generating 'Video Script' using search engine
    script_template = PromptTemplate(
        input_variables = ['title', 'DuckDuckGo_Search','duration'], 
        template='Create a script for a YouTube video based on this title for me. TITLE: {title} of duration: {duration} minutes using this search data {DuckDuckGo_Search} '
    )

    # #Setting up OpenAI LLM
    # llm = ChatOpenAI(temperature=creativity,openai_api_key=api_key,
    #         model_name='gpt-3.5-turbo') 
    llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=api_key
        )
    
    #Creating chain for 'Title' & 'Video Script'
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)

    
    # https://python.langchain.com/docs/modules/agents/tools/integrations/ddg
    search = DuckDuckGoSearchRun()

    # Executing the chains we created for 'Title'
    data = title_chain.invoke(prompt)
    text = data['text']
    start_index = text.find("Title: \"") + len("Title: \"")
    end_index = text.find("\"", start_index)
    title = text[start_index:end_index]
    # Executing the chains we created for 'Video Script' by taking help of search engine 'DuckDuckGo'
    search_result = search.run(prompt) 
    script = script_chain.run(title=title, DuckDuckGo_Search=search_result,duration=video_length)

    # Returning the output
    return search_result,title,script