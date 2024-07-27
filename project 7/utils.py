from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint

def query_agent(data, query):
    try:
        # Parse the CSV file and create a Pandas DataFrame from its contents.
        df = pd.read_csv(data)
        print(f"DataFrame loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

        # Define the Hugging Face endpoint for the language model.
        llm = HuggingFaceEndpoint(
            repo_id="huggingfaceh4/zephyr-7b-alpha"
        )
        
        # Create a Pandas DataFrame agent.
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            allow_dangerous_code=True,
            verbose=True,
            handle_parsing_errors=True  # Added to handle parsing errors
        )

        # Python REPL: A Python shell used to evaluate and execute Python commands.
        # It takes python code as input and outputs the result.
        # The input python code can be generated from another tool in the LangChain.
        output = agent.invoke(query, action="python_exec")
        print(f"Raw output from the model: {output}")

        # Check if the output includes 'Observation'
        if "Observation" in output:
            # Extract the observation part from the output
            observation = output.split("Observation")[-1].strip()
            return f"Model observation: {observation}"
        return output
    except Exception as e:
        print(f"An error occurred: {e}")
        return None