import streamlit as st
import pandas as pd
# from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# Load environment variables from .env
load_dotenv()

# Access your API key
api_key = os.getenv('OPENAI_API_KEY')

# # Print the API key once to check if it's correctly loaded
# if api_key:
#     print(f"Loaded API key: {api_key}")
# else:
#     print("Failed to load API key.")
# dotenv_path = find_dotenv()
# if dotenv_path:
#     load_dotenv(dotenv_path, override=True)

print(os.getenv('MY_API_KEY'))
# Load the dataset
iris = pd.read_excel('Dummy Dataset for Challenge #1.xlsx', 'Database')

# Create the chatbot agent
chat = ChatOpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo', temperature=0.0)
agent = create_pandas_dataframe_agent(chat, iris, verbose=True, allow_dangerous_code=True)

# Set up the Streamlit app
st.title('Chatbot with Streamlit')
st.write("Ask a question to the chatbot")

# User input
question = st.text_input("Your question:")

if st.button('Ask'):
    if question:
        # Get the response from the chatbot agent
        try:
            response = agent(question)
            st.write(response['output'])
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.write("Please enter a question.")
