# import streamlit as st
# import requests
# from langchain_community.chat_models import ChatOpenAI
# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# import pandas as pd
# from openai import OpenAI
# from flask import Flask, request, jsonify
# from dotenv import load_dotenv
# import os
# # Set up the Streamlit app
# st.title('Chatbot with Flask and Streamlit')
# st.write("Ask a question to the chatbot")

# # User input
# question = st.text_input("Your question:")

# # URL of the Flask backend
# backend_url = "http://127.0.0.1:5001/chat"

# if st.button('Ask'):
#     if question:
#         # Send the question to the Flask backend
#         response = requests.post(backend_url, json={'question': question})
        
#         if response.status_code == 200:
#             answer = response.json().get('response')
#             st.write(answer['output'])
#         else:
#             st.write("Error: Unable to get a response from the chatbot.")
#     else:
#         st.write("Please enter a question.")
