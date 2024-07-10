# from langchain_community.chat_models import ChatOpenAI
# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# import pandas as pd
# from openai import OpenAI
# from flask import Flask, request, jsonify
# from dotenv import load_dotenv
# import os

# # Load environment variables from .env
# load_dotenv()

# # Access your API key
# api_key = os.getenv('OPENAI_API_KEY')


# # Create the Flask app
# app = Flask(__name__)


# iris = pd.read_excel('Dummy Dataset for Challenge #1.xlsx','Database')
# chat = ChatOpenAI(openai_api_key = api_key, model_name='gpt-3.5-turbo', temperature=0.0)

# agent = create_pandas_dataframe_agent(chat, iris, verbose=True, allow_dangerous_code=True)
# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     question = data.get('question')
#     if not question:
#         return jsonify({'error': 'No question provided'}), 400
#     response = agent(question)
#     return jsonify({'response': response})

# # if __name__ == '__main__':
# #     app.run(debug=True)

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)