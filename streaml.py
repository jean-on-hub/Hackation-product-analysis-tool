import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from functions import *
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from streamlit_chat import message
from streamlit_image_select import image_select
# from langchain.llms import OpenAI

# Load environment variables from .env
# load_dotenv()

def get_text(n):
    input_text= st.text_input('How can I help?', '', key="input{}".format(n))
    return input_text 
def main():
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
    # agent = create_pandas_dataframe_agent(chat, iris, verbose=True, allow_dangerous_code=True)
    agent = create_pandas_dataframe_agent(chat, iris, return_intermediate_steps=True, save_charts=True, verbose=True,allow_dangerous_code=True)
    # Set up the Streamlit app
    st.title('Chatbot with Streamlit')
    st.write("Ask a question to the chatbot")
    imgs_png = glob.glob('*.png')
    imgs_jpg = glob.glob('*.jpg')
    imgs_jpeeg = glob.glob('*.jpeg')
    imgs_ = imgs_png + imgs_jpg + imgs_jpeeg
    if len(imgs_) > 0:
        img = image_select("Generated Charts/Graphs", imgs_, captions =imgs_, return_value = 'index')
        st.write(img)
    # User input
    # question = st.text_input("Ask Your question:")
    x = 0
    user_input = get_text(x)
    if st.button('Ask'):
        # if question:
        #     # Get the response from the chatbot agent
        #     try:
        #         response = agent(question)
        #         st.write(response['output'])
        #     except Exception as e:
        #         st.error(f"Error: {str(e)}")
        # else:
        #     st.write("Please enter a question.")
        x+=1
            #st.write("You:", user_input)
        print(user_input, len(user_input))
        response, thought, action, action_input, observation = run_query(agent, user_input)
        #st.write("Pandas Agent: ")
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        for i in range(0, len(thought)):
            st.sidebar.write(thought[i])
            st.sidebar.write(action[i])
            st.sidebar.write(action_input[i])
            st.sidebar.write(observation[i])
            st.sidebar.write('====')

if __name__ == "__main__":
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []
    
    if 'tabs' not in st.session_state:
        st.session_state['tabs'] = []

    main()
