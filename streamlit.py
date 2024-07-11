import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from functions import *
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from streamlit_chat import message

# Load environment variables from .env
load_dotenv()

def get_text(n):
    input_text = st.text_input('How can I help?', '', key="input{}".format(n))
    return input_text

def main():
    # Access your API key
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        st.error("OpenAI API key is missing. Please set it in the .env file.")
        return

    # Load the dataset
    iris = pd.read_excel('Dummy Dataset for Challenge #1.xlsx', 'Database')

    # Create the chatbot agent
    chat = ChatOpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo', temperature=0.0)
    agent = create_pandas_dataframe_agent(chat, iris, return_intermediate_steps=True, verbose=True, allow_dangerous_code=True)

    # Set up the Streamlit app
    st.title('Welcome to ChatInsights')
    st.write("Ask a question to the chatbot")

    # Explain subscription tiers
    st.sidebar.title("Subscription Tiers")
    st.sidebar.write("### Free Tier:")
    st.sidebar.write("- Access to basic chatbot features")
    st.sidebar.write("- Display of dataframes")
    st.sidebar.write("### Pro Tier:")
    st.sidebar.write("- Access to all Free Tier features")
    st.sidebar.write("- Display of generated graphs and plots")

    # Subscription selection
    subscription_tier = st.sidebar.selectbox("Select your subscription tier:", ("Free", "Pro"))

    # User input
    x = 0
    user_input = get_text(x)

    if st.button('Ask'):
        if user_input:
            try:
                response, thought, action, action_input, observation, plot_objects, dataframe_objects = run_query(agent, user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
                st.session_state.plots.append(plot_objects)
                st.session_state.dataframes.append(dataframe_objects)

                for i in range(len(st.session_state['generated']) - 1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

                    if subscription_tier == "Pro":
                        if st.session_state['plots'][i]:
                            for plot in st.session_state['plots'][i]:
                                st.pyplot(plot)
                    
                    if st.session_state['dataframes'][i]:
                        for df in st.session_state['dataframes'][i]:
                            st.dataframe(df)

                # for i in range(len(thought)):
                #     st.sidebar.write(f"Thought: {thought[i]}")
                #     st.sidebar.write(f"Action: {action[i]}")
                #     st.sidebar.write(f"Action Input: {action_input[i]}")
                #     st.sidebar.write(f"Observation: {observation[i]}")
                #     st.sidebar.write('====')
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'plots' not in st.session_state:
        st.session_state['plots'] = []

    if 'dataframes' not in st.session_state:
        st.session_state['dataframes'] = []

    main()