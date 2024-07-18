
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from functions import *
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from streamlit_chat import message
import openai
import sounddevice as sd
import numpy as np
import wavio
from openai import OpenAI
client = OpenAI()
# Load environment variables from .env
load_dotenv()

# Initialize OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    openai.api_key = api_key
else:
    st.error("OpenAI API key is missing. Please set it in the .env file.")

def record_audio():
    duration = 3  # seconds
    fs = 44100  # sample rate
    st.info("Recording for 3 seconds...")

    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished

    wavio.write("recording.wav", myrecording, fs, sampwidth=2)
    return "recording.wav"

def main():
    # Load the preloaded dataset
    preloaded_iris = pd.read_excel('Dummy Dataset for Challenge #1.xlsx', 'Database')
    preloaded_iris = format_sales_figures(preloaded_iris)

    # Set up the Streamlit app
    st.title('Welcome to ChatInsights')
    st.write("Ask a question to the chatbot based on the dummy dataset provided for the hackathon.")

    # Explain subscription tiers
    st.sidebar.title("Subscription Tiers")
    st.sidebar.write("### Free Tier:")
    st.sidebar.write("- Access to basic chatbot features")
    st.sidebar.write("- Display of dataframes")
    st.sidebar.write("### Pro Tier:")
    st.sidebar.write("- Display of generated graphs and plots")
    st.sidebar.write("### Premium Tier:")
    st.sidebar.write("- Ability to upload your own datasets in place of the preloaded dataset")

    # Subscription selection
    subscription_tier = st.sidebar.selectbox("Select your subscription tier:", ("Free", "Pro", "Premium"))

    # User input
    x = 0
    user_input = get_text(x)

    user_uploaded_dfs = []
    if subscription_tier == "Premium":
        st.title("Upload your datasets")
        uploaded_files = st.file_uploader("Choose files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = load_csv_with_error_handling(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                if df is not None:
                    df = format_sales_figures(df)
                    user_uploaded_dfs.append(df)
            except Exception as e:
                st.error(f"An unexpected error occurred with file {uploaded_file.name}: {e}")
                return

    # Use user-uploaded datasets if available, otherwise use preloaded dataset
    if user_uploaded_dfs:
        combined_df = pd.concat(user_uploaded_dfs, ignore_index=True)
    else:
        combined_df = preloaded_iris

    # Create the chatbot agent
    chat = ChatOpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo', temperature=0.0)
    agent = create_pandas_dataframe_agent(chat, combined_df, return_intermediate_steps=True, verbose=True, allow_dangerous_code=True)

    if st.button('Record a question'):
        audio_file_path = record_audio()
        st.session_state['user_input'] = transcribe_audio(audio_file_path)
    
    
    
    if 'user_input' in st.session_state:
        user_input = st.session_state['user_input']
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

                    if subscription_tier in ["Pro", "Premium"]:
                        if st.session_state['plots'][i]:
                            for plot in st.session_state['plots'][i]:
                                st.pyplot(plot, clear_figure=True)
                    
                    if st.session_state['dataframes'][i]:
                        for df in st.session_state['dataframes'][i]:
                            st.dataframe(df)
                    elif subscription_tier == "Free" and st.session_state['plots'][i]:
                        st.warning("Upgrade to the Pro or Premium tier to view generated graphs and plots.")
                        break
                
                audio_response = text_to_speech(response)
                play_audio(audio_response)

            except Exception as e:
                st.error(f"Error: {str(e)}")
        del st.session_state['user_input']

    if st.button('Ask', key='ask_button'):
        st.session_state['user_input'] = user_input

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