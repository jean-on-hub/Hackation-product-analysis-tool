import glob
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from openai import OpenAI
client = OpenAI()
import streamlit as st

def load_dataframe():
    selected_df = []
    all_files_csv = glob.glob("*.csv")
    all_files_xlsx = glob.glob("*.xlsx")
    all_files_xls = glob.glob("*.xls")

    for filename in all_files_csv:
        df = pd.read_csv(filename)
        selected_df.append(df)
    for filename in all_files_xlsx:
        df = pd.read_excel(filename)
        selected_df.append(df)
    for filename in all_files_xls:
        df = pd.read_excel(filename)
        selected_df.append(df)

    selected_df_names = all_files_csv + all_files_xlsx + all_files_xls
    return selected_df, selected_df_names

def run_query(agent, query_):
    output = agent(query_)
    response, intermediate_steps = output['output'], output['intermediate_steps']
    thought, action, action_input, observation, steps = decode_intermediate_steps(intermediate_steps)
    store_convo(query_, steps, response)
    
    # Extract plot and dataframe objects if any
    plot_objects = extract_plot_objects(intermediate_steps)
    dataframe_objects = extract_dataframe_objects(intermediate_steps)
    
    return response, thought, action, action_input, observation, plot_objects, dataframe_objects

def decode_intermediate_steps(steps):
    log, thought_, action_, action_input_, observation_ = [], [], [], [], []
    text = ''
    for step in steps:
        action_details = step[0]  # Assuming step[0] is an AgentAction object
        thought_.append(action_details.log.split('Action:')[0].strip())
        action_.append(action_details.log.split('Action:')[1].split('Action Input:')[0].strip())
        action_input_.append(action_details.log.split('Action:')[1].split('Action Input:')[1].strip())
        observation_.append(step[1])
        log.append(action_details.log)
        text = action_details.log + ' Observation: {}'.format(step[1])
    return thought_, action_, action_input_, observation_, text

def extract_plot_objects(intermediate_steps):
    plot_objects = []
    for step in intermediate_steps:
        action_details = step[0]
        if 'plot' in action_details.log or 'chart' in action_details.log or 'graph' in action_details.log:
            plot_objects.append(plt.gcf())
    plt.close('all')  # Close all figures to prevent re-plotting
    return plot_objects

def extract_dataframe_objects(intermediate_steps):
    dataframe_objects = []
    for step in intermediate_steps:
        if isinstance(step[1], pd.DataFrame):
            dataframe_objects.append(step[1])
    return dataframe_objects

def get_convo():
    convo_file = 'convo_history.json'
    with open(convo_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data, convo_file

def store_convo(query, response_, response):
    data, convo_file = get_convo()
    current_dateTime = datetime.now()
    data['{}'.format(current_dateTime)] = []
    data['{}'.format(current_dateTime)].append({'Question': query, 'Answer': response, 'Steps': response_})

    with open(convo_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def format_sales_figures(df):
    # Assuming the column to format is named 'sales'
    if 'sales' in df.columns:
        df['sales'] = df['sales'].apply(lambda x: "{:,.2f}".format(x))
    return df

def load_csv_with_error_handling(file):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(file, encoding=encoding)
            return df
        except Exception as e:
            continue
    st.error(f"Failed to read the file {file.name} with the available encodings.")
    return None



def get_text(n):
    input_text = st.text_input('Type your question here.', '', key="input{}".format(n), on_change=submit_question, args=(n,))
    return input_text

def submit_question(n):
    if st.session_state.get(f'input{n}', ''):
        st.session_state['user_input'] = st.session_state[f'input{n}']
        st.session_state[f'input{n}'] = ''



def transcribe_audio(file_path):
    audio_file = open(file_path, "rb")
    transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file,
  response_format="text"
)
    return transcription

def text_to_speech(text):
   with   client.audio.speech.with_streaming_response.create(
  model="tts-1",
  voice="alloy",
  input=text,
  response_format='aac'
) as response:
    response.stream_to_file("response.mp3")
    
    return "response.mp3"

def play_audio(file_path):
    audio_file = open(file_path, "rb")
    st.audio(audio_file.read(), format="audio/mp3")