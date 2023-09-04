
import streamlit as st
from streamlit_chat import message
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Dict
import json
from io import StringIO
from random import randint
import boto3
from langchain import PromptTemplate
import os
import sys
import logging

st.set_page_config(page_title="Retrieval Augmented Generation with AWS", page_icon=":robot:", layout="wide")
st.header("RAG Chatbot with Amazon Bedrock and Amazon Kendra")

logging.basicConfig(filename='./rag.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logging.info("Iniciando sesión")

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

# AUTHENTICATION
import streamlit as st
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
name, authentication_status, username = authenticator.login('Login', 'main')
if authentication_status:
# BEDROCK ----------
    boto3_bedrock = bedrock.get_bedrock_client(
        endpoint_url='https://bedrock.us-east-1.amazonaws.com',
        region='us-east-1',
        profile_name='bedrock'
    )

    # KENDRA ------------
    kendra_index_id = '523d11ff-3547-4270-ac48-eca3b1b875fc' ###REPLACE WITH YOUR OWN KENDRA INDEX ID
    region='eu-west-1'
    boto3_kendra = bedrock.get_kendra_client(
        assumed_role='arn:aws:iam::889960878219:role/service-role/AmazonKendra-rodzanto', ###REPLACE WITH YOUR OWN ROLE
        region=region
    )


    # - create the Anthropic Model
    llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':8000})

    from langchain.retrievers import AmazonKendraRetriever
    from langchain.llms.bedrock import Bedrock

    from langchain.chains.question_answering import load_qa_chain
    from langchain.vectorstores import Pinecone
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper

    # LANGCHAIN ------------
    prompt_template = """
    Human: Considera la conversación, tu propio conocimiento, y el contexto en los tags XML <context></context> para responder a la pregunta 'Question' de forma simple y en Español.
    No empieces tu respuesta con la frase 'Según el contexto', solo provee la respuesta directamente.

    <context>
    {context}
    </context>

    Question: {question}
    Assistant: Answer:
    """
    #Si no sabes la respuesta a partir del contexto o tu propio conocimiento, solo responde 'Lo siento pero no tengo esa información'. No intentes inventar una respuesta.


    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=AmazonKendraRetriever(
            index_id=kendra_index_id,
            region_name=region,
            client=boto3_kendra,
            attribute_filter={
                'EqualsTo': {
                    'Key': '_language_code',
                    'Value': {'StringValue': 'es'}
                }
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )



    #@st.cache_resource
    #def load_chain(_prompt):
    chatchain = RetrievalQA.from_chain_type(
            llm = Bedrock(
                #model_id ='anthropic.claude-instant-v1'
                model_id='anthropic.claude-v2',
                client=boto3_bedrock,
                model_kwargs={
                    'max_tokens_to_sample':8000,
                    'temperature':0,
                    'top_p':0.9,
                    'stop_sequences': ["Human"]
                }
            ),
            chain_type="stuff",
            retriever=AmazonKendraRetriever(
                index_id=kendra_index_id,
                region_name=region,
                client=boto3_kendra,
                attribute_filter={
                    'EqualsTo': {
                        'Key': '_language_code',
                        'Value': {'StringValue': 'es'}
                    }
                }
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    #    return chain

    #chatchain = load_chain(prompt)

    # initialise session variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'widget_key' not in st.session_state:
        st.session_state['widget_key'] = str(randint(1000, 100000000))

    # Sidebar...
    st.markdown(
        f"""
        <style>
        .sidebar .sidebar-content {{
            width: 100px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <style>
        [data-testid=stSidebar] [data-testid=stImage] {{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("<p style='text-align: center;'><b>CUSTOMER NAME</b></p>", unsafe_allow_html=True) ###REPLACE WITH CUSTOMER NAME
    st.sidebar.divider()
    st.sidebar.markdown('Powered by')
    st.sidebar.image('./images/AWS_logo_RGB.png', width=50)

    # this is the container that displays the past conversation
    response_container = st.container()
    # this is the container with the input text box
    container = st.container()

    with container:
        # define the input text box
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Usted:", key='input', height=50)
            submit_button = st.form_submit_button(label='Enviar')

        # when the submit button is pressed we send the user query to the chatchain object and save the chat history
        if submit_button and user_input:
            #input_prompt = prompt.format(
            #    user_input=user_input,
            #)
            output = chatchain({"query": user_input})
            for i in output["source_documents"]:
                sources = i.metadata["source"]
            #output = chatchain(input_prompt)["response"]
            st.session_state['past'].append(user_input)
            if sources:
                st.session_state['generated'].append(output["result"] + "Referencia:\n" + sources)
                logging.info(f'Respuesta: {output["result"]} Referencia:\n {sources}')
                sources = ""
            else:
                st.session_state['generated'].append(output["result"])
                logging.info(f'Respuesta: {output["result"]}')

    # this loop is responsible for displaying the chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer", seed=120)
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed=123)
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
