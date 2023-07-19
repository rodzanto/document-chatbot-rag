
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

st.set_page_config(page_title="Retrieval Augmented Generation", page_icon=":robot:", layout="wide")
st.header("Document Insights Chatbot with Amazon Bedrock")

bedrock = boto3.client(
 service_name='bedrock',
 region_name='us-east-1',
 endpoint_url='https://bedrock.us-east-1.amazonaws.com'
)

# PINECONE ------------

# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# - create the Anthropic Model
llm = Bedrock(model_id="anthropic.claude-v1", client=bedrock, model_kwargs={'max_tokens_to_sample':8000})
bedrock_embeddings = BedrockEmbeddings(client=bedrock)

import pinecone
# find API key in console at app.pinecone.io
YOUR_API_KEY = "c8f831ec-e5ab-43a0-a6c8-ac62364a192e" 
# find ENV (cloud region) next to API key in console
YOUR_ENV = "us-west4-gcp-free" 
index_name = 'langchain-retrieval-agent'
pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV
)

from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

vectorstore_pinecone = Pinecone.from_existing_index(
    embedding=bedrock_embeddings,
    index_name = index_name
)

wrapper_store_pinecone = VectorStoreIndexWrapper(vectorstore=vectorstore_pinecone)


# LANGCHAIN ------------

#qa = RetrievalQA.from_chain_type(

prompt_template = """
Human: Consider the context in the <context></context> XML tags and your own knowledge, to answer the question at the end.

<context>
{context}
</context>

Question: {question}
Assistant:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

#@st.cache_resource
#def load_chain(_prompt):
chatchain = RetrievalQA.from_chain_type(
        llm = Bedrock(
            #model_id ='anthropic.claude-instant-v1'
            model_id='anthropic.claude-v1',
            client=bedrock,
            model_kwargs={
                'max_tokens_to_sample':8000,
                'temperature':0,
                'top_p':0.9,
                'stop_sequences': ["Human"]
            }
        ),
        chain_type="stuff",
        retriever=vectorstore_pinecone.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
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

# Sidebar - the clear button is will flush the memory of the conversation
#st.sidebar.title("Sidebar")
#st.sidebar.image('./images/AWS_logo_RGB.png', width=150)
st.sidebar.image('./images/miniclip_logo.png', width=150)

st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 150px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

# this is the container that displays the past conversation
response_container = st.container()
# this is the container with the input text box
container = st.container()

with container:
    # define the input text box
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=50)
        submit_button = st.form_submit_button(label='Send')

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
            st.session_state['generated'].append(output["result"] + "Source document(s):\n" + sources)
            sources = ""
        else:
            st.session_state['generated'].append(output["result"])

# this loop is responsible for displaying the chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer", seed=120)
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed=123)
