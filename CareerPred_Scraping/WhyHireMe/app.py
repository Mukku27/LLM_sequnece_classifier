import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

# To run LLM locally
import torch
import transformers
from transformers import AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Langchain functions
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss, chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOllama
from langchain.llms.huggingface_pipeline import HuggingFacePipeline


from htmlTemplates import css, bot_template, user_template

# By default, models stored in .cache but they are removed. For persistent store
os.environ['TRANSFORMERS_CACHE'] = 'C:/D_drive/UCSD/Projects/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'C:/D_drive/UCSD/Projects/huggingface_cache'


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # pdf_reader creates pages that we can loop through and extract
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    "Returns list of text chunks"
    text_splitter = CharacterTextSplitter(
        separator='\n', 
        chunk_size = 500, # 1000 characters
        chunk_overlap = 50, # To keep redundant information if chunk ends abruptly
        length_function = len
        )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstores(text_chunks):
    # This downloads the embeddings on system. So we are not sending anything to Huggingface
    embeddings =  HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_llm():
    """
        get_conversation_chain is called once so this function should also be called once
    """
    model_id = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    pipeline = transformers.pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=400,
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline = pipeline)

    return llm


def get_conversation_chain(vectorstore):
    
    # since chatbot has memory, we initialize instance of memory
    # import ConversationalBufferMemory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # study different memory types
    # Get llm. 
    # llm = ChatOpenAI() # Only works if OpenAI subscription
    llm = get_llm()

    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conv_chain


def handle_userInput(user_question):
    # conversation already contains vectorstore and memory
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # st.write(response)
    for i, message in enumerate(st.session_state.chat_history[::-1]):
        if i%2 == 1:
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
    

def main():
    # The variables from .env are loaded in env
    load_dotenv()
    st.set_page_config(page_title='Chat with Docs', page_icon=':books:')
    st.write(css, unsafe_allow_html=True) # To use custom formating
    
    st.header("Chat with Docs :books: ")
    user_question = st.text_input('Ask a question about the uploaded docs')
    if user_question:
        handle_userInput(user_question)

    # We want conversation to be persistent
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    # Need to initialize chat_history
    if 'chat_history' not in st.session_state: 
        st.session_state.chat_history = None


    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs here and click on 'Process' ", accept_multiple_files=True)
        
        # st.button becomes true when pressed
        if st.button("Process"):
            # User sees a spining wheel while this part is running
            with st.spinner('Processing'):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                # create vector store
                vectorstore = get_vectorstores(text_chunks)

                # Creating a conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                # streamlit app reruns entire code on every interaction
                # If we want a variable to be persistant in the session, store in st.session_state and it won't be re-init


                

if __name__ == '__main__':
    main()

