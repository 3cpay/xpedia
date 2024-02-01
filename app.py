import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from angle_emb import AnglE
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs): 
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 

def get_text_chunks(rawtext):
   text_splitter = CharacterTextSplitter(
       separator = "\n",
       chunk_size = 1000,
       chunk_overlap = 200,
       length_function = len
   )
   chunks = text_splitter.split_text(rawtext)
   return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_vectorstore_gf(text_chunks):
    embeddings = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy="cls")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    st.set_page_config(page_title="Spirituspedia", 
                       page_icon=":book:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.title("Spirituspedia :books:")

    st.write(bot_template.replace("{{MSG}}","Ask me something :)"), unsafe_allow_html=True)


    #user_question = st.text_input("Your question:")
    user_question = st.chat_input("Your question:")
    if user_question:
        if hasattr(st.session_state, 'conversation') and st.session_state.conversation:
            handle_user_input(user_question)
        else:
            st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", "Please, give me some context to work by uploading PDFs on the left panel :)"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Knowledge base")
        pdf_docs = st.file_uploader("Files", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get the PDF text
                rawtext = get_pdf_text(pdf_docs)
                                
                # get the text chunks
                text_chunks = get_text_chunks(rawtext)
                # st.write(text_chunks)

                # create vector (use get_vectorstore_gf for huggingface)
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.write("Ready")


if __name__ == "__main__":
    main()