###pip install streamlit langchain langchain-openai langchain_community beautifulsoup4 python-dotenv cromadb 

# Chat with website 
import streamlit as st 
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import  load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

#get_response of user query function 
def get_response(usr_input):
    #comment the if condition when actual porcess happens 
    if usr_input  :
        usr_input = ""
    return("Sorry, I dont know")

#URL to Vector Store 
def get_vectorstore_from_url(url):
    #website document to text form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    #split document
    text_splitter = RecursiveCharacterTextSplitter() 
    document_chunk = text_splitter.split_documents(document)
    
    #create vector store from chunks 
    url_vector_store = chroma.from_documnts(document_chunk, OpenAIEmbeddings()) 
    
    return(url_vector_store)

def get_context_retriever_chain(vector_stores):
    llm = ChatOpenAI()
    
    retriever = vector_stores.as_retriever()

    prompt = ChatPromptTemplate.from_messages([]
        
    )


def put_on_chat_sidebar(sidetxt):
#Chat Debug using sidebar
    with st.sidebar:
        st.write(sidetxt)

    return()

#App Config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")

st.title("Chat with websites")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

#Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_area("Website URL")
if website_url is None or website_url == "":
    st.info("Please enter a website to use the Bot.")

else: 
    #Website to text 
    url_documents = get_vectorstore_from_url(website_url)
    put_on_chat_sidebar(url_documents)  
           
    #User Input    
    user_query = st.chat_input("Type your message here ...")

    #Web Messaging 
        
    if user_query !="" and user_query is not None:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        
    #Chat Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)        
                