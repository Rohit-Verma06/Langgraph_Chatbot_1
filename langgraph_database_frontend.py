import streamlit as st
import os
from dotenv import load_dotenv
from langgraph_database_backend import chatbot , HumanMessage  , get_all_threads,AIMessage , thread_document_metadata,ingest_pdf,SystemMessage, get_user_ids
import uuid
load_dotenv()
def check_password():
    """Returns true if the correct password is entered by the user"""
    def enter_password():
        """Checks whether the password entered by the user is correct or not"""
        
        # FIX 1: Use .get() to completely prevent the KeyError!
        entered_pwd = st.session_state.get("password", "")
        
        # FIX 2: Check BOTH local .env and the Streamlit Cloud Secrets
        correct_pwd = os.getenv("APP_PASSWORD")
        if not correct_pwd and "APP_PASSWORD" in st.secrets:
            correct_pwd = st.secrets["APP_PASSWORD"]

        if entered_pwd == correct_pwd:
            st.session_state["password_correct"] = True
            # Safely delete the password from memory only if it exists
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Please enter the access password", type="password", on_change=enter_password, key="password")
        return False
        
    elif not st.session_state["password_correct"]:
        st.text_input("Please enter the access password", type="password", on_change=enter_password, key="password")
        st.error("😕 Password incorrect")
        return False
        
    else:
        # Password is correct! Let them in.
        return True

if not check_password():
    st.stop()
#*************************************************************Utility Functions***********************************************************
def generate_thread_id():
    thread_id = uuid.uuid4() # Universally Unique Identifier
    return thread_id

def generate_user_id():
    user_id = uuid.uuid4()
    return user_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    append_thread_id(st.session_state["thread_id"])
    # st.session_state["chat_titles"][st.session_state["thread_id"]] = "Current Chat"
    st.session_state["history"]=[]

def append_thread_id(thread_id):
    if(thread_id not in st.session_state["list_thread_ids"]):
        st.session_state["list_thread_ids"].append(thread_id)

def append_user_id(user_id):
    if(user_id not in st.session_state["list_user_ids"]):
        st.session_state["list_user_ids"].append(user_id)

def load_conversation(thread_id):
    output = chatbot.get_state(config={"configurable" : {"thread_id" : thread_id}})
    return output 
     
# def get_title(user_input):
#     prompt = f"Summarize this query into a 3-5 word title\n query->{user_input}"
#     title = model.invoke(prompt).content
#     return title

#***************************************************************Session Setup****************************************************************
if "history" not in st.session_state:
    st.session_state["history"]=[]

if 'thread_id' not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if 'user_id' not in st.session_state:
    st.session_state["user_id"] = generate_user_id()

if 'list_thread_ids' not in st.session_state:
    st.session_state["list_thread_ids"] = get_all_threads()
    if(len(st.session_state["list_thread_ids"])==0):
        st.session_state["list_thread_ids"].append(st.session_state["thread_id"])
    append_thread_id(st.session_state["thread_id"])

if 'list_user_ids' not in st.session_state:
    st.session_state["list_user_ids"] = get_user_ids()
    if(len(st.session_state["list_user_ids"]) == 0):
        st.session_state["list_user_ids"].append(st.session_state["user_id"])
    append_user_id(st.session_state["user_id"])
    
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"]={}


thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key , {})


# if "chat_titles" not in st.session_state:
#     st.session_state["chat_titles"] = {}
#     all_threads = st.session_state["list_thread_ids"]
#     for thread_id in all_threads:
#         # 1. Fetch the conversation history for this thread
#         state = load_conversation(thread_id)
        
#         # 2. Check if there are messages to generate a title from
#         if state and state.values and "messages" in state.values:
#             messages = state.values["messages"]
#             # Filter for HumanMessages to find what the user asked
#             human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
            
#             if human_msgs:
#                 # 3. Use the first 4-5 words of the first message as the title
#                 # (We use simple text slicing here to keep startup fast, avoiding LLM calls)
#                 content = human_msgs[0].content
#                 restored_title = " ".join(content.split()[:5]) + "..."
#                 st.session_state["chat_titles"][thread_id] = restored_title
#             else:
#                 st.session_state["chat_titles"][thread_id] = "Past Chat"
#         else:
#             st.session_state["chat_titles"][thread_id] = "Current Chat"
    
    

#***************************************************************Sidebar UI********************************************************************
st.sidebar.title("Multiutility LangGraph Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{str(st.session_state['thread_id'])}`")
if(st.sidebar.button("New Chat", use_container_width=True)):
    reset_chat()
    st.rerun()

if(thread_docs):
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using {latest_doc.get('filename')}"
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No pdf ingested yet")

uploaded_pdf = st.sidebar.file_uploader("Upload a pdf for this chat" , type= ["pdf"])
if(uploaded_pdf):
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"The file {uploaded_pdf.name} already processed for this chat")
    else:
        with st.sidebar.status("Indexing pdf...",expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_key,
                uploaded_pdf.name
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label = "✅ PDF indexed" , state = "complete", expanded=False)

st.sidebar.subheader("Past conversations")
if(not st.session_state["list_thread_ids"]):
    st.sidebar.write("No past conversations yet")
else:
    for thread_id in st.session_state["list_thread_ids"][::-1]:
        if(st.sidebar.button(str(thread_id) )):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)
            if(messages and messages.values):
                temp_message = []
                for message in messages.values["messages"]:
                    if(isinstance(message , HumanMessage)):
                        role = "user"
                    elif(isinstance(message , AIMessage)):
                        role = "assistant"
                    else:
                        continue
                    temp_message.append({"role" : role , "content" : message.content})
                if(len(temp_message)>0 and temp_message[-1]["role"] == "user"):
                    temp_message = temp_message[:-1]
                st.session_state["history"] = temp_message
                st.session_state["ingested_docs"].setdefault(str(thread_id) , {})
                st.rerun()
            else:
                st.session_state["history"]=[]
                st.rerun()
                st.info("No Conversation happened")
    

for el in st.session_state["history"]:
    if(el["content"]):
        with st.chat_message(el["role"]):
            st.markdown(el["content"])


user = st.chat_input("Type here")
config = {"configurable" : {"thread_id" : st.session_state["thread_id"] ,"user_id" : st.session_state["user_id"]} , "metadata": {"thread_id" : st.session_state["thread_id"]}}
if(user):
    st.session_state["history"].append({"role" : "user" , "content" : user})
    with st.chat_message("user"):
        st.markdown(user)
    
    
    stream = chatbot.stream({'messages' : HumanMessage(user)} , config=config , stream_mode="messages")

    with st.chat_message("assistant"):
        
        output = st.write_stream(
            chunk.content for chunk , metadata in stream if(isinstance(chunk, AIMessage))
        )

    st.session_state["history"].append({"role" : "assistant" , "content" : output})
    doc_meta = thread_document_metadata(thread_key)
    if(doc_meta):
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

    # if(st.session_state["chat_titles"][st.session_state["thread_id"]] == "Current Chat"):
    #     st.session_state["chat_titles"][st.session_state["thread_id"]] = get_title(user)
    #     st.rerun()
