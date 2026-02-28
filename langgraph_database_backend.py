from langgraph.graph import StateGraph , START,END
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_groq import ChatGroq 
# from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from typing import TypedDict ,Annotated ,Optional
import os
import tempfile
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,AIMessage
# from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.tools import tool
import requests
from langgraph.prebuilt import tools_condition ,ToolNode
from dotenv import load_dotenv
import sqlite3

load_dotenv()

_THREAD_RETRIEVERS = {}
_THREAD_METADATA = {}
_THREAD_VECTORSTORES = {}
def _get_retriever(thread_id : Optional[str]):
    if(thread_id and thread_id in _THREAD_RETRIEVERS):
        return _THREAD_RETRIEVERS[thread_id]
    return None
model = ChatGroq(model = "llama-3.3-70b-versatile" , temperature=0)
# model = ChatOpenAI()

@tool 
def web_search(query : str):
    """This tool does Web Search for a given query"""
    return DuckDuckGoSearchRun(region = "us-en").invoke(query)

@tool
def calculator(num1 : float , num2 : float , operation :str)->dict:
    """This tool does mathematical operations which includes addition,subtraction,multiplication,division"""
    try:
        if(operation == "addition"):
            result = num1+num2
        elif(operation == "subtraction"):
            result = num1-num2
        elif(operation == "multiplication"):
            result = num1*num2
        elif(operation == "division"):
            if(num2==0):
                return {"error" : "Division by zero is not defined"}
            else:
                result = num1/num2
        else:
            print("unsupported operation type -> " ,operation)
        return {"result" : result} 
    except Exception as e:
        return {"error" : str(e)}
@tool
def get_stock_price(company : str)->dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL' , 'TSLA)
    using Finnhub with API key in the url"""
    api_key = os.getenv("FINNHUB_API_KEY")
    url = f"https://finnhub.io/api/v1/quote?symbol={company}&token={api_key}"
    response = requests.get(url).json()
    return response

def ingest_pdf(file_bytes : bytes , thread_id : str , filename : Optional[str] = None)->dict:
    """Build a FAISS retriever for the uploaded pdf and store it for the thread 
    Return a summary dict which can be surfaced in the UI."""
    if not file_bytes:
        raise ValueError("No files uploaded for ingestion")
    
    with tempfile.NamedTemporaryFile(delete=False , suffix = ".pdf") as temp_file:
        temp_file.write(file_bytes)
        file_path = temp_file.name
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(separators = ["\n\n" , "\n" , " ", ""] , chunk_size = 1000,chunk_overlap = 200)
        chunks = splitter.split_documents(docs)

        if(str(thread_id) in _THREAD_VECTORSTORES):
            vector_store = _THREAD_VECTORSTORES[str(thread_id)]
            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=OpenAIEmbeddings(model = "text-embedding-3-small")
            )
            _THREAD_VECTORSTORES[str(thread_id)] =vector_store
            
        retriever = vector_store.as_retriever(search_type="similarity" , search_kwargs = {"k" : 4})
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        if(str(thread_id) in _THREAD_METADATA):
            existing_metadata = _THREAD_METADATA[str(thread_id)]
            _THREAD_METADATA[str(thread_id)] = {
                "filename" : f"{existing_metadata['filename']} , {filename or os.path.basename(file_path)}",
                "documents" : existing_metadata["documents"] + len(docs),
                "chunks" : existing_metadata["chunks"] + len(chunks)
            }
        else:
            _THREAD_METADATA[str(thread_id)] = {
            "filename" : filename or os.path.basename(file_path),
            "documents" : len(docs),
            "chunks" : len(chunks)
        }
        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


@tool
def rag_tool(query :str , thread_id : Optional[str] = None):
    """Retriever relevant document from the pdf document
    Use this tool when the user asks factual or conceptual questions that might be answered from the stored documents"""
    retriever = _get_retriever(thread_id)
    if(retriever is None):
        return {
            "error" : "No document indexed for this chat. Upload a pdf first",
            "query" : query
        }
    result = retriever.invoke(query)
    context = [d.page_content for d in result]
    metadata = [d.metadata for d in result]
    return {
        "query" : query,
        "context" : context,
        "metadata" : metadata
    }

tools = [web_search , calculator,get_stock_price ,rag_tool]
model_with_tools = model.bind_tools(tools,parallel_tool_calls=False)

tools = ToolNode(tools)
class chat_state(TypedDict):
    messages : Annotated[list[BaseMessage]  ,add_messages]

def chat(state : chat_state ,config=None)->chat_state:
        """LLM node that may answer or request for a tool call"""
        if(config and isinstance(config, dict)):
            thread_id = config.get("configurable" , {}).get("thread_id")
        system_message = SystemMessage(
            content=(
                "### ROLE\n"
                "You are a versatile assistant. You answer general knowledge directly OR use tools for specific data.\n\n"
                "### TOOL RULES\n"
                f"- For questions about PDFs, ONLY use `rag_tool` with thread_id: '{thread_id}'.\n"
                "- For math, use `calculator`. For stocks, use `get_stock_price`. For web info, use `web_search` tool.\n\n"
                "### CRITICAL - AVOID API ERRORS\n"
                "- IF the user asks for an essay, poem, recipe, or general code: DO NOT CALL ANY TOOLS.\n"
                "- DO NOT say 'Let me check the document' or 'I will search for that'.\n"
                "- If you decide to answer directly, START typing the answer immediately. NO PREAMBLE.\n"
                "- If you decide to use a tool, TRIGGER it immediately. NO PREAMBLE."
            )
        )

        messages = [system_message, *state["messages"]]
        output = model_with_tools.invoke(messages)
        return {"messages" : [output]}
conn = sqlite3.connect(database = "Chatbot.db" , check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

def get_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

graph = StateGraph(chat_state)
graph.add_node("Chat" , chat)
graph.add_node("tools" , tools)
graph.add_edge(START,"Chat")
graph.add_conditional_edges("Chat" , tools_condition)
graph.add_edge("tools" , "Chat")
chatbot = graph.compile(checkpointer=checkpointer)

def thread_document_metadata(thread_id :str):
    return _THREAD_METADATA.get(str(thread_id) , {})