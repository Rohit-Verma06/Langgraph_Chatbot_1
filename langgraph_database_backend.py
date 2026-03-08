from langgraph.graph import StateGraph , START,END
from langchain_openai import  OpenAIEmbeddings
from langchain_groq import ChatGroq 
# from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from typing import TypedDict ,Annotated ,Optional , Literal
import os
import tempfile
import uuid
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,AIMessage,RemoveMessage
# from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row
from psycopg_pool  import ConnectionPool
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import requests
from langgraph.prebuilt import ToolNode
from pydantic import Field , BaseModel
from typing import List 
from langgraph.store.base import BaseStore
from dotenv import load_dotenv
# import sqlite3

load_dotenv()

_THREAD_RETRIEVERS = {}
_THREAD_METADATA = {}
_THREAD_VECTORSTORES = {}
def _get_retriever(thread_id : Optional[str]):
    if(thread_id and thread_id in _THREAD_RETRIEVERS):
        return _THREAD_RETRIEVERS[thread_id]
    return None
model = ChatGroq(api_key=os.getenv("GROQ_API_KEY") , model = "llama-3.3-70b-versatile" , temperature=0)
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

tools_list = [web_search , calculator,get_stock_price ,rag_tool]
model_with_tools = model.bind_tools(tools_list,parallel_tool_calls=False)
tools = ToolNode(tools_list)

class chat_state(TypedDict):
    messages : Annotated[list[BaseMessage]  ,add_messages]

class memory_item(BaseModel):
    text : str = Field(description= "Atomic user memory as a short sentence")
    is_new : bool = Field(description= "True if this memory is New and should be stored. False if duplicate/already known")

class memory_schema(BaseModel):
    should_write : bool = Field(description= "Whether to store any memories")
    memory : List[memory_item]  = Field(default_factory=list , description= "Atomic user memories to add")

structured_model = model.with_structured_output(schema = memory_schema)

def create(state : chat_state , config : RunnableConfig , store : BaseStore):
    user_id = str(config["configurable"].get("user_id" , ""))
    namespace = ("users" , user_id , "details")
    items = store.search(namespace)
    if(items):
        previous_memory = "\n".join(item.value["data"]["text"] for item in items)
    else:
        previous_memory = "No previous memory"
    for message in state["messages"][::-1]:
        if(isinstance(message , HumanMessage)):
            last_message = message
            break
    template = PromptTemplate(
        template = """
        You are a precision memory extractor. Your goal is to find durable facts about the user.

        PREVIOUS MEMORIES:
        {previous_memory}

        USER MESSAGE:
        {message}

        TASK:
        1. Extract ALL durable facts. Look specifically for:
        - Identity: Name, Location, Bio.
        - Professional: Job, Skills, Internships.
        - Technical: Preferred programming languages (e.g., Python), frameworks, tools.
        - Projects: Current or past work.
        2. **Atomic Splitting**: If a message contains multiple facts (e.g., "I am a student AND I like Python"), treat them as separate items.
        3. **Duplicate Filter**: Compare each fact to PREVIOUS MEMORIES.
        - If the fact is totally NEW: set is_new = true.
        - If the fact is ALREADY KNOWN: set is_new = false.
        4. **Normalized Third-Person**: "I prefer python" -> "User prefers the Python programming language."

        Example: 
        Message: "My project is a chatbot and I use Python."
        If "chatbot" is already known, but "Python" is not:
        - Item 1: "User is working on a chatbot." (is_new: false)
        - Item 2: "User prefers using the Python programming language." (is_new: true)

        Return as valid JSON.
        """,
        input_variables=["message", "previous_memory"]
    )
    prompt = template.invoke({"message" : last_message , "previous_memory" : previous_memory})
    output = structured_model.invoke(prompt)
    if(output.should_write):
        for memory in output.memory:
            if(memory.is_new):
                store.put(namespace , str(uuid.uuid4()) , {"data" : memory.model_dump()})
        
    return {}

def chat(state : chat_state ,config : RunnableConfig , store : BaseStore):
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
        user_id = str(config["configurable"].get("user_id" , ""))
        namespace = ("users" , user_id , "details")
        items = store.search(namespace)
        if(items):
            existing_memories = "\n".join(item.value["data"]["text"] for item in items)
        else:
            existing_memories = "No existing_memories"
        
        template = """You are a helpful assistant with memory capabilities.
        If user-specific memory is available, use it to personalize 
        your responses based on what you know about the user.

        Your goal is to provide relevant, friendly, and tailored 
        assistance that reflects the user’s preferences, context, and past interactions.

        If the user’s name or relevant personal context is available, always personalize your responses by:
            – Always Address the user by name (e.g., "Sure, [User Name]...") when appropriate
            – Referencing known projects, tools, or preferences (e.g., "your MCP  server python based project")
            – Adjusting the tone to feel friendly, natural, and directly aimed at the user

        Avoid generic phrasing when personalization is possible. For example, instead of "In TypeScript apps..." 
        say "Since your project is built with TypeScript..."

        Use personalization especially in:
            – Greetings and transitions
            – Help or guidance tailored to tools and frameworks the user uses
            – Follow-up messages that continue from past context

        Always ensure that personalization is based only on known user details and not assumed.

        In the end suggest 3 relevant further questions based on the current response and user profile

        The user’s memory (which may be empty) is provided as: {user_details_content}
        """
        prompt = template.format(user_details_content = existing_memories)

        messages = [[SystemMessage(content = prompt)] , [system_message], *state["messages"]]
        output = model_with_tools.invoke(messages)
        return {"messages" : [output]}

def summarize(state : chat_state)->dict:
    """This node summarizes the older 8 messages if the len(messages) becomes greater than 12"""
    messages = state["messages"]
    to_remove = messages[:8]
    template = PromptTemplate(
        template = """You are an expert AI memory manager for a conversational assistant.
        Your task is to condense the provided conversation history into a single, highly concentrated summary message. 

        The history provided below may contain a previous summary, user queries, tool outputs, and AI responses. 

        Follow these strict rules:
        1. Preserve all core facts, user preferences, technical requirements, and specific entities mentioned (e.g., code snippets, locations, or data points).
        2. Eliminate conversational filler, pleasantries (e.g., "Hello", "How are you"), and redundant back-and-forth dialogue.
        3. If the history already begins with an older summary, you MUST seamlessly integrate the new information into it. Do not drop older facts just because they are from a previous summary.
        4. Write the summary from an objective, third-person perspective (e.g., "The user asked for...", "The AI provided a script for...").

        Conversation History to Summarize:
        {messages}

        Concise Updated Summary:""",
        input_variables=["messages"]
    )
    prompt = template.invoke({"messages" : to_remove})
    summary = model.invoke(prompt).content
    return {"messages" : [RemoveMessage(id = m.id) for m in to_remove] + [SystemMessage(content = summary)]}
# conn = sqlite3.connect(database = "Chatbot.db" , check_same_thread=False)
def check_condition(state : chat_state)->Literal[END , "Summarize" , "tools"]:
    last_message = state['messages'][-1]
    if(len(state["messages"]) > 12):
        return "Summarize"
    elif(hasattr(last_message , "tool_calls") and len(last_message.tool_calls)>0):
        return "tools"
    else:
        return END

# checkpointer = SqliteSaver(conn=conn)
DB_URL = os.getenv("SUPABASE_DB_URL")
pool = ConnectionPool(conninfo=DB_URL, kwargs={"autocommit": True , "row_factory": dict_row})

# try:
#     with pool.connection() as conn:
#         conn.execute("DROP TABLE IF EXISTS checkpoints_migrations CASCADE;")
#         conn.execute("DROP TABLE IF EXISTS store_migrations CASCADE;")
#         conn.execute("DROP TABLE IF EXISTS checkpoints CASCADE;")
#         conn.execute("DROP TABLE IF EXISTS checkpoint_blobs CASCADE;")
#         conn.execute("DROP TABLE IF EXISTS checkpoint_writes CASCADE;")
#         conn.execute("DROP TABLE IF EXISTS store CASCADE;")
#         print("Successfully dropped all tables and migrations!")
# except Exception as e:
#     print(f"Error dropping tables: {e}")

with pool.connection() as conn:
    # Setup Checkpointer
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    
    # Setup Store
    store = PostgresStore(conn)
    store.setup()

checkpointer = PostgresSaver(pool)

#store 
store = PostgresStore(conn = pool)


def get_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def get_user_ids():
    all_user_ids= set()
    for el in checkpointer.list(None):
        all_user_ids.add(el.config["configurable"]["user_id"])
    return list(all_user_ids)

graph = StateGraph(chat_state)
graph.add_node("Create_memory" , create)
graph.add_node("Chat" , chat)
graph.add_node("Summarize" , summarize)
graph.add_node("tools" , tools)
graph.add_edge(START,"Create_memory")
graph.add_edge("Create_memory","Chat")
graph.add_conditional_edges("Chat" , check_condition)
graph.add_edge("Summarize" , "Chat")
graph.add_edge("tools" , "Chat")
chatbot = graph.compile(checkpointer=checkpointer , store = store)

def thread_document_metadata(thread_id :str):
    return _THREAD_METADATA.get(str(thread_id) , {})