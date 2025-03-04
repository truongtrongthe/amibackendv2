from typing import Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing_extensions import TypedDict
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage  # Explicit import
import os
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from faissaccess import initialize_vector_store,faiss_index_path


class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    user_id: str

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1536)

vector_store = initialize_vector_store("tfl")
def chatbot(state: State):
    latest_message = state["messages"][-1].content
    user_id = state["user_id"]
    timestamp = datetime.now().isoformat()
    
    doc = Document(
        page_content=latest_message, 
        metadata={"timestamp": timestamp, "source": "user","user_id":user_id}
        )
    if not hasattr(chatbot, 'pending_docs'):
        chatbot.pending_docs = []
    chatbot.pending_docs.append(doc)
    
    try:
        vector_store.add_documents(chatbot.pending_docs)
        vector_store.save_local(faiss_index_path)
        chatbot.pending_docs.clear()
        print(f"FAISS updated for {user_id}.")
    except Exception as e:
        print(f"Error saving FAISS: {e}")
    
    try:
        relevant_docs = vector_store.similarity_search(
            latest_message,
            k=10,
            filter={"user_id": user_id}
        )
        if relevant_docs and len(relevant_docs)>0:
            print("Found relevant info:",len(relevant_docs))
        seen = set() #recent message is supposed to be seen
        memory_context_lines = []
        for doc in relevant_docs:
            content = doc.page_content.strip()
            if content and content not in seen and len(content)>3:
                memory_context_lines.append(f"[{doc.metadata.get('timestamp','unknown')}]{content}")
                seen.add(content)
        memory_context = "\n".join(memory_context_lines) or "No significant memories yet."

    except Exception as e:
        print(f"Error retrieving memories: {e}")
        memory_context = "Memory retrieval failed."
    print("Relevant memory:", memory_context) 

    convo_history = "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])
    #prompt = f"Conversation history:\n{memory_context}\n\nUser: {latest_message}"
    print("recent chats:",convo_history)
    prompt = f"""
    You are Ami, an assistant who can recall of everything said.
    Chat history: 
    {convo_history}
    User: {latest_message}
    Base on the recent conversation, respond to user empathetically
    """
    #print("chatbot response:",prompt)
    return {"prompt_str": prompt, "user_id": user_id}
    #return {"messages": [{"role": "assistant", "content": llm.stream(prompt)}], "user_id": user_id}
#Building graph here
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)


