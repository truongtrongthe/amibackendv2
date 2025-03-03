from typing import Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing_extensions import TypedDict
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage  # Explicit import
import os
from datetime import datetime
import numpy as np

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

faiss_index_path = "faiss_index"
if os.path.exists(faiss_index_path):
    vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    initial_timestamp = datetime.now().isoformat()
    vector_store = FAISS.from_documents(
        [Document(page_content="", metadata={"timestamp": initial_timestamp, "source": "init"})],
        embeddings
    )
    vector_store.save_local(faiss_index_path)

def chatbot(state: State):
    print("Chatbot node running with state:", state)
    latest_message = state["messages"][-1].content
  
    timestamp = datetime.now().isoformat()
    doc = Document(page_content=latest_message, metadata={"timestamp": timestamp, "source": "user"})
    vector_store.add_documents([doc])
    vector_store.save_local(faiss_index_path)

    relevant_docs = vector_store.similarity_search(latest_message, k=10)
    context = "\n".join([f"[{doc.metadata.get('timestamp', 'unknown')}] {doc.page_content}" 
                        for doc in relevant_docs if doc.page_content])
    print("Context:",context)
    prompt = f"Conversation history:\n{context}\n\nUser: {latest_message}"
    #response = llm.invoke(prompt).content
    response_chunks = []
    for chunk in llm.stream(prompt):
        response_chunks.append(chunk.content)
    response = "".join(response_chunks)
    return {
        "messages": [AIMessage(content=response)]
                }
#Building graph here
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
# Compile
convo_graph = graph_builder.compile()


