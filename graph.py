from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.docstore.document import Document
import os
from datetime import datetime

llm = ChatOpenAI(model="gpt-4.5-preview", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Requires OpenAI API key

class State(TypedDict):
    messages: Annotated[list, add_messages]

faiss_index_path = "faiss_index"
if os.path.exists(faiss_index_path):
    vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    # Initialize with a dummy document that has metadata
    initial_timestamp = datetime.now().isoformat()
    vector_store = FAISS.from_documents(
        [Document(page_content="", metadata={"timestamp": initial_timestamp, "source": "init"})],
        embeddings
    )
    vector_store.save_local(faiss_index_path)  # Save it right away

graph_builder = StateGraph(State)


def chatbot(state: State):
    # Get the latest user message
    latest_message = state["messages"][-1].content
    # Create a document with timestamp
    timestamp = datetime.now().isoformat()
    doc = Document(page_content=latest_message, metadata={"timestamp": timestamp, "source": "user"})
    
    # Add to FAISS
    vector_store.add_documents([doc])
    
    # Save to disk after each update (persistent)
    vector_store.save_local(faiss_index_path)
    
    # Query FAISS for relevant memories (top 5)
    relevant_docs = vector_store.similarity_search(latest_message, k=5)

    context_lines = []

    for doc in relevant_docs:
        timestamp = doc.metadata.get("timestamp", "unknown")  # Fallback if timestamp missing
        content = doc.page_content
        if content:  # Skip empty dummy content
            context_lines.append(f"[{timestamp}] {content}")
    context = "\n".join(context_lines)
    prompt = f"Conversation history:\n{context}\n\nUser: {latest_message}"
    # Generate response
    response = llm.invoke(prompt)
    return {"messages": [response]}


graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

g_app = graph_builder.compile()

