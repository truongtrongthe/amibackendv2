from typing import Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing_extensions import TypedDict
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langchain_core.messages import AIMessage, HumanMessage  # Explicit import
import os
from datetime import datetime
import numpy as np

class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_topic: str
    topic_embedding: list
    interrupt: dict  # For HITL interrupts

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


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def chatbot(state: State):
    print("Chatbot node running with state:", state)
    latest_message = state["messages"][-1].content
    current_topic = state.get("current_topic")
    topic_embedding = state.get("topic_embedding")

    if "what topics" in latest_message.lower():
        topic_docs = vector_store.similarity_search("", k=100, filter={"source": "summary"})
        topics = [f"{doc.metadata.get('timestamp')} - {doc.metadata.get('topic', 'Unnamed')}" 
                  for doc in topic_docs if "topic" in doc.metadata]
        response = "Here are your historic topics:\n" + "\n".join(topics) if topics else "No topics saved yet."
        print("Chatbot response:", response)
        return {"messages": [AIMessage(content=response)]}

    latest_embedding = embeddings.embed_query(latest_message)

    if current_topic and topic_embedding:
        similarity = cosine_similarity(latest_embedding, topic_embedding)
        if similarity < 0.7:
            response = f"New topic detected! Save '{current_topic}' and brainstorm '{latest_message}'? (Yes/No)"
            print("Chatbot response (interrupt):", response)
            return {
                "messages": [AIMessage(content=response)],
                "interrupt": {"query": "save_topic", "current_topic": current_topic, "new_topic": latest_message}
            }

    timestamp = datetime.now().isoformat()
    doc = Document(page_content=latest_message, metadata={"timestamp": timestamp, "source": "user"})
    vector_store.add_documents([doc])
    vector_store.save_local(faiss_index_path)

    relevant_docs = vector_store.similarity_search(latest_message, k=5)
    context = "\n".join([f"[{doc.metadata.get('timestamp', 'unknown')}] {doc.page_content}" 
                        for doc in relevant_docs if doc.page_content])
    prompt = f"Conversation history:\n{context}\n\nUser: {latest_message}"
    #response = llm.invoke(prompt).content
    response_chunks = []
    for chunk in llm.stream(prompt):
        response_chunks.append(chunk.content)
    response = "".join(response_chunks)
    print("Chatbot response:", response)
    return {
        "messages": [AIMessage(content=response)],
        "current_topic": latest_message if not current_topic else current_topic,
        "topic_embedding": latest_embedding if not topic_embedding else topic_embedding
    }


def handle_interrupt(state: State):
    print("Handle interrupt running with state:", state)
    interrupt_data = state.get("interrupt")
    if interrupt_data and interrupt_data["query"] == "save_topic":
        user_response = state["messages"][-1].content.lower()
        current_topic = interrupt_data["current_topic"]
        new_topic = interrupt_data["new_topic"]
        
        if "yes" in user_response:
            history_docs = vector_store.similarity_search("", k=10)
            history = "\n".join([doc.page_content for doc in history_docs if doc.page_content])
            summary_prompt = f"Summarize this conversation about {current_topic}:\n{history}"
            summary = llm.invoke(summary_prompt).content
            doc = Document(
                page_content=summary,
                metadata={"timestamp": datetime.now().isoformat(), "source": "summary", "topic": current_topic}
            )
            vector_store.add_documents([doc])
            vector_store.save_local(faiss_index_path)
            response = f"Saved '{current_topic}'. Brainstorming {new_topic}:\n" + llm.invoke(f"Brainstorm ideas for: {new_topic}").content
        elif "no" in user_response:
            response = f"Discarded '{current_topic}'. Brainstorming {new_topic}:\n" + llm.invoke(f"Brainstorm ideas for: {new_topic}").content
        else:
            response = "Please say 'yes' or 'no'."
            print("Handle interrupt response:", response)
            return {"messages": [AIMessage(content=response)], "interrupt": interrupt_data}

        print("Handle interrupt response:", response)
        return {
            "messages": [AIMessage(content=response)],
            "current_topic": new_topic,
            "topic_embedding": embeddings.embed_query(new_topic),
            "interrupt": None
        }
    return state

#Building graph here
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("handle_interrupt", handle_interrupt)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    lambda state: "handle_interrupt" if state.get("interrupt") else "chatbot",
    {"handle_interrupt": "handle_interrupt", "chatbot": END}
)
graph_builder.add_edge("handle_interrupt", "chatbot")
# Compile
g_app = graph_builder.compile()


