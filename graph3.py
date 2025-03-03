from langgraph.graph import StateGraph
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from typing import Dict
import json

# AI model for processing responses
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# In-memory LangMem (Replace with a real vector DB later)
langmem_store = []

# Define memory state
class ConversationState:
    history: list
    extracted_knowledge: str = ""

def start_conversation(state: Dict):
    return {
        "history": state.get("history", []) + [SystemMessage(content="I'm AMI, your AI assistant. Can you share a sales technique that works well for you?")]
    }

def extract_knowledge(state: Dict):
    last_message = state["history"][-1].content  
    response = llm.invoke(f"Summarize the key learning from this expert insight: {last_message}")
    
    return {
        "history": state["history"] + [AIMessage(content=f"Got it! Here's what I learned: {response}")],
        "extracted_knowledge": response
    }
def request_clarification(state: Dict):
    return {
        "history": state["history"] + [HumanMessage(content="That sounds great! Can you provide a concrete example?")]
    }
def confirm_and_store(state: Dict):
    knowledge = state["extracted_knowledge"]
    
    # Ask expert to confirm
    confirmation_prompt = f"I've recorded this insight: {knowledge}\nDoes this look accurate? (yes/no)"
    
    return {
        "history": state["history"] + [AIMessage(content=confirmation_prompt)]
    }
def store_in_langmem(state: Dict):
    knowledge = state["extracted_knowledge"]
    
    # Store in a structured format
    langmem_entry = {
        "topic": "Sales Techniques",
        "insight": knowledge,
        "source": "Expert Conversation"
    }
    langmem_store.append(langmem_entry)
    
    return {
        "history": state["history"] + [AIMessage(content="Thank you! I've saved this insight for future use.")],
        "extracted_knowledge": ""
    }

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Load FAISS database
vectorstore = FAISS.load_local("langmem_db", OpenAIEmbeddings())

def retrieve_expert_knowledge(query):
    """Search FAISS for relevant sales knowledge."""
    docs = vectorstore.similarity_search(query, k=2)  # Retrieve top 2 most relevant entries
    return [doc.page_content for doc in docs] if docs else []


# Define conversation flow
graph = StateGraph(ConversationState)

graph.add_node("start", start_conversation)
graph.add_node("extract", extract_knowledge)
graph.add_node("clarify", request_clarification)
graph.add_node("confirm", confirm_and_store)
graph.add_node("store", store_in_langmem)

# Define conversation edges
graph.add_edge("start", "extract")  
graph.add_conditional_edges("extract", 
    lambda state: "example" not in state["extracted_knowledge"], "clarify", "confirm")

graph.add_edge("clarify", "confirm")
graph.add_conditional_edges("confirm", 
    lambda state: "yes" in state["history"][-1].content.lower(), "store", "start")

graph.set_entry_point("start")

# Compile graph
conversation_flow = graph.compile()

state = {"history": []}

# Simulate conversation
while True:
    # Run a single step of the graph
    state = conversation_flow.invoke(state)
    
    # Display AMI's response
    response = state["history"][-1].content
    print("🤖 AMI:", response)

    # Get expert input (simulate human response)
    user_input = input("👤 Expert: ")
    
    # Add expert response to history
    state["history"].append(HumanMessage(content=user_input))

    # Exit condition (user types 'exit')
    if user_input.lower() == "exit":
        print("🚀 Conversation ended. Saved knowledge:", langmem_store)
        break

