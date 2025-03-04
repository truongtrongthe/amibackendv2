import json
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone_datastores import pinecone_index

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", streaming=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)


# Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    user_id: str
# Chatbot node relying only on conversation history
def copilot(state: State):
    latest_message = state["messages"][-1]
    user_id = state["user_id"]
    
    current_convo_history = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"]
    )

    pinecone_history = get_pinecone_data(user_id, query=latest_message.content, limit=5)  # Use latest message as query
    pinecone_context = "\n".join(
        f"[{entry['timestamp']}] {entry['summarized_message']} (Raw: {entry['raw_message']})"
        for entry in pinecone_history
    ) or "No prior context available."

    print("context found:",pinecone_context)
    # Construct the prompt using only the conversation history
    prompt = f"""
    You are Ami, a Sale assistant who can recall everything said.
    Current chat history:
    {current_convo_history}
    Prior conversation context (from Pinecone):
    {pinecone_context}
    User: {latest_message.content}
    Respond empathetically based on the conversation context and sales expertise
    """
    return {"prompt_str": prompt, "user_id": user_id}

# Build and compile the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", copilot)
graph_builder.add_edge(START, "chatbot")
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def pilot_stream(user_input, user_id, thread_id="sale_thread"):
    print(f"Sending user input to AI model: {user_input}")
    
    # Retrieve the existing conversation history from MemorySaver
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    history = checkpoint["channel_values"].get("messages", []) if checkpoint else []
    
    # Append the new user input to the history
    new_message = HumanMessage(content=user_input)
    updated_messages = history + [new_message]
    
    try:
        # Invoke the graph with the full message history
        state = convo_graph.invoke(
            {"messages": updated_messages, "user_id": user_id},
            {"configurable": {"thread_id": thread_id}}
        )
        prompt = state["prompt_str"]
        print(f"Prompt to AI model: {prompt}")
        
        # Stream the LLM response
        full_response = ""
        for chunk in llm.stream(prompt):
            if chunk.content.strip():
                full_response += chunk.content
                yield f"data: {json.dumps({'message': chunk.content})}\n\n"
        
        # Save the AI response back to the graph
        ai_message = AIMessage(content=full_response)
        convo_graph.invoke(
            {"messages": [ai_message], "user_id": user_id},
            {"configurable": {"thread_id": thread_id}}
        )
        print(f"AI response saved to graph: {full_response[:50]}...")
        
    except Exception as e:
        error_msg = f"Error in event stream: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"


# Optional: Utility function to inspect history (for debugging)
def get_conversation_history(thread_id="global_thread"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    if checkpoint:
        state = checkpoint["channel_values"]
        return state.get("messages", [])
    return []


def get_pinecone_data(user_id: str, query: str = None, limit: int = 10):
    """Retrieve raw and summarized messages from Pinecone."""
    if query:
        query_embedding = embeddings.embed_query(query)
        results = pinecone_index.query(vector=query_embedding, top_k=limit, include_metadata=True, filter={"user_id": user_id})
    else:
        results = pinecone_index.query(vector=[0]*1536, top_k=limit, include_metadata=True, filter={"user_id": user_id})
    
    return [
        {
            "raw_message": match["metadata"]["raw_message"],
            "summarized_message": match["metadata"]["summarized_message"],
            "timestamp": match["metadata"]["timestamp"]
        }
        for match in results["matches"]
    ]