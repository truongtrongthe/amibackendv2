import json
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from pinecone_datastores import pinecone_index

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", streaming=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

# Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    user_id: str

def summarize_message(content: str) -> str:
    """Generate a concise summary of the message using the LLM."""
    summary_prompt = f"""
    Summarize the following message in a concise way, capturing the main point:
    {content}
    """
    response = llm.invoke(summary_prompt)
    return response.content.strip()


# Chatbot node relying only on conversation history
def chatbot(state: State):
    latest_message = state["messages"][-1]
    user_id = state["user_id"]
    timestamp = datetime.now().isoformat()
    
    # Summarize the latest message
    summarized_content = summarize_message(latest_message.content)
    raw_embedding = embeddings.embed_query(latest_message.content)
    summary_embedding = embeddings.embed_query(summarized_content)
    vector_id = f"{user_id}_{timestamp}"
    metadata = {
        "user_id": user_id,
        "timestamp": timestamp,
        "source": "user" if isinstance(latest_message, HumanMessage) else "ai",
        "raw_message": latest_message.content,      # Full raw message
        "summarized_message": summarized_content,    # Summarized version
        "approval_status":"pending",
        "confidence_score":"0.7"
    }

    pinecone_index.upsert([(vector_id, summary_embedding, metadata)])
    convo_history = "\n".join(
        f"{m.type}: {summarize_message(m.content)}" 
        for m in state["messages"]
    )
    
    # Construct the prompt using only the conversation history
    prompt = f"""
    You are Ami, an assistant are at active learning mode.
    Chat history:
    {convo_history}
    User: {latest_message.content}
    You're listening to user and save everything user said.
    Respond based on the conversation history, keep the response short, example: "I got it! I'm excited to learn more!"
    """
    
    return {"prompt_str": prompt, "user_id": user_id}

# Build and compile the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

# Event stream function
def event_stream(user_input, user_id, thread_id="global_thread"):
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

