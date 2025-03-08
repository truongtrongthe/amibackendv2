import json
import re
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone_datastores import pinecone_index  # Your custom import
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Initialize OpenAI
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

# Use your pinecone_index
PINECONE_INDEX = pinecone_index

# Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    user_id: str
    user_lang: str
    vibe: str

# Language detection
def detect_language(text):
    detected_language = "en"
    if any(word in text.lower() for word in ["chào", "bạn", "anh", "chị", "em"]):
        detected_language = "vi"
    return detected_language

# Vibe detection with GPT-4o
def detect_vibe(state: State):
    if not state["messages"]:
        return "casual"
    history = "\n".join(f"{m.type}: {m.content}" for m in state["messages"])
    response = llm.invoke(
        f"Return only the vibe (casual, knowledge, skills, lessons) for this chat:\n{history}"
    )
    vibe = response.content.strip().lower()
    vibe_options = ["casual", "knowledge", "skills", "lessons"]
    for option in vibe_options:
        if option in vibe:
            vibe = option
            break
    else:
        vibe = "casual"
    print(f"Detected vibe: {vibe}")
    return vibe

# Chat node: Guess vibe and propose it
def chat_node(state: State):
    latest_message = state["messages"][-1].content if state["messages"] else "Hello!"
    user_id = state["user_id"]
    user_lang = detect_language(latest_message)
    
    if "vibe" not in state or not state["vibe"]:
        state["vibe"] = detect_vibe(state)
    vibe = state["vibe"]
    
    # Use system message to enforce role
    system_msg = SystemMessage(content="You’re AMI, a young, energetic sales apprentice for [Company X]. Always start with the exact vibe check phrase.")
    user_msg = f"In {user_lang}, say exactly: 'Hey, I’m AMI! I think this is a {vibe} vibe—am I on track?' Then add a short vibe-specific follow-up (e.g., casual: 'What’s up?', knowledge: 'CRM boosts retention by 20%')."
    state["messages"] = [system_msg] + state["messages"]  # Prepend system message
    prompt = user_msg
    return {"prompt_str": prompt, "user_id": user_id, "user_lang": user_lang}

# Confirm node: Handle confirmation and save
def confirm_node(state: State):
    if len(state["messages"]) < 2:
        return {"prompt_str": "Let’s chat first—what’s up?"}
    
    latest_response = state["messages"][-1].content.lower()
    vibe = state.get("vibe", "casual")  # Fallback to casual
    user_lang = state["user_lang"]
    
    try:
        confirmation_check = llm.invoke(
            f"User said: '{latest_response}'. Is this a confirmation (yes/no) or correction? If correction, return 'correction: <new_vibe>'."
        ).content.lower()
        print(f"Confirmation check: {confirmation_check}")
    except Exception as e:
        print(f"Error in confirmation check: {e}")
        confirmation_check = "no"  # Default to no if LLM fails
    
    if "yes" in confirmation_check:
        message_to_save = state["messages"][-2].content
        embedding = embeddings.embed_query(message_to_save)
        PINECONE_INDEX.upsert([(
            f"msg_{state['user_id']}_{datetime.now().isoformat()}", 
            embedding, 
            {"vibe": vibe, "text": message_to_save, "user_id": state["user_id"]}
        )])
        prompt = f"Saved as {vibe}! What’s next? (Respond in {user_lang})"
        return {"prompt_str": prompt, "vibe": vibe}
    
    elif "correction:" in confirmation_check:
        new_vibe = confirmation_check.split("correction:")[-1].strip()
        prompt = f"Got it, switching to {new_vibe}—right now? (Respond in {user_lang})"
        return {"prompt_str": prompt, "vibe": new_vibe}
    
    else:
        prompt = f"Not sure—did I get {vibe} right or should I guess again? (Respond in {user_lang})"
        return {"prompt_str": prompt}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chat_node)
graph_builder.add_node("confirm", confirm_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "confirm")
graph_builder.add_edge("confirm", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

# Streaming function
def learning_stream(user_input, user_id, thread_id="learning_thread"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    history = checkpoint["channel_values"].get("messages", []) if checkpoint else []
    
    new_message = HumanMessage(content=user_input)
    updated_messages = history + [new_message]
    
    try:
        state = convo_graph.invoke(
            {"messages": updated_messages, "user_id": user_id},
            {"configurable": {"thread_id": thread_id}}
        )
        prompt = state["prompt_str"]
        
        full_response = ""
        for chunk in llm.stream(prompt):
            if chunk.content.strip():
                full_response += chunk.content
                yield f"data: {json.dumps({'message': chunk.content})}\n\n"
        
        ai_message = AIMessage(content=full_response)
        state["messages"] = updated_messages + [ai_message]
        convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state)
        print(f"AI response saved: {full_response[:50]}...")
        
    except Exception as e:
        error_msg = f"Error in stream: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

# Test
if __name__ == "__main__":
    user_id = "test_user"
    thread_id = "test_thread"
    
    print("Testing 'Hi!'")
    for chunk in learning_stream("Hi!", user_id, thread_id):
        print(chunk)
    
    print("\nTesting 'Yep'")
    for chunk in learning_stream("Yep", user_id, thread_id):
        print(chunk)
    
    history = convo_graph.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]
    for msg in history:
        print(f"{msg.type}: {msg.content}")