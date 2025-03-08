import json
import re
import unicodedata
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone_datastores import pinecone_index
from langchain_core.messages import AIMessage, HumanMessage

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

# Normalize Unicode to ASCII
def normalize_to_ascii(text):
    text = unicodedata.normalize('NFKD', text)
    replacements = {
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"', '\u2014': '-', '\u2013': '-'
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text

# Language detection
def detect_language(text):
    detected_language = "en"
    if any(word in text.lower() for word in ["chào", "bạn", "anh", "chị", "em"]):
        detected_language = "vi"
    return detected_language

# Vibe detection with clearer guidance
def detect_vibe(state: State):
    if not state["messages"]:
        return "casual"
    window_size = 5
    recent_messages = state["messages"][-window_size:]
    history = "\n".join(f"{m.type}: {m.content}" for m in recent_messages)
    latest_message = state["messages"][-1].content if state["messages"] else ""
    response = llm.invoke(
        f"Given this chat history:\n{history}\nFocus especially on the latest message: '{latest_message}'.\n"
        f"Return only one vibe based on these definitions:\n"
        f"- casual: informal greetings or chit-chat (e.g., 'Hi!', 'What's up?')\n"
        f"- knowledge: seeking or sharing info (e.g., 'Tell me about your CRM')\n"
        f"- skills: practical tips or advice (e.g., 'Try this sales trick')\n"
        f"- lessons: personal experiences or stories (e.g., 'I once lost a deal')\n"
        f"Options: casual, knowledge, skills, lessons"
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
    
    state["vibe"] = detect_vibe(state)
    vibe = state["vibe"]
    
    vibe_check = f"Hey, I'm AMI! I think this is a {vibe} vibe—am I on track?"
    follow_up = {
        "casual": "What's up?",
        "knowledge": "Our CRM boosts retention by 20%—need more?",
        "skills": "Try 'What’s stopping you?' to close deals.",
        "lessons": "I once lost a deal by overselling—learned to listen."
    }.get(vibe, "What's up?")
    
    response = f"{vibe_check} {follow_up}"
    if user_lang != "en":
        response = llm.invoke(f"Translate to {user_lang}: '{response}'").content.strip()
    
    response = normalize_to_ascii(response)
    return {"prompt_str": response, "user_id": user_id, "user_lang": user_lang}

# Confirm node: Handle confirmation and save
def confirm_node(state: State):
    if len(state["messages"]) < 2:
        return {"prompt_str": normalize_to_ascii("Let's chat first—what's up?")}
    
    latest_response = state["messages"][-1].content.lower()
    vibe = state.get("vibe", "casual")
    user_lang = state["user_lang"]
    
    try:
        confirmation_check = llm.invoke(
            f"User said: '{latest_response}'. Return only: 'yes', 'no', or 'correction: <vibe>' where vibe is one of: casual, knowledge, skills, lessons."
        ).content.lower()
        print(f"Confirmation check: {confirmation_check}")
    except Exception as e:
        print(f"Error in confirmation check: {e}")
        confirmation_check = "no"
    
    if "yes" in confirmation_check:
        message_to_save = state["messages"][-2].content
        embedding = embeddings.embed_query(message_to_save)
        PINECONE_INDEX.upsert([(
            f"msg_{state['user_id']}_{datetime.now().isoformat()}", 
            embedding, 
            {"vibe": vibe, "text": message_to_save, "user_id": state["user_id"]}
        )])
        response = f"Saved as {vibe}! What's next?"
    elif "correction:" in confirmation_check:
        new_vibe = confirmation_check.split("correction:")[-1].strip()
        if new_vibe in ["casual", "knowledge", "skills", "lessons"]:
            state["vibe"] = new_vibe
            response = f"Got it, switching to {new_vibe}—right now?"
        else:
            response = "Oops, that’s not a vibe I know—try casual, knowledge, skills, or lessons!"
    else:
        return {"prompt_str": state["prompt_str"]}  # Keep chat_node’s response
    
    if user_lang != "en":
        response = llm.invoke(f"Translate to {user_lang}: '{response}'").content.strip()
    response = normalize_to_ascii(response)
    return {"prompt_str": response}

# Build the graph with conditional edges
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chat_node)
graph_builder.add_node("confirm", confirm_node)
graph_builder.add_edge(START, "chatbot")

def route_to_confirm(state: State):
    return "confirm" if len(state["messages"]) > 1 else END

graph_builder.add_conditional_edges("chatbot", route_to_confirm)
graph_builder.add_edge("confirm", END)

checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

# Streaming function with word-by-word chunks
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
        response = state["prompt_str"]
        
        for word in response.split():
            yield f"data: {json.dumps({'message': word})}\n\n"
        
        ai_message = AIMessage(content=response)
        state["messages"] = updated_messages + [ai_message]
        convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state)
        print(f"AI response saved: {response[:50]}...")
        
    except Exception as e:
        error_msg = f"Error in stream: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

# Test with extended conversation
if __name__ == "__main__":
    user_id = "test_user"
    thread_id = "test_thread"
    
    print("Testing 'Hi!'")
    for chunk in learning_stream("Hi!", user_id, thread_id):
        print(chunk)
    
    print("\nTesting 'Yep'")
    for chunk in learning_stream("Yep", user_id, thread_id):
        print(chunk)
    
    print("\nTesting 'Tell me about your CRM'")
    for chunk in learning_stream("Tell me about your CRM", user_id, thread_id):
        print(chunk)
    
    print("\nTesting 'Hey Ami, I have an experience...police! It works!'")
    for chunk in learning_stream("Hey Ami, I have an experience wanna tell you: just threaten customer who deny to pay outstanding by calling police! It works!", user_id, thread_id):
        print(chunk)
    
    history = convo_graph.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]
    for msg in history:
        print(f"{msg.type}: {msg.content}")