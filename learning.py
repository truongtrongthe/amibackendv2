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
import textwrap

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
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"', '\u2014': '-', '\u2013': '-',
        '\u00A0': ' '  # Non-breaking space
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text.strip()

# Language detection
def detect_language(text):
    detected_language = "en"
    if any(word in text.lower() for word in ["chào", "bạn", "anh", "chị", "em"]):
        detected_language = "vi"
    return detected_language

# ... [Keep imports, detect_vibe unchanged unless noted] ...

# ... [Keep imports unchanged] ...

# Refined vibe detection (unchanged from last fix)
def detect_vibe(state: State):
    if not state["messages"]:
        return "casual"
    window_size = 5
    recent_messages = state["messages"][-window_size:]
    history = "\n".join(f"{m.type}: {m.content}" for m in recent_messages)
    latest_message = state["messages"][-1].content if state["messages"] else ""
    prior_vibe = state.get("vibe", "casual")
    
    confirmation_words = ["yes", "yeah", "yep", "it is", "yes it is", "it is a confirm", "it was a confirm", "sure"]
    if any(word in latest_message.lower() for word in confirmation_words) and not any(word in latest_message.lower() for word in ["but", "no", "nah"]):
        return prior_vibe
    
    response = llm.invoke(
        f"Given this chat history:\n{history}\nFocus especially on the latest message: '{latest_message}'.\n"
        f"Return only one vibe based on these definitions and examples:\n"
        f"- casual: informal greetings or chit-chat (e.g., 'Hi!', 'What's up?', 'Cool, bro')\n"
        f"- knowledge: seeking or sharing info (e.g., 'Tell me about your CRM', 'Chat about sales')\n"
        f"- skills: practical tips or advice (e.g., 'Try this sales trick', 'Handle rejection', 'Be patient in convo')\n"
        f"- lessons: personal experiences or stories (e.g., 'I once lost a deal', 'Ask about past experience')\n"
        f"If vague (e.g., 'Cool', 'OK'), use prior vibe: '{prior_vibe}'.\n"
        f"Options: casual, knowledge, skills, lessons"
    )
    vibe = response.content.strip().lower()
    vibe_options = ["casual", "knowledge", "skills", "lessons"]
    if vibe not in vibe_options:
        vibe = prior_vibe if state.get("vibe") else "casual"
    print(f"Detected vibe: {vibe}")
    return vibe

# Chat node: Propose vibe with eager learning twist
def chat_node(state: State):
    latest_message = state["messages"][-1].content if state["messages"] else "Hello!"
    user_id = state["user_id"]
    user_lang = detect_language(latest_message)
    
    vibe = detect_vibe(state)
    state["vibe"] = vibe
    
    # First message: Greeting
    if len(state["messages"]) == 1:
        response = "Hey there! I’m AMI, pumped to chat and learn—what’s on your mind today?"
    else:
        # After "no": Re-confirm
        if len(state["messages"]) > 2 and state["messages"][-2].type == "ai" and "confirmation_check='no'" in str(state.get("messages", [])):
            vibe_check = f"Alright, I’m locked on {vibe} now—am I nailing it?"
        # New topic: Eager to learn
        else:
            vibe_check = f"Ooh, I’m picking up a {vibe} vibe here—am I on the right track?"
        follow_up = {
            "casual": "What’s the word on the street?",
            "knowledge": "What’s the scoop—any hot info to share?",
            "skills": "Got a slick trick up your sleeve?",
            "lessons": "What’s a wild story from the trenches?"
        }.get(vibe, "What’s the word on the street?")
        response = f"{vibe_check} Teach me something cool about it! {follow_up}"
    
    if user_lang != "en":
        response = llm.invoke(f"Translate to {user_lang}: '{response}'").content.strip()
    
    response = normalize_to_ascii(response)
    return {"prompt_str": response, "user_id": user_id, "user_lang": user_lang, "vibe": vibe}

# Confirm node: Handle confirmation with new responses
def confirm_node(state: State):
    if len(state["messages"]) < 2:
        return {"prompt_str": normalize_to_ascii("Hey there! I’m AMI, pumped to chat and learn—what’s on your mind today?")}
    
    latest_response = state["messages"][-1].content
    prior_message = state["messages"][-2].content
    vibe = state.get("vibe", "casual")
    user_lang = state["user_lang"]
    
    confirm_check_prompt = """
    Return only: 'yes', 'no', or 'correction: <vibe>' where vibe is one of: casual, knowledge, skills, lessons—do not include explanations or reasoning.
    Chat history:
    - AI: '{prior_message}'
    - User: '{latest_response}'
    Prior proposed vibe was: '{vibe}'.
    Determine the user's intent based on their response:
    - 'yes' if prior message explicitly proposes a vibe (e.g., contains 'am I on track?' or 'am I nailing it?') AND response clearly affirms it (e.g., 'yes,' 'yeah') without adding new info.
    - 'no' if prior message isn’t a proposal OR response introduces a new topic or doesn’t clearly affirm the prior vibe.
    - 'correction: <vibe>' only if user explicitly rejects the prior vibe with 'no' or 'nah' AND names a vibe (e.g., 'No, it’s <vibe>', 'Nah, that’s <vibe>').
    Use these vibe definitions:
    - casual: informal greetings or chit-chat (e.g., 'Hi!', 'What's up?')
    - knowledge: seeking or sharing info (e.g., 'Tell me about your CRM', 'Chat about sales')
    - skills: practical tips or advice (e.g., 'Handle rejection', 'Be patient in convo')
    - lessons: personal experiences or stories (e.g., 'I once lost a deal', 'Ask about past experience')
    Rules:
    - If response negates (e.g., 'no', 'nah') but doesn’t explicitly name a vibe, return 'no'—treat as new topic, not correction.
    Examples:
    - AI: 'casual vibe—am I on track?' User: 'Yeah' → 'yes'
    - AI: 'casual vibe—am I on track?' User: 'Let’s chat about sales' → 'no'
    - AI: 'knowledge vibe—am I on track?' User: 'We need to be very patient...' → 'no'
    - AI: 'skills vibe—am I on track?' User: 'No, it’s lessons' → 'correction: lessons'
    - AI: 'skills vibe—am I on track?' User: 'Cool' → 'no'
    - AI: 'lessons vibe—am I on track?' User: 'It is a confirm' → 'yes'
    - AI: 'Saved as skills! What's next?' User: 'Let’s move to handling tough...' → 'no'
    - AI: 'skills vibe—am I on track?' User: 'You could ask customer...' → 'no'
    - AI: 'casual vibe—am I on track?' User: 'Nah, let’s talk pricing' → 'no'
    - AI: 'knowledge vibe—am I on track?' User: 'Nah, tell me a story' → 'no'
    - AI: 'skills vibe—am I on track?' User: 'Sure' → 'yes'
    - AI: 'skills vibe—am I on track?' User: 'Nope, tell me a story' → 'correction: lessons'
    - AI: 'skills vibe—am I on track?' User: 'Yeah, but let’s switch' → 'no'
    - AI: 'knowledge vibe—am I on track?' User: 'OK' → 'no'
    - AI: 'skills vibe—am I on track?' User: 'No way, it’s casual!' → 'correction: casual'
    """
    try:
        formatted_prompt = confirm_check_prompt.format(
            prior_message=prior_message,
            latest_response=latest_response,
            vibe=vibe
        )
        confirmation_check = llm.invoke(formatted_prompt).content.lower()
        print(f"Confirmation check: {confirmation_check}")
    except Exception as e:
        print(f"Error in confirmation check: {e}")
        confirmation_check = "no"
    
    # Save prior user message on "yes" if confirming a proposal
    prior_message_clean = normalize_to_ascii(prior_message).lower()
    is_proposal = "am i on the right track?" in prior_message_clean or "am i nailing it?" in prior_message_clean
    print(f"Checking save: confirmation_check='{confirmation_check}', messages_len={len(state['messages'])}, is_proposal={is_proposal}")
    if confirmation_check == "yes" and len(state["messages"]) >= 3 and is_proposal:
        print(f"Save block triggered for vibe: {vibe}")
        prior_user_message = state["messages"][-3].content  # Grab user input before AI proposal
        print(f"Saving user message: '{prior_user_message}'")
        embedding = embeddings.embed_query(prior_user_message)
        PINECONE_INDEX.upsert([(
            f"msg_{state['user_id']}_{datetime.now().isoformat()}", 
            embedding, 
            {"vibe": vibe, "text": prior_user_message, "user_id": state["user_id"]}
        )])
        response = f"Boom, locked in as {vibe}! What’s the next gem you’ve got for me?"
        return {"prompt_str": response, "vibe": vibe}
    
    # Handle corrections
    if confirmation_check.startswith("correction:"):
        new_vibe = confirmation_check.split("correction:")[-1].strip()
        detected_vibe = detect_vibe(state)
        if new_vibe == detected_vibe:
            print(f"Correction matches detected vibe: {detected_vibe}, keeping original response")
            return {"prompt_str": state["prompt_str"], "vibe": vibe}
        if new_vibe in ["casual", "knowledge", "skills", "lessons"]:
            state["vibe"] = new_vibe
            response = f"Whoa, I see it now—this feels like {new_vibe}! Am I catching your drift?"
            return {"prompt_str": response, "vibe": new_vibe}
        else:
            response = "Oops, that’s not a vibe I know—try casual, knowledge, skills, or lessons!"
            return {"prompt_str": response, "vibe": vibe}
    
    # Default: keep chat_node response
    print(f"No save or correction, using chat_node response: {state['prompt_str']}")
    return {"prompt_str": state["prompt_str"], "vibe": vibe}

# ... [Keep graph building, learning_stream unchanged] ...

# ... [Keep graph building, learning_stream unchanged] ...

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chat_node)
graph_builder.add_node("confirm", confirm_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", lambda state: "confirm" if len(state["messages"]) > 1 else END)
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
        response = state["prompt_str"]
        
        for chunk in textwrap.wrap(response, width=80):
            yield f"data: {json.dumps({'message': chunk})}\n\n"
        
        ai_message = AIMessage(content=response)
        state["messages"] = updated_messages + [ai_message]
        convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state)
        print(f"AI response saved: {response[:50]}...")
        
    except Exception as e:
        error_msg = f"Error in stream: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

# Test sequence
if __name__ == "__main__":
    user_id = "test_user"
    thread_id = "test_thread"
    inputs = [
        "Hi",
        "Let's have a chat about sales",
        "We need to be very patient in customer conversation",
        "Yeah",
        "Let's move to handling tough situation like rejection, are you willing to rock?",
        "You could ask customer about the past experience before coming back",
        "It is a confirm",
        "Hey, how’s it going?",
        "Nah, let’s talk pricing",
        "Cool"
    ]
    for i, user_input in enumerate(inputs):
        print(f"\nTesting '{user_input}'")
        for chunk in learning_stream(user_input, user_id, thread_id):
            print(chunk)
    history = convo_graph.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]
    for msg in history:
        print(f"{msg.type}: {msg.content}")