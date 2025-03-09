import json
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
    vietnamese_keywords = ["cho tôi", "về", "làm thế nào", "là gì", "chào", "bạn"]
    if any(keyword in text.lower() for keyword in vietnamese_keywords):
        return "vi"
    return "en"

# Vibe detection (for teach flow)
def detect_vibe(state: State):
    if not state["messages"]:
        return "casual"
    latest_message = state["messages"][-1].content
    prior_vibe = state.get("vibe", "casual")
    
    response = llm.invoke(
        f"Given this message: '{latest_message}'.\n"
        f"Return only one vibe based on these definitions and examples:\n"
        f"- knowledge: seeking or sharing info (e.g., 'Tell me about your CRM', 'Chat about sales')\n"
        f"- skills: practical tips or advice (e.g., 'Try this sales trick', 'Handle rejection', 'Be patient in convo')\n"
        f"- lessons: personal experiences or stories (e.g., 'I once lost a deal', 'Ask about past experience')\n"
        f"If vague (e.g., 'Cool', 'OK'), use prior_vibe: '{prior_vibe}'.\n"
        f"Options: knowledge, skills, lessons"
    )
    vibe = response.content.strip().lower()
    vibe_options = ["knowledge", "skills", "lessons"]
    if vibe not in vibe_options:
        vibe = prior_vibe
    print(f"Detected vibe: {vibe}")
    return vibe

# Recall from Pinecone (tweaked with debug)
def recall_from_pinecone(query, user_id, user_lang):
    original_query = query
    triggers_en = ["tell me about", "how to", "what is"]
    triggers_vi = ["cho tôi biết về", "làm thế nào để", "là gì"]
    trigger_map = {
        "cho tôi biết về": "tell me about",
        "làm thế nào để": "how to",
        "là gì": "what is"
    }
    translation_map = {
        "sales": "sales",
        "trò chuyện với khách hàng": "chat with customers"
    }
    input_lower = query.lower()
    print(f"Input: '{input_lower}'")
    query = input_lower
    for trigger in triggers_en + triggers_vi:
        print(f"Checking trigger: '{trigger}'")
        if trigger in input_lower:
            print(f"Trigger matched: '{trigger}'")
            query = input_lower.replace(trigger, "").strip()  # Strip trigger
            english_trigger = trigger if trigger in triggers_en else trigger_map[trigger]
            english_query = english_trigger + " " + translation_map.get(query, query)
            break
    else:
        print("No trigger matched")
        english_query = translation_map.get(query, query)
    print(f"Pre-padded query: '{query}'")
    print(f"English query: '{english_query}'")
    query_embedding = embeddings.embed_query(english_query)  # Embed English
    print(f"Embedding snippet: {query_embedding[:5]}")
    results = PINECONE_INDEX.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        filter={"user_id": user_id}
    )
    print(f"Raw Pinecone results: {results}")
    match_dict = {}
    for m in results.get("matches", []):
        text = m["metadata"]["text"]
        if text not in match_dict or m["score"] > match_dict[text]["score"]:
            match_dict[text] = m
    matches = [m for m in match_dict.values() if m["score"] > 0.45]
    print(f"Match scores: {[m['score'] for m in matches]}")
    if not matches:
        response = f"Vault’s empty on '{original_query}'—teach me something about it!"
        if user_lang == "vi":
            response = llm.invoke(f"Translate to Vietnamese: '{response}'").content.strip()
        return response
    response = "Here’s what I’ve got from the vault:" if user_lang == "en" else "Đây là những gì tôi có từ kho:"
    for i, match in enumerate(matches, 1):
        text = match["metadata"]["text"]
        vibe = match["metadata"]["vibe"]
        if user_lang == "vi":
            text = llm.invoke(f"Translate to Vietnamese: '{text}'").content.strip()
            vibe = {"knowledge": "kiến thức", "skills": "kỹ năng", "lessons": "bài học"}.get(vibe, vibe)
        response += f" {i}. '{text}' ({vibe})"
    return response

def chat_node(state: State):
    latest_message = state["messages"][-1].content if state["messages"] else "Hello!"
    user_id = state["user_id"]
    user_lang = detect_language(latest_message)
    
    if len(state["messages"]) == 1:
        response = "Hey there! I’m AMI, pumped to chat and learn—what’s on your mind today?"
        if user_lang == "vi":
            response = "Chào bạn! Tôi là AMI, hào hứng trò chuyện và học—bạn đang nghĩ gì vậy?"
    else:
        response = ""  # Pass to confirm for teach or recall
    
    response = normalize_to_ascii(response)
    return {"prompt_str": response, "user_id": user_id, "user_lang": user_lang}

# Confirm node: Teach or recall
def confirm_node(state: State):
    if len(state["messages"]) < 2:
        return {"prompt_str": state["prompt_str"]}
    
    latest_message = state["messages"][-1].content
    user_id = state["user_id"]
    user_lang = detect_language(latest_message)
    print(f"User ID: {user_id}")
    
    # Check for recall triggers (English or Vietnamese)
    triggers_en = ["tell me about", "how to", "what is"]
    triggers_vi = ["cho tôi biết về", "làm thế nào để", "là gì"]
    input_lower = latest_message.lower()
    if any(trigger in input_lower for trigger in triggers_en + triggers_vi):
        print(f"Full query: '{latest_message}'")
        response = recall_from_pinecone(latest_message, user_id, user_lang)  # Pass full query
    else:
        # Teach flow (unchanged)
        vibe = detect_vibe(state)
        state["vibe"] = vibe
        summary = llm.invoke(f"Summarize in 5-10 words: '{latest_message}'").content.strip()
        acknowledgements = {
            "knowledge": f"Got it—that’s some dope knowledge! Summary: '{summary}' Locked it in the vault.",
            "skills": f"Got it—that’s a slick skills tip! Summary: '{summary}' Locked it in the vault.",
            "lessons": f"Got it—that’s a wild lesson! Summary: '{summary}' Locked it in the vault."
        }
        response = acknowledgements.get(vibe, f"Got it—that’s cool! Summary: '{summary}' Locked it in the vault.")
        
        if user_lang == "vi":
            response = llm.invoke(f"Translate to Vietnamese: '{response}'").content.strip()
        
        embedding = embeddings.embed_query(latest_message)
        PINECONE_INDEX.upsert([(
            f"msg_{user_id}_{datetime.now().isoformat()}", 
            embedding, 
            {"vibe": vibe, "tags": [vibe], "text": latest_message, "summary": summary, "user_id": user_id}
        )])
        print(f"Saved to Pinecone: vibe={vibe}, text='{latest_message}', summary='{summary}'")
    
    response = normalize_to_ascii(response)
    return {"prompt_str": response, "vibe": state.get("vibe", "casual")}
# Build the graph
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
        #"Let's have a chat about sales",
        #"When you chat with customer, keep it gentle",
        #"I once lost a deal by rushing",
        #"Tell me about sales",
        "Cho tôi biết về sales",
        #"How to chat with customers",
        "Làm thế nào để trò chuyện với khách hàng",
        #"What is upselling?",
        #"Upselling là gì?"
    ]
    for i, user_input in enumerate(inputs):
        print(f"\nTesting '{user_input}'")
        for chunk in learning_stream(user_input, user_id, thread_id):
            print(chunk)
    history = convo_graph.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]
    for msg in history:
        print(f"{msg.type}: {msg.content}")