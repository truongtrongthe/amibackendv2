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
    replacements = {'\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"', '\u2014': '-', '\u2013': '-', '\u00A0': ' '}
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text.strip()

# Language detection
def detect_language(text):
    vietnamese_keywords = ["cho tôi", "về", "làm thế nào", "là gì", "chào", "bạn"]
    return "vi" if any(keyword in text.lower() for keyword in vietnamese_keywords) else "en"

# Vibe detection with your latest prompt

def detect_vibe(state: State, prior_vibe="casual"):
    if not state["messages"]:
        return "casual"
    latest_message = state["messages"][-1].content
    
    response = llm.invoke(
        f"Given this message: '{latest_message}'.\n"
        f"Return only one vibe based on these definitions and examples:\n"
        f"- knowledge: Asking for or sharing factual information. Example: 'Tell me about your CRM', 'Ghosting customers sucks'.\n"
        f"- skills: Giving practical tips or advice. Example: 'Send a gentle email after a week', 'How to re-engage a ghosting customer'.\n"
        f"- lessons: Sharing personal experiences or stories. Example: 'I once lost a deal', 'Learned this the hard way'.\n"
        f"If the message is too vague (under 3 words AND lacks context), return prior_vibe: '{prior_vibe}'.\n"
        f"If a message fits multiple categories, prioritize: lessons > skills > knowledge.\n"
        f"Options: knowledge, skills, lessons"
    )
    vibe = response.content.strip().lower()
    vibe_options = ["knowledge", "skills", "lessons"]
    if vibe not in vibe_options:
        vibe = prior_vibe
    print(f"Detected vibe: {vibe}")
    return vibe
# Intent detection with fixed compliment check
def detect_intent(latest_message, prior_messages=[]):
    if prior_messages and len(prior_messages) >= 2 and prior_messages[-2].type == "ai":
        # Check if the prior AI message was a query response
        if "Here's what I dug up" in prior_messages[-2].content:
            compliments = ["cool", "nice", "sweet", "awesome", "great", "dope"]
            if latest_message.lower() in compliments:
                print(f"Detected intent: compliment")
                return "compliment"
    
    response = llm.invoke(
        f"Classify this message as 'query' (asking something) or 'statement' (sharing something):\n"
        f"'{latest_message}'\n"
        f"Examples:\n"
        f"- 'How to re-engage ghosting customer' -> query\n"
        f"- 'Send a gentle email after a week' -> statement\n"
        f"- 'Ghosting customers suck' -> query (implicit question)\n"
        f"- 'I lost a deal once' -> statement\n"
        f"- 'Cool' -> unclear (return 'unclear')\n"
        f"Return only: query, statement, or unclear"
    )
    intent = response.content.strip().lower()
    print(f"Detected intent: {intent}")
    return intent
# Main interaction node (uncommented)

# Main interaction node with safe metadata handling
def ami_node(state: State):
    latest_message = state["messages"][-1].content if state["messages"] else "Hey!"
    user_id = state["user_id"]
    user_lang = detect_language(latest_message)
    
    # Safely get prior vibe (default to state["vibe"] if no metadata)

    prior_vibe = state.get("vibe", "casual")
    for msg in reversed(state["messages"][:-1]):  # Skip latest message
        if msg.type == "human" and hasattr(msg, "metadata"):
            prior_vibe = msg.metadata.get("vibe", prior_vibe)
        break
    vibe = detect_vibe(state, prior_vibe)

    if len(state["messages"]) == 1:
        response = ("Yo, I’m AMI—ready to vibe and learn! What’s up?" if user_lang == "en" else 
                    "Chào, tôi là AMI—sẵn sàng trò chuyện và học! Bạn khỏe không?")
        return {"prompt_str": response, "user_id": user_id, "user_lang": user_lang, "vibe": vibe, "metadata": {"vibe": vibe}}

    intent = detect_intent(latest_message, state["messages"])
    if intent == "query":
        embedding = embeddings.embed_query(latest_message)
        results = PINECONE_INDEX.query(vector=embedding, top_k=3, include_metadata=True, filter={"user_id": user_id})
        matches = sorted([m for m in results.get("matches", []) if m["score"] > 0.3], key=lambda x: x["score"], reverse=True)
        seen_texts = set()
        unique_matches = [m for m in matches if not (m["metadata"]["text"] in seen_texts or seen_texts.add(m["metadata"]["text"]))]

        response = "Here’s what I dug up:\n" if user_lang == "en" else "Đây là những gì tôi tìm thấy:\n"
        if unique_matches:
            for i, match in enumerate(unique_matches[:1], 1):
                text = match["metadata"]["text"] if user_lang == "en" else llm.invoke(f"Translate to Vietnamese: '{match['metadata']['text']}'").content.strip()
                response += f"- {i}. {text} ({match['metadata']['vibe']})\n"
            response += "\nGot more ghost-busting ideas to dig into?" if user_lang == "en" else "\nBạn có thêm ý tưởng nào để giải quyết vấn đề này không?"
        else:
            response += ("Nada yet—how about a discount trick or something?" if user_lang == "en" else 
                         "Chưa có gì—thử mẹo giảm giá chẳng hạn?")
    elif intent == "compliment":
        vibe = prior_vibe  # Use prior human message vibe
        response = ("Glad you vibe with that—whatcha got next?" if user_lang == "en" else 
                    "Rất vui bạn thích—còn gì hay ho nữa không?")
    else:  # statement or unclear
        summary = llm.invoke(f"Summarize as a {vibe}, use original words if possible, max 15 words: '{latest_message}'").content.strip()
        embedding = embeddings.embed_query(latest_message)
        metadata = {"vibe": vibe, "tags": [vibe], "text": latest_message, "summary": summary, "user_id": user_id}
        PINECONE_INDEX.upsert([(
            f"msg_{user_id}_{datetime.now().isoformat()}", 
            embedding, 
            metadata
        )])
        print(f"Upserted to Pinecone: {metadata}")
        responses = {
            "knowledge": f"Whoa, brain food! Saved: '{summary}'. What else you got?",
            "skills": f"Slick move! Saved: '{summary}'. More tricks?",
            "lessons": f"Life lesson! Saved: '{summary}'. Spill more?"
        }
        response = responses.get(vibe, f"Cool stuff! Saved: '{summary}'. What’s next?") if user_lang == "en" else (
            llm.invoke(f"Translate to Vietnamese: '{responses.get(vibe, f"Cool stuff! Saved: '{summary}'. What’s next?")}'").content.strip()
        )
        if intent == "unclear":
            response += " (Wait, you asking or telling? Hit me clearer!)" if user_lang == "en" else (
                " (Khoan, bạn hỏi hay kể vậy? Nói rõ hơn nhé!)"
            )

    return {"prompt_str": normalize_to_ascii(response), "user_id": user_id, "user_lang": user_lang, "vibe": vibe, "metadata": {"vibe": vibe}}
# Build simplified graph
graph_builder = StateGraph(State)
graph_builder.add_node("ami", ami_node)
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)

checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

# Streaming function
def learning_stream(user_input, user_id, thread_id="learning_thread"):
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = checkpointer.get(config)
    #print(f"Raw checkpoint: {checkpoint}")
    
    if checkpoint and isinstance(checkpoint, dict):
        history = checkpoint.get("channel_values", {}).get("messages", [])
    else:
        history = []
    
    new_message = HumanMessage(content=user_input)
    updated_messages = history + [new_message]

    try:
        state = convo_graph.invoke(
            {"messages": updated_messages, "user_id": user_id},
            config
        )
        response = state["prompt_str"]
        for chunk in textwrap.wrap(response, width=80):
            yield f"data: {json.dumps({'message': chunk})}\n\n"
        
        ai_message = AIMessage(content=response)
        state["messages"] = updated_messages + [ai_message]
        convo_graph.update_state(config, state)
        print(f"AMI: {response}")
    except Exception as e:
        error_msg = f"Error in stream: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

# Test it out
if __name__ == "__main__":
    user_id = "test_user"
    thread_id = "test_thread"
    inputs = [
        "Hi",
        "Send a gentle email after a week of silence.",
        "How to re-engage ghosting customer",
        "Ghosting customers suck",
        "Cool"
    ]
    for user_input in inputs:
        print(f"\nUser: {user_input}")
        for chunk in learning_stream(user_input, user_id, thread_id):
            print(chunk)