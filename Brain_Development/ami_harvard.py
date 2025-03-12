# ami_core_v1.2
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 12, 2025
# Purpose: Ami's Training Path—proactive start, dynamic LLM, Pinecone-ready

import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from pinecone_datastores import pinecone_index

# Setup
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
PINECONE_INDEX = pinecone_index

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    brain: list

# Preset Brain (SPIN/Challenger)
class PresetBrain:
    def spin_question(self, type, content):
        type = type.lower()
        if type == "situation":
            return f"Tell me about {content} in your current setup."
        elif type == "problem":
            return f"What challenges do you face with {content}?"
        elif type == "implication":
            return f"How does {content} impact your goals?"
        elif type == "need-payoff":
            return f"What if {content} worked better for you?"
        return "Unknown SPIN type"

    def challenger_teach(self, insight):
        from random import choice
        templates = [
            f"Here’s a twist they might miss: {insight}",
            f"Reframe it like this: {insight}",
            f"Shake ‘em up with this: {insight}"
        ]
        return choice(templates)

# Detect Intent
def detect_intent(msg):
    intent_prompt = f"""
    Analyze this message and return ONLY the intent as a quoted phrase:
    - Message: '{msg}'
    - "casual": Chit-chat, no sales angle.
    - "teaching": Sales-related—like a technique, fact, or result.
    """
    return llm.invoke(intent_prompt).content.strip()

# Extract Knowledge
def extract_knowledge(msg):
    extract_prompt = f"""
    Extract EXACT TEXT from this message for sales-related info:
    - Message: '{msg}'
    Return a JSON object with:
    {{
        "skills": ["exact text of suggested techniques or questions, e.g., 'Ask \\'How’s your team doing?\\'' from 'Let’s ask \\'How’s your team doing?\\''"],
        "knowledge": ["exact text of facts, e.g., 'The CRM syncs in real-time'"],
        "lessons": ["exact text of outcomes, e.g., 'Last week we closed 10 deals'"]
    }}
    - COPY THE EXACT TEXT—NO rephrasing, NO summarizing, NO placeholders like 'Sales techniques or questions'.
    - Skills include suggested sales moves or questions.
    - Empty lists if no match.
    - Plain JSON, no markers.
    """
    raw_response = llm.invoke(extract_prompt).content.strip()
    cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"skills": [], "knowledge": [], "lessons": []}

# Ami Node
def ami_node(state: State, preset: PresetBrain):
    msg = state["messages"][-1].content if state["messages"] else ""
    intent = detect_intent(msg)
    brain = state.get("brain", [])
    is_first = not state.get("messages", [])  # Only true on first turn

    pickup = "Hey, I’m Ami—sharp but green. What’s your one sales move I need to steal?" if not brain else "Hey, I’m Ami—got some tricks. What’s your next-level play?"

    if intent == '"teaching"':
        extracted = extract_knowledge(msg)
        for skill in extracted["skills"]:
            brain.append({"type": "skill", "content": skill})
            vector = embeddings.embed_query(skill)
            PINECONE_INDEX.upsert([("skill_" + str(len(brain)), vector)])
        for knowledge in extracted["knowledge"]:
            brain.append({"type": "knowledge", "content": knowledge})
            vector = embeddings.embed_query(knowledge)
            PINECONE_INDEX.upsert([("knowledge_" + str(len(brain)), vector)])
        for lesson in extracted["lessons"]:
            brain.append({"type": "lesson", "content": lesson})
            vector = embeddings.embed_query(lesson)
            PINECONE_INDEX.upsert([("lesson_" + str(len(brain)), vector)])
        
        reply_parts = []
        for skill in extracted["skills"]:
            spin_q = preset.spin_question("situation", skill)
            reply_parts.append(f"Whoa, {skill}? Slick move! {spin_q}")
        for knowledge in extracted["knowledge"]:
            reply_parts.append(f"Got it, {knowledge}—key stuff!")
        for lesson in extracted["lessons"]:
            challenger = preset.challenger_teach(lesson)
            reply_parts.append(f"Damn, {lesson}? Clutch! {challenger}")
        reply = " ".join(reply_parts) or "Spill more—I’m hooked!"
        reply += " How’d you figure that out?"
    else:
        casual_prompt = f"""
        Respond to '{msg}' like a sharp, charming Harvard girl—confident, curious, punchy.
        Short, energetic, fun follow-up question. No emojis.
        """
        reply = llm.invoke(casual_prompt).content

    prompt_str = f"{pickup} {reply}" if is_first else reply
    return {"prompt_str": prompt_str, "brain": brain, "messages": state["messages"]}

# Graph Setup
preset = PresetBrain()
graph_builder = StateGraph(State)
graph_builder.add_node("ami", lambda state: ami_node(state, preset))
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

# Streaming Output
def convo_stream(user_input=None, thread_id="learning_thread"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}}) or {}
    channel_values = checkpoint.get("channel_values", {})
    history = channel_values.get("messages", [])
    
    if not history and user_input is None:  # True first turn
        state = {"messages": [], "prompt_str": "", "brain": channel_values.get("brain", [])}
    else:  # Subsequent turns
        new_message = HumanMessage(content=user_input or "")
        state = {"messages": history + [new_message], "prompt_str": "", "brain": channel_values.get("brain", [])}
    
    state = convo_graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    # Debug prints in the stream
    msg = state["messages"][-1].content if state["messages"] else ""
    intent = detect_intent(msg)
    yield f"Debug: Intent detected = {intent}\n"
    if intent == '"teaching"':
        extracted = extract_knowledge(msg)
        yield f"Debug: Raw extract response = {json.dumps(extracted)}\n"
    
    response_lines = state["prompt_str"].split('\n')
    for line in response_lines:
        if line.strip():
            yield f"data: {json.dumps({'message': line.strip()})}\n\n"
    
    convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state, as_node="ami")

# Test Run
if __name__ == "__main__":
    print("\nAmi starts:")
    for chunk in convo_stream():  # Proactive first move
        print(chunk)
    test_inputs = [
        "Hey, good to meet you!",
        "Let’s ask 'How’s your team doing?' instead of 'How are you?'",
        "The CRM syncs in real-time, and last week we closed 10 deals!",
        "Gotta run, later!"
    ]
    for input in test_inputs:
        print(f"\nYou: {input}")
        for chunk in convo_stream(input, "test_thread"):
            print(chunk)