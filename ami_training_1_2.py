# ami_training_1_2
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 11, 2025
# Purpose: Ami's Training Path (v1.2) with natural multi-turn convo, reduced hardcoding, memory, and Markdown feedback
# Features: Stable last_topic, explicit knowledge/skills/lessons extraction, Pinecone storage, LLM-driven responses for natural flow

import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from pinecone_datastores import pinecone_index

# Initialize OpenAI, Embeddings, and Pinecone
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
PINECONE_INDEX = pinecone_index

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    last_topic: str
    mode: str
    convo_context: str
    tone: str

# Preset Brain
class PresetBrain:
    def spin_question(self, type, content):
        if type == "Situation":
            return f"Tell me about {content} in your current setup."
        elif type == "Problem":
            return f"What challenges do you face with {content}?"
        return "Unknown SPIN type"

    def challenger_teach(self, insight):
        return f"Here’s something they might not realize: {insight}"

# Training Path
class TrainingPath:
    def __init__(self):
        self.preset = PresetBrain()
        self.confidence_scores = {}

    def evaluate_confidence(self, response, lecture):
        accuracy = 90  # TODO: Replace with real metric
        relevance = 85
        fluency = 88
        return (accuracy + relevance + fluency) / 3

# Intent Detection with Stable last_topic
def detect_intent(state: State):
    latest_message = state["messages"][-1].content
    prior_messages = state["messages"][:-1]
    prior_contents = [m.content for m in prior_messages[-3:]] if prior_messages else []
    print(f"Debug: Prior messages (last 3) = {prior_contents}")
    
    intent_prompt = f"""
    Analyze this latest message, considering the convo so far:
    Latest Message: {latest_message}
    Prior Messages (last 3 for context): {prior_contents}
    Convo Context: {state.get('convo_context', 'No prior context')}
    Possible Intents:
    - "Teaching a sales skill": Instructing Ami on a sales move (e.g., "ask about," "try this").
    - "Sharing a sales lesson": Sharing a sales insight (e.g., "clients hate downtime").
    - "Sharing a lesson": General insight/knowledge (e.g., "sleep helps burn fat")—only if new info is introduced.
    - "Switching topics": Clear topic shift with no lesson vibe.
    - "Asking a question": Querying Ami (e.g., "What do you think?").
    - "Casual chat": Friendly banter (e.g., "Hey", "Not much")—default if vague and no prior context.
    - "Continuing chat": Responding to Ami or continuing the thread (e.g., "Okay I get it" after a lesson)—default if vague and prior context exists.
    Default to "Continuing chat" if message is short/vague and prior context is active; otherwise "Casual chat" if no context.
    Return the intent as a quoted phrase, e.g., "Casual chat".
    """
    intent = llm.invoke(intent_prompt).content.strip()
    print(f"Debug: Intent detected as '{intent}' for message: {latest_message}")

    last_topic = state.get("last_topic", "")
    if intent in ['"Teaching a sales skill"', '"Sharing a sales lesson"', '"Sharing a lesson"']:
        topic_prompt = f"""
        Guess the topic based on this message:
        Latest Message: {latest_message}
        Convo Context: {state.get('convo_context', 'No prior context')}
        Prior Topic (if any): {last_topic}
        Return only the topic as a short phrase (e.g., "sleep and health"), or "unknown" if unclear.
        """
        current_topic = llm.invoke(topic_prompt).content.strip()
    elif intent in ['"Casual chat"', '"Asking a question"', '"Continuing chat"']:
        topic_prompt = f"""
        Does this message suggest a new topic unrelated to the prior topic?
        Latest Message: {latest_message}
        Convo Context: {state.get('convo_context', 'No prior context')}
        Prior Topic (if any): {last_topic}
        Return only:
        - A new topic (e.g., "inventory management") if it shifts.
        - "{last_topic}" if it aligns or doesn’t shift.
        - "unknown" if no prior topic and no clear shift.
        """
        current_topic = llm.invoke(topic_prompt).content.strip()
    elif intent == '"Switching topics"':
        topic_prompt = f"""
        Guess the new topic from this message:
        Latest Message: {latest_message}
        Return a short phrase (e.g., "sleep and health") or "unknown".
        """
        current_topic = llm.invoke(topic_prompt).content.strip()

    context_prompt = f"""
    Summarize the convo so far in one sentence, blending this message with prior context:
    Latest Message: {latest_message}
    Prior Context: {state.get('convo_context', 'No prior context')}
    """
    convo_context = llm.invoke(context_prompt).content.strip()

    if last_topic and last_topic != current_topic and intent == '"Switching topics"':
        if not is_topic_saved(last_topic):
            summary = generate_summary(prior_messages)
            save_to_memory(last_topic, summary)

    return {
        "intent": intent,
        "last_topic": current_topic,
        "convo_context": convo_context
    }

# Ami Node with Markdown Feedback
def ami_node(state: State, trainer: TrainingPath):
    result = detect_intent(state)
    intent = result["intent"]
    latest_message = state["messages"][-1].content
    tone = state.get("tone", "friendly")
    print(f"Debug: ami_node received intent: '{intent}'")

    if intent in ['"Teaching a sales skill"', '"Sharing a sales lesson"', '"Sharing a lesson"', '"Continuing chat"']:
        print(f"Debug: Entering extraction block for intent: {intent}")
        knowledge_prompt = f"""
        Extract knowledge, skills, or lessons from this message and return it as a valid JSON object:
        Message: {latest_message}
        Intent: {intent}
        Prior Topic (if any): {state.get('last_topic', 'unknown')}
        Use this exact JSON structure:
        {{
            "topic": "The topic (e.g., 'inventory management', 'sleep and health'; use Prior Topic for 'Continuing chat' unless shifted)",
            "method": "One of: spin_situation, spin_problem, challenger_teach (sales skills), or 'none'",
            "content": "Core idea (e.g., 'their current inventory system', 'sleep impacts weight')",
            "knowledge": ["Factual info (e.g., 'Clients use spreadsheets')"],
            "skills": ["Techniques (e.g., 'Ask about their current inventory system')"],
            "lessons": ["Best practices or insights (e.g., 'Good sleep helps control weight')"]
        }}
        If the message is vague (e.g., 'Okay'), infer minimal context or leave fields empty.
        """
        raw_response = llm.invoke(knowledge_prompt).content.strip()
        cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()

        try:
            extracted_knowledge = json.loads(cleaned_response)
            print(f"Debug: Extracted knowledge: {extracted_knowledge}")
        except json.JSONDecodeError as e:
            print(f"Debug: JSON error: {e} - Cleaned response was: '{cleaned_response}'")
            extracted_knowledge = {"topic": state.get("last_topic", "unknown"), "method": "none", "content": "N/A", "knowledge": [], "skills": [], "lessons": []}

        markdown_summary = f"""
### Here's what I picked up:
- **Topic**: {extracted_knowledge['topic']}
- **Knowledge**: {', '.join(extracted_knowledge['knowledge']) if extracted_knowledge['knowledge'] else 'None yet'}
- **Skills**: {', '.join(extracted_knowledge['skills']) if extracted_knowledge['skills'] else 'None yet'}
- **Lessons**: {', '.join(extracted_knowledge['lessons']) if extracted_knowledge['lessons'] else 'None yet'}
        """

        if intent in ['"Teaching a sales skill"', '"Sharing a sales lesson"']:
            if "spin" in extracted_knowledge["method"]:
                type = extracted_knowledge["method"].split("_")[1].capitalize()
                content = extracted_knowledge["content"].replace("Ask about ", "").replace("the client's ", "").strip()
                response = trainer.preset.spin_question(type, content)
                intro = "Nice tip! I’ll try asking: " if intent == '"Teaching a sales skill"' else "Love that insight—here’s how I’d pitch it: "
            elif "challenger" in extracted_knowledge["method"]:
                content = extracted_knowledge["content"]
                response = trainer.preset.challenger_teach(content)
                intro = "Love that insight—here’s how I’d pitch it: "
            else:
                response = "I’ll need a bit more to work with there!"
                intro = "Hmm, "
            confidence = trainer.evaluate_confidence(response, latest_message)
            trainer.confidence_scores[response] = confidence
            follow_up_prompt = f"""
            Based on this response ({confidence}% confidence) and convo context ({state.get('convo_context', 'No context')}):
            {response}
            Ask a concise, friendly follow-up question.
            """
            follow_up = llm.invoke(follow_up_prompt).content.strip()
            response = f"{markdown_summary}\n\n{intro}{response} How about this: {follow_up}"
        elif intent in ['"Sharing a lesson"', '"Continuing chat"']:
            response_prompt = f"""
            Respond naturally to this message in the ongoing convo, keeping it friendly and relevant:
            Latest Message: {latest_message}
            Extracted Knowledge: {json.dumps(extracted_knowledge)}
            Convo Context: {state.get('convo_context', 'No context')}
            Prior Messages (last 3): {[m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else ['None yet']}
            Tone: {tone}
            Include a concise, curious follow-up question to keep the chat flowing.
            """
            response = llm.invoke(response_prompt).content.strip()
            confidence = trainer.evaluate_confidence(response, latest_message)
            trainer.confidence_scores[response] = confidence
            response = f"{markdown_summary}\n\n{response}"

        save_to_pinecone({
            "topic": extracted_knowledge["topic"],
            "response": response,
            "confidence": confidence,
            "source": "Rep A, 3/11/25",
            "knowledge": extracted_knowledge["knowledge"],
            "skills": extracted_knowledge["skills"],
            "lessons": extracted_knowledge["lessons"]
        })

    elif intent == '"Switching topics"':
        last_topic = state.get('last_topic', '')
        from_text = f"from {last_topic}" if last_topic and last_topic != "unknown" else "from what we were chatting about"
        response = f"Cool, looks like we’re shifting gears {from_text}. What’s this {result['last_topic']} angle about?"
    elif intent == '"Asking a question"':
        response = f"Good one! Let me think... What’s your take on it first?"
    elif intent == '"Casual chat"':
        prior_contents = [m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else []
        casual_prompt = f"""
        Respond naturally to this casual message, keeping it friendly and open-ended:
        Latest Message: {latest_message}
        Prior Messages (last 3): {prior_contents if prior_contents else ['None yet']}
        Convo Context: {state.get('convo_context', 'No prior context')}
        Tone: {tone}
        """
        response = llm.invoke(casual_prompt).content.strip()
    else:
        response = "Hey, I’m here to soak up your wisdom—whatcha got for me?"

    last_topic_to_use = extracted_knowledge["topic"] if intent in ['"Teaching a sales skill"', '"Sharing a sales lesson"', '"Sharing a lesson"', '"Continuing chat"'] else result["last_topic"]
    return {
        "prompt_str": response,
        "last_topic": last_topic_to_use,
        "convo_context": result["convo_context"],
        "tone": tone
    }

# Helpers
def generate_summary(messages):
    conversation = "\n".join([m.content for m in messages])
    summary_prompt = f"Summarize this convo in one or two sentences:\n{conversation}"
    return llm.invoke(summary_prompt).content.strip()

def is_topic_saved(topic: str) -> bool:
    return False

def save_to_pinecone(knowledge: dict):
    print(f"Saved to Pinecone: {knowledge}")

def save_to_memory(topic: str, summary: str):
    print(f"Saved to MemorySaver: {topic} - {summary}")

# Graph Setup
trainer = TrainingPath()
graph_builder = StateGraph(State)
graph_builder.add_node("ami", lambda state: ami_node(state, trainer))
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

# Streaming with Markdown Chunking
def convo_stream(user_input, user_id, thread_id="learning_thread"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}}) or {}
    channel_values = checkpoint.get("channel_values", {})
    history = channel_values.get("messages", [])
    new_message = HumanMessage(content=user_input)
    updated_messages = history + [new_message]
    
    try:
        state = {
            "messages": updated_messages,
            "prompt_str": channel_values.get("prompt_str", ""),
            "last_topic": channel_values.get("last_topic", ""),
            "mode": channel_values.get("mode", "learner"),
            "convo_context": channel_values.get("convo_context", ""),
            "tone": channel_values.get("tone", "friendly")
        }
        print(f"Debug: Before invoke - state: {state['last_topic'] = }, {state['convo_context'] = }")
        
        state = convo_graph.invoke(state, {"configurable": {"thread_id": thread_id}})
        print(f"Debug: After invoke - state: {state['last_topic'] = }, {state['convo_context'] = }")
        
        response_lines = state["prompt_str"].split('\n')
        for line in response_lines:
            if line.strip():
                yield f"data: {json.dumps({'message': line.strip()})}\n\n"
        
        updated_state = {
            "messages": state["messages"],
            "prompt_str": state["prompt_str"],
            "last_topic": state["last_topic"],
            "mode": state["mode"],
            "convo_context": state["convo_context"],
            "tone": state["tone"]
        }
        convo_graph.update_state({"configurable": {"thread_id": thread_id}}, updated_state, as_node="ami")
        updated_checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}}) or {}
        channel_values = updated_checkpoint.get("channel_values", {})
        print(f"Debug: After update_state - checkpoint: {channel_values.get('last_topic', 'Not found') = }, {channel_values.get('convo_context', 'Not found') = }")
        print(f"AI response saved: {state['prompt_str']}...")
        
    except Exception as e:
        error_msg = f"Error in stream: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

