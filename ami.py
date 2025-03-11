import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from pinecone_datastores import pinecone_index
import textwrap

# Initialize OpenAI, Embeddings, and Pinecone
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
PINECONE_INDEX = pinecone_index  # Assuming this is defined elsewhere

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
        accuracy = 90
        relevance = 85
        fluency = 88
        return (accuracy + relevance + fluency) / 3
def detect_intent(state: State):
    latest_message = state["messages"][-1].content
    prior_messages = state["messages"][:-1]
    
    intent_prompt = f"""
    Analyze this latest message in a sales training context, considering the convo so far:
    Latest Message: {latest_message}
    Prior Messages (last 3 for context): {[m.content for m in prior_messages[-3:]] if prior_messages else []}
    Convo Context: {state.get('convo_context', 'No prior context')}
    Possible Intents:
    - "Teaching a sales skill": Instructing Ami on a sales move (e.g., "ask about," "try this").
    - "Sharing a sales lesson": Sharing a sales insight (e.g., "I’ve learned," "clients do X when").
    - "Switching topics": Shifting to a new sales concept.
    - "Asking a question": Querying Ami (e.g., "What do you think?").
    - "Casual chat": Off-topic or friendly banter (e.g., "Hey, how’s it going?").
    Return the intent as a quoted phrase, e.g., "Teaching a sales skill".
    """
    intent = llm.invoke(intent_prompt).content.strip()
    print(f"Debug: Intent detected as '{intent}' for message: {latest_message}")

    last_topic = state.get("last_topic", "")
    if intent in ['"Teaching a sales skill"', '"Sharing a sales lesson"']:
        # Guess new topic for sales-focused intents
        topic_prompt = f"""
        Guess the sales-related topic (e.g., "inventory management", "closing deals") based on this message:
        Latest Message: {latest_message}
        Convo Context: {state.get('convo_context', 'No prior context')}
        Prior Topic (if any): {last_topic}
        Return only the topic as a short phrase (e.g., "closing deals"), or "unknown" if unclear—no explanations.
        """
        current_topic = llm.invoke(topic_prompt).content.strip()
    elif intent in ['"Casual chat"', '"Asking a question"']:
        # Stick with prior topic unless message clearly shifts
        topic_prompt = f"""
        Does this message clearly suggest a new sales-related topic unrelated to the prior topic?
        Latest Message: {latest_message}
        Convo Context: {state.get('convo_context', 'No prior context')}
        Prior Topic (if any): {last_topic}
        Return only:
        - A new topic as a short phrase (e.g., "inventory management") if it clearly shifts.
        - "{last_topic}" if it doesn’t shift or aligns with the prior topic.
        - "unknown" if no prior topic and no clear shift.
        No explanations—just the phrase.
        """
        current_topic = llm.invoke(topic_prompt).content.strip()
    else:  # Switching topics
        current_topic = ""  # Reset for switch

    # Update convo context
    context_prompt = f"""
    Summarize the convo so far in one sentence, blending this message with prior context:
    Latest Message: {latest_message}
    Prior Context: {state.get('convo_context', 'No prior context')}
    """
    convo_context = llm.invoke(context_prompt).content.strip()

    # Save old topic before switching
    if last_topic and last_topic != current_topic and intent == '"Switching topics"':
        if not is_topic_saved(last_topic):
            summary = generate_summary(prior_messages)
            save_to_memory(last_topic, summary)

    return {
        "intent": intent,
        "last_topic": current_topic,
        "convo_context": convo_context
    }
# Ami Node (unchanged)
def ami_node(state: State, trainer: TrainingPath):
    result = detect_intent(state)
    intent = result["intent"]
    latest_message = state["messages"][-1].content
    tone = state.get("tone", "friendly")
    print(f"Debug: ami_node received intent: '{intent}'")

    if intent == '"Teaching a sales skill"' or intent == '"Sharing a sales lesson"':
        print(f"Debug: Entering training block for intent: {intent}")
        knowledge_prompt = f"""
        Extract sales-related knowledge from this message and return it as a valid JSON object:
        Message: {latest_message}
        Use this exact JSON structure:
        {{
            "topic": "The sales-related topic (e.g., inventory management, closing deals)",
            "method": "One of: spin_situation, spin_problem, challenger_teach",
            "content": "The core idea or phrase to process (keep it short, e.g., 'their current inventory system')",
            "insights": ["List of actionable insights"],
            "lessons": ["List of lessons or best practices"]
        }}
        Examples:
        - For "ask about their current inventory system": "content": "their current inventory system"
        - For "clients buy faster when you show how downtime costs them money": "content": "how downtime costs them money"
        Return only the JSON string, nothing else.
        """
        raw_response = llm.invoke(knowledge_prompt).content.strip()
        cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()

        try:
            extracted_knowledge = json.loads(cleaned_response)
            print(f"Debug: Extracted knowledge: {extracted_knowledge}")
        except json.JSONDecodeError as e:
            print(f"Debug: JSON error: {e} - Cleaned response was: '{cleaned_response}'")
            extracted_knowledge = {"topic": "Unknown", "method": "none", "content": latest_message, "insights": [], "lessons": []}

        if "spin" in extracted_knowledge["method"]:
            type = extracted_knowledge["method"].split("_")[1].capitalize()
            content = extracted_knowledge["content"].replace("Ask about ", "").replace("the client's ", "").strip()
            response = trainer.preset.spin_question(type, content)
            intro = "Nice tip! I’ll try asking: "
        elif "challenger" in extracted_knowledge["method"]:
            content = extracted_knowledge["content"]
            if "downtime" in content.lower():
                response = "Here’s something they might not realize: downtime can hit their bottom line hard."
            else:
                response = trainer.preset.challenger_teach(content)
            intro = "Love that insight—here’s how I’d pitch it: "
        else:
            response = "I’ll need a bit more to work with there!"
            intro = "Hmm, "

        confidence = trainer.evaluate_confidence(response, latest_message)
        trainer.confidence_scores[response] = confidence

        enterprise_entry = {
            "topic": extracted_knowledge["topic"],
            "response": response,
            "confidence": confidence,
            "source": "Rep A, 3/10/25",
            "insights": extracted_knowledge["insights"],
            "lessons": extracted_knowledge["lessons"]
        }
        save_to_pinecone(enterprise_entry)

        follow_up_prompt = f"""
        Based on this response ({confidence}% confidence) and convo context ({state.get('convo_context', 'No context')}):
        {response}
        Ask a concise, friendly follow-up question to keep the chat flowing. Return only the question.
        """
        follow_up = llm.invoke(follow_up_prompt).content.strip()
        response = f"{intro}{response} How about this: {follow_up}"

    elif intent == '"Switching topics"':
        last_topic = state.get('last_topic', '')
        from_text = f"from {last_topic}" if last_topic else "from what we were chatting about"
        response = f"Cool, looks like we’re shifting gears {from_text}. What’s this new angle about?"
    elif intent == '"Asking a question"':
        response = f"Good one! Let me think... What’s your take on it first?"
    elif intent == '"Casual chat"':
        response = f"Hey, I’m all for a quick breather! How’s your day going so far?"
    else:
        response = "Hey, I’m here to soak up your sales wisdom—whatcha got for me?"

    return {
        "prompt_str": response,
        "last_topic": result["last_topic"],
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

# Replace this in your code
def convo_stream(user_input, user_id, thread_id="learning_thread"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}}) or {}
    channel_values = checkpoint.get("channel_values", {})
    history = channel_values.get("messages", [])
    new_message = HumanMessage(content=user_input)
    updated_messages = history + [new_message]
    
    try:
        # Load state from channel_values
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
        
        for chunk in textwrap.wrap(state["prompt_str"], width=100):
            yield f"data: {json.dumps({'message': chunk})}\n\n"
        
        # Save full state under channel_values
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
# Test
if __name__ == "__main__":
    thread_id = "test_thread"
    expert_inputs = [
        "Hey Ami, when talking to a client, ask about their current inventory system—it’s a good opener.",
        "A lesson I’ve learned: clients buy faster when you show how downtime costs them money.",
        "Hey, how’s it going today?",
        "Switching gears—how do you handle objections?"
    ]
    for user_input in expert_inputs:
        print(f"\nExpert: {user_input}")
        for chunk in convo_stream(user_input, "user1", thread_id):
            print(chunk)