# ami_training_1_15
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 12, 2025
# Purpose: Ami's Training Path (v1.15) with refined intent and topic precision
# Features: Exact rules, forced topic shifts for skills, fixed SPIN

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
        type = type.lower()
        print(f"Debug: SPIN type received: {type}")
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
            f"Here’s something they might not realize: {insight}",
            f"Reframe this for them: {insight}",
            f"Shake their view—check this out: {insight}"
        ]
        return choice(templates)

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

# Intent Detection
def detect_intent(state: State):
    msg = state["messages"][-1].content.lower().strip()
    prior_msgs = [m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else []
    print(f"Debug: Prior messages (last 3) = {prior_msgs}")
    
    casual_phrases = ["hey", "hi", "hello", "yo", "bye", "talk later", "gotta go", "see ya"]
    if any(msg == phrase or msg.startswith(phrase + " ") or msg.startswith(phrase + ",") for phrase in casual_phrases):
        intent = '"Casual chat"'
        print(f"Debug: Rule-based intent: '{intent}' (casual) for: {msg}")
    else:
        intent_prompt = f"""
        Analyze this message to determine its intent, strongly considering prior context:
        - Latest Message: {msg}
        - Prior Messages: {prior_msgs}
        - Context: {state.get('convo_context', 'No prior context')}
        Possible Intents:
        - "Teaching a sales skill": Suggesting a sales technique or question (e.g., "let’s ask", "adjust", "script").
        - "Continuing chat": Building on prior topic without new skill/lesson (e.g., "yes", "how're u doing?" after greeting, confirmations).
        - "Sharing a sales lesson": Sharing a sales outcome/insight (e.g., "closed deals", "boss happy", "sales numbers").
        - "Sharing a lesson": General non-sales insight.
        - "Switching topics": Explicit topic shift (e.g., "let’s talk about sales").
        - "Asking a question": Standalone query without continuation.
        - "Casual chat": Friendly banter or exit not caught by rules.
        - Strongly favor "Continuing chat" for follow-ups like "how're u doing?" after greetings or "yes" after suggestions.
        - Strongly favor "Sharing a sales lesson" for sales-specific results (e.g., "deals", "closed", "boss").
        Return ONLY the intent as a quoted phrase (e.g., "Casual chat").
        """
        intent = llm.invoke(intent_prompt).content
        print(f"Debug: LLM intent: '{intent}' for: {msg}")

    last_topic = state.get("last_topic", "unknown")
    topic_prompt = f"""
    Infer a concise topic from this message:
    - Latest Message: {msg}
    - Intent: {intent}
    - Prior Topic: {last_topic}
    Instructions:
    - Prioritize latest message keywords: "script", "line", "adjust", "pickup", "update" → "sales scripts"; "deal", "closed", "boss", "sales" → "sales performance"; "workload", "team" → "team management".
    - For "Teaching a sales skill", override prior topic with tools/techniques (e.g., "adjust", "ask", "script" → "sales scripts").
    - For "Sharing a sales lesson", focus on outcomes (e.g., "deal", "closed", "boss" → "sales performance").
    - Return "unknown" if unclear or intent is "Casual chat"—don’t inherit prior topic unless "Continuing chat" and relevant.
    - Keep it short—no essays.
    """
    current_topic = llm.invoke(topic_prompt).content
    if not current_topic:
        current_topic = "unknown"
    
    if intent == '"Teaching a sales skill"':
        if any(kw in msg for kw in ["script", "line", "adjust", "pickup", "update"]):
            current_topic = "sales scripts"
        elif any(kw in msg for kw in ["workload", "team"]):
            current_topic = "team management"
    elif intent in ['"Casual chat"', '"Asking a question"']:
        current_topic = "unknown"
    elif intent == '"Continuing chat"' and current_topic == "unknown" and last_topic != "unknown":
        current_topic = last_topic

    context_prompt = f"""
    Summarize the convo so far in one sentence, blending this message with prior context:
    Latest Message: {msg}
    Prior Context: {state.get('convo_context', 'No prior context')}
    """
    convo_context = llm.invoke(context_prompt).content

    return {
        "intent": intent,
        "last_topic": current_topic,
        "convo_context": convo_context
    }

# Ami Node
def ami_node(state: State, trainer: TrainingPath):
    result = detect_intent(state)
    intent = result["intent"]
    msg = state["messages"][-1].content
    tone = state.get("tone", "friendly")
    print(f"Debug: ami_node received intent: '{intent}'")

    if intent in ['"Teaching a sales skill"', '"Sharing a sales lesson"', '"Sharing a lesson"', '"Continuing chat"']:
        print(f"Debug: Entering extraction block for intent: {intent}")
        knowledge_prompt = f"""
        Extract knowledge, skills, or lessons from this message:
        Message: {msg}
        Intent: {intent}
        Prior Topic: {state.get('last_topic', 'unknown')}
        JSON structure:
        {{
            "topic": "The topic",
            "method": "spin_situation, spin_problem, spin_implication, spin_need_payoff (for skills), challenger_teach (for lessons), or 'none'",
            "content": "Core idea",
            "knowledge": ["Facts"],
            "skills": ["Techniques"],
            "lessons": ["Insights"]
        }}
        - For "Teaching a sales skill": Use SPIN (e.g., "spin_situation", "spin_need_payoff") if suggesting a question/technique, else "none".
        - For "Sharing a sales lesson" or "Sharing a lesson": Use "challenger_teach" if an insight, else "none".
        - For "Continuing chat": "none" unless clear skill/lesson.
        - If vague, minimal context or empty fields.
        """
        raw_response = llm.invoke(knowledge_prompt).content
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

        if intent == '"Teaching a sales skill"':
            if "spin" in extracted_knowledge["method"]:
                type = extracted_knowledge["method"].replace("spin_", "")
                content = extracted_knowledge["content"].strip()
                response = trainer.preset.spin_question(type, content)
                intro = "Whoa, killer tip! How’d you come up with that? Practicing my SPIN here: "
            elif "challenger" in extracted_knowledge["method"]:
                content = extracted_knowledge["lessons"][0] if extracted_knowledge["lessons"] else extracted_knowledge["content"]
                response = trainer.preset.challenger_teach(content)
                intro = "Oh, that’s gold! What sparked that? Trying a Challenger move: "
            else:
                response = "I’ll need a bit more to work with—gimme a nudge!"
                intro = "Hmm, gotcha—"
            confidence = trainer.evaluate_confidence(response, msg)
            trainer.confidence_scores[response] = confidence
            follow_up_prompt = f"""
            Based on this response ({confidence}% confidence) and convo context ({state.get('convo_context', 'No context')}):
            {response}
            Return ONLY a concise, energetic follow-up question.
            """
            follow_up = llm.invoke(follow_up_prompt).content
            response = f"{markdown_summary}\n\n{intro}{response} {follow_up}"
        elif intent == '"Sharing a sales lesson"':
            response_prompt = f"""
            Respond energetically to this sales lesson, keeping it curious and relevant:
            Latest Message: {msg}
            Extracted Knowledge: {json.dumps(extracted_knowledge)}
            Convo Context: {state.get('convo_context', 'No context')}
            Prior Messages (last 3): {[m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else ['None yet']}
            Tone: {tone}
            Include a punchy follow-up question.
            """
            response = llm.invoke(response_prompt).content
            confidence = trainer.evaluate_confidence(response, msg)
            trainer.confidence_scores[response] = confidence
            response = f"{markdown_summary}\n\nWhoa, spill more! How’d that play out? {response}"
        elif intent == '"Sharing a lesson"':
            response_prompt = f"""
            Respond naturally to this message, keeping it energetic and curious:
            Latest Message: {msg}
            Extracted Knowledge: {json.dumps(extracted_knowledge)}
            Convo Context: {state.get('convo_context', 'No context')}
            Prior Messages (last 3): {[m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else ['None yet']}
            Tone: {tone}
            Include a concise, excited follow-up question.
            """
            response = llm.invoke(response_prompt).content
            confidence = trainer.evaluate_confidence(response, msg)
            trainer.confidence_scores[response] = confidence
            response = f"{markdown_summary}\n\nWhoa, spill more! How’d that play out? {response}"
        elif intent == '"Continuing chat"':
            response_prompt = f"""
            Respond naturally to this message in the ongoing convo, keeping it energetic and curious:
            Latest Message: {msg}
            Extracted Knowledge: {json.dumps(extracted_knowledge)}
            Convo Context: {state.get('convo_context', 'No context')}
            Prior Messages (last 3): {[m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else ['None yet']}
            Tone: {tone}
            Include a concise, excited follow-up question.
            """
            response = llm.invoke(response_prompt).content
            confidence = trainer.evaluate_confidence(response, msg)
            trainer.confidence_scores[response] = confidence
            response = f"{markdown_summary}\n\nSweet, I’m all in—what’s the next move? {response}"

        save_to_pinecone({
            "topic": extracted_knowledge["topic"],
            "response": response,
            "confidence": confidence,
            "source": "Rep A, 3/12/25",
            "knowledge": extracted_knowledge["knowledge"],
            "skills": extracted_knowledge["skills"],
            "lessons": extracted_knowledge["lessons"]
        })

    elif intent == '"Switching topics"':
        last_topic = state.get('last_topic', 'unknown')
        from_text = f"from {last_topic}" if last_topic and last_topic != "unknown" else "from the last vibe"
        response = f"Ooh, new vibe alert! What’s cooking with {result['last_topic']}? "
    elif intent == '"Asking a question"':
        response = f"Whoa, you got me—gimme a sec! What’s your angle on it? "
    elif intent == '"Casual chat"':
        prior_contents = [m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else []
        casual_prompt = f"""
        Respond naturally and energetically to this casual message like a pumped-up sales bro:
        Latest Message: {msg}
        Prior Messages (last 3): {prior_contents if prior_contents else ['None yet']}
        Convo Context: {state.get('convo_context', 'No context')}
        Tone: {tone}
        Vary your opener—stay punchy. Adapt to greetings or exits, and include a fun follow-up question.
        """
        response = llm.invoke(casual_prompt).content
    else:
        response = "Yo, I’m pumped to learn—what’s on your mind? "

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
    return llm.invoke(summary_prompt).content

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
            "last_topic": channel_values.get("last_topic", "unknown"),
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

# Test Run
if __name__ == "__main__":
    inputs = ["Hey,it's me Brian",
              "How're u doing?",
              "Let's talk about sales.",
              "Last week we closed 100 deals. Boss was really happy!",
              "I think we can adjust the pickup line for new customers",
              "Well, instead of saying 'How're you doing', let's ask 'How’s your team handling the workload?'",
              "Yes, please update the new script!",
              "I gotta go. Talk to you later"]
    for input in inputs:
        print(f"\nExpert: {input}")
        for chunk in convo_stream(input, "user1"):
            print(chunk)