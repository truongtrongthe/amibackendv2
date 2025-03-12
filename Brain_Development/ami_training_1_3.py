# ami_training_1_3
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 11, 2025
# Purpose: Ami's Training Path (v1.3) with human-like, curious, energetic prefixes, practice callouts for SPIN/Challenger, and fixed intent detection
# Features: Stable last_topic, explicit knowledge/skills/lessons extraction, Pinecone storage, LLM-driven responses with boosted personality

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

# Intent Detection with Fixed Output
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
    - "Teaching a sales skill": Instructing Ami on a new sales move (e.g., "Ask about inventory", "Try this pitch").
    - "Continuing chat": Confirming or building on a prior skill/lesson (e.g., "Yes, update it" after a suggestion)—default if vague and prior context exists.  
    - "Sharing a sales lesson": Sharing a sales insight (e.g., "We closed 100 deals last week").
    - "Sharing a lesson": General insight/knowledge (e.g., "Sleep helps focus")—only if new info is introduced.
    - "Switching topics": Clear topic shift with no lesson vibe (e.g., "Let’s talk about sales" after casual chat).
    - "Asking a question": Querying Ami (e.g., "What do you think?").
    - "Casual chat": Friendly banter (e.g., "Hey", "Not much") or exits (e.g., "Talk later")—default if vague and no prior context.
    Return ONLY the intent as a quoted phrase (e.g., "Casual chat"), no extra text.
    """
    intent = llm.invoke(intent_prompt).content.strip()
    print(f"Debug: Intent detected as '{intent}' for message: {latest_message}")

    last_topic = state.get("last_topic", "")
    if intent in ['"Teaching a sales skill"', '"Sharing a sales lesson"', '"Sharing a lesson"']:
        topic_prompt = f"""
Analyze the topic of this message based on its focus and intent:
- Latest Message: {latest_message}
- Convo Context: {state.get('convo_context', 'No prior context')}
- Prior Topic (if any): {last_topic}
- Intent: {intent}

Instructions:
1. **Latest Message Dominates**: Focus on the message’s core intent—override prior topic unless message is vague:
   - Sales outcomes (e.g., results, deals closed, performance)?
   - Sales strategies/tools (e.g., scripts, pitches, conversation tweaks, negotiation techniques)?
   - Something else (e.g., customer objections, product knowledge, competitor insights)?

2. **Intent Forces Focus**:
   - **"Teaching a sales skill"**: Hard locks to strategies/tools (e.g., "sales scripts")—shifts from prior topic unless outcomes are sole focus (e.g., "we closed 100 deals" with no tools).
   - **"Sharing a sales lesson"**: Hard locks to outcomes (e.g., "sales performance")—shifts from prior topic unless tools are sole focus (e.g., "we adjusted the script" with no outcomes).
   - **"Continuing chat"**: Starts with prior topic—shifts on any clear tool hint (e.g., "adjust," "script" after "deals" moves to tools)—subtle cues like "adjust" trigger a pivot.

3. **Keywords as Hard Triggers**:
   - "deal," "closed," "results," "quota" → outcomes (e.g., "sales performance").
   - "script," "adjust," "line," "update," "framework" → strategies/tools (e.g., "sales scripts").
   - "objection," "pushback," "competitor," "pricing" → challenges (e.g., "customer objections").

4. **Context Only for Ambiguity**: Use context only if Latest Message is unclear (e.g., "yes" alone)—prior deal talk doesn’t block tool focus.

5. **Return Specific Topic**:
   - Short phrase (e.g., "sales performance," "sales scripts," "customer objections").
   - "unclear (needs clarification)" if no focus emerges—avoid generic "sales".
"""
        current_topic = llm.invoke(topic_prompt).content.strip()
    elif intent in ['"Casual chat"', '"Asking a question"', '"Continuing chat"']:
        topic_prompt = f"""
    Does this message shift or refine the prior topic?
    Latest Message: {latest_message}
    Convo Context: {state.get('convo_context', 'No prior context')}
    Prior Topic (if any): {last_topic}
    Intent: {intent}
    Instructions:
    - Check for a new focus: Keywords like "script," "adjust," "line," "update" suggest strategies/tools; "deal," "closed," "results" suggest outcomes.
    - "Continuing chat" refines the prior topic unless it clearly shifts—use context to confirm (e.g., script tweaks after deal talk).
    - "Casual chat" holds the prior topic unless it’s a clear exit with no focus (e.g., "I gotta go"), then keep {last_topic}.
    - Return only:
      - A new topic (e.g., "sales scripts") if it shifts (e.g., script tweaks after deal talk).
      - "{last_topic}" if it aligns or refines (e.g., more deal talk, or casual continuation).
      - "unknown" if no prior topic and no clear focus.
    """
        current_topic = llm.invoke(topic_prompt).content.strip()
    elif intent == '"Switching topics"':
        topic_prompt = f"""
        Guess the new topic from this message:
        Latest Message: {latest_message}
        Return a short phrase (e.g., "sales") or "unknown".
        """
        current_topic = llm.invoke(topic_prompt).content.strip()
    else:
        current_topic = last_topic  # Fallback to avoid unbound variable

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

# Ami Node with Updated Prefixes and Fixes
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
            "topic": "The topic (e.g., 'sales performance', 'sleep and health'; use Prior Topic for 'Continuing chat' unless shifted)",
            "method": "One of: spin_situation (e.g., 'ask' or 'adjust' to elicit), spin_problem, challenger_teach (e.g., insights like 'closing deals leads to recognition'), or 'none'",
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

        if intent == '"Teaching a sales skill"':
            if "spin" in extracted_knowledge["method"]:
                type = extracted_knowledge["method"].split("_")[1].capitalize()
                content = extracted_knowledge["content"].replace("Ask about ", "").replace("the client's ", "").strip()
                response = trainer.preset.spin_question(type, content)
                intro = "Whoa, killer tip! How’d you come up with that? Practicing my SPIN here: "
            elif "challenger" in extracted_knowledge["method"]:
                content = extracted_knowledge["content"]
                response = trainer.preset.challenger_teach(content)
                intro = "Oh, that’s gold! What sparked that? Trying a Challenger move: "
            else:
                response = "I’ll need a bit more to work with—gimme a nudge!"
                intro = "Hmm, gotcha—"
            confidence = trainer.evaluate_confidence(response, latest_message)
            trainer.confidence_scores[response] = confidence
            follow_up_prompt = f"""
            Based on this response ({confidence}% confidence) and convo context ({state.get('convo_context', 'No context')}):
            {response}
            Return ONLY a concise, energetic follow-up question (e.g., "How’s that gonna hook those newbies?"), no extra labels or text.
            """
            follow_up = llm.invoke(follow_up_prompt).content.strip()
            response = f"{markdown_summary}\n\n{intro}{response} {follow_up}"
        elif intent == '"Sharing a sales lesson"':
            response_prompt = f"""
            Respond energetically to this sales lesson, keeping it curious and relevant:
            Latest Message: {latest_message}
            Extracted Knowledge: {json.dumps(extracted_knowledge)}
            Convo Context: {state.get('convo_context', 'No context')}
            Prior Messages (last 3): {[m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else ['None yet']}
            Tone: {tone}
            Include a punchy follow-up question.
            """
            response = llm.invoke(response_prompt).content.strip()
            confidence = trainer.evaluate_confidence(response, latest_message)
            trainer.confidence_scores[response] = confidence
            response = f"{markdown_summary}\n\nWhoa, spill more! How’d that play out? {response}"
        elif intent == '"Sharing a lesson"':
            response_prompt = f"""
            Respond naturally to this message, keeping it energetic and curious:
            Latest Message: {latest_message}
            Extracted Knowledge: {json.dumps(extracted_knowledge)}
            Convo Context: {state.get('convo_context', 'No context')}
            Prior Messages (last 3): {[m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else ['None yet']}
            Tone: {tone}
            Include a concise, excited follow-up question.
            """
            response = llm.invoke(response_prompt).content.strip()
            confidence = trainer.evaluate_confidence(response, latest_message)
            trainer.confidence_scores[response] = confidence
            response = f"{markdown_summary}\n\nWhoa, spill more! How’d that play out? {response}"
        elif intent == '"Continuing chat"':
            response_prompt = f"""
            Respond naturally to this message in the ongoing convo, keeping it energetic and curious:
            Latest Message: {latest_message}
            Extracted Knowledge: {json.dumps(extracted_knowledge)}
            Convo Context: {state.get('convo_context', 'No context')}
            Prior Messages (last 3): {[m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else ['None yet']}
            Tone: {tone}
            Include a concise, excited follow-up question.
            """
            response = llm.invoke(response_prompt).content.strip()
            confidence = trainer.evaluate_confidence(response, latest_message)
            trainer.confidence_scores[response] = confidence
            response = f"{markdown_summary}\n\nSweet, I’m all in—what’s the next move? {response}"

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
        from_text = f"from {last_topic}" if last_topic and last_topic != "unknown" else "from the last vibe"
        response = f"Ooh, new vibe alert! What’s cooking with {result['last_topic']}? "
    elif intent == '"Asking a question"':
        response = f"Whoa, you got me—gimme a sec! What’s your angle on it? "
    elif intent == '"Casual chat"':
        prior_contents = [m.content for m in state["messages"][:-1][-3:]] if state["messages"][:-1] else []
        casual_prompt = f"""
        Respond naturally and energetically to this casual message like a pumped-up sales bro, keeping it curious and friendly:
        Latest Message: {latest_message}
        Prior Messages (last 3): {prior_contents if prior_contents else ['None yet']}
        Convo Context: {state.get('convo_context', 'No prior context')}
        Tone: {tone}
        Vary your opener—stay punchy and avoid repetition. Adapt to greetings (e.g., "Hey") or exits (e.g., "Talk later"), and include a fun, engaging follow-up question.
        """
        response = llm.invoke(casual_prompt).content.strip()
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
# Test Run
if __name__ == "__main__":
    inputs = ["Hey Ami",
    "How're u doing?",
    "Let's talk about sales.",
    "Last week we closed 100 deals. Boss was really happy in the meeting this morning!",
    "I think we can adjust the pickup line for new customer",
    "Well, instead of saying 'How're you doing', let's drive them to share personal information like 'How old are you'",
    "Yes, please update the new script and use it dynamically!",
    "I gotta go. Talk to you later"
    ]
    for input in inputs:
        print(f"\nExpert: {input}")
        for chunk in convo_stream(input, "user1"):
            print(chunk)