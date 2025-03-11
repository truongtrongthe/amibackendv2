import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
import textwrap

llm = ChatOpenAI(model="gpt-4o", streaming=True)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    meat_points: int
    pending_save: dict
    teach_streak: bool
    last_topic: str

def detect_intent(state: State):
    latest_message = state["messages"][-1].content
    meat_points = state.get("meat_points", 0)
    print(f"Detect - Start Meat Points: {meat_points}")
    teach_streak = state.get("teach_streak", False)
    last_topic = state.get("last_topic", "")
    prior_messages = state["messages"][:-1]

    weights = {"low_say": 1, "mid_say_teach": 5, "high_teach": 8}
    
    last_ami = next((msg for msg in reversed(prior_messages) if msg.type == "ai"), None)
    context_teach_vibe = teach_streak or (last_ami and "teaching me" in last_ami.content.lower())
    is_comeback = "actually" in latest_message.lower() or "meant" in latest_message.lower()
    is_confirm = last_ami and "locked that" in last_ami.content.lower() and latest_message.lower() in ["yep", "yeah"]

    topic_words = last_topic.lower().split() if last_topic else []
    same_topic = any(word in latest_message.lower() for word in topic_words) or not topic_words  # Keep for later trimming

    response = llm.invoke(
        f"Prior convo: {json.dumps([msg.content for msg in prior_messages[-2:]])}.\n"
        f"Latest: '{latest_message}'.\n"
        f"Last AMI: '{last_ami.content if last_ami else ''}'.\n"
        f"Classify intent:\n"
        f"- 'low_say' (+1): Casual/vague.\n"
        f"- 'mid_say_teach' (+5): Practical hint.\n"
        f"- 'high_teach' (+8): Directive/teaching.\n"
        f"Boost mid/high if: teach vibe, 'actually', or 'yeah' adds detail.\n"
        f"Force 'confirm' if last AMI asked 'locked that' and this is 'yep'/'yeah'.\n"
        f"Return ONLY: low_say, mid_say_teach, high_teach, or confirm"
    )
    intent = response.content.strip().lower()
    print(f"LLM Raw: '{intent}'")

    new_points = weights.get(intent, 1) if intent != "confirm" else 0
    if context_teach_vibe or is_comeback:
        new_points += 1 if intent in ["mid_say_teach", "high_teach"] else 0
    total_meat = meat_points + new_points  # No reset—let it stack!

    new_teach_streak = intent in ["mid_say_teach", "high_teach", "confirm"] or context_teach_vibe
    new_topic = last_topic if intent in ["low_say", "confirm"] else latest_message

    print(f"Detect - End Meat Points: {total_meat}")
    print(f"Intent: {intent}, New Points: {new_points}, Total Meat: {total_meat}, Teach Streak: {new_teach_streak}, Topic: {new_topic}")
    return intent, total_meat, new_teach_streak, new_topic

def ami_node(state: State):
    latest_message = state["messages"][-1].content
    meat_points = state.get("meat_points", 0)
    print(f"Node - Start Meat Points: {meat_points}")
    pending_save = state.get("pending_save", {})
    last_topic = state.get("last_topic", "")

    if len(state["messages"]) == 1:
        result = {"prompt_str": "Yo, I’m AMI—whatcha vibing on today?", "meat_points": 0, "teach_streak": False, "last_topic": ""}
        print(f"Node - End Meat Points: {result['meat_points']}")
        return result

    intent, total_meat, teach_streak, new_topic = detect_intent(state)

    if pending_save and "summary" in pending_save:
        if intent == "confirm":
            action = "Updated" if pending_save.get("update", False) else "Saved"
            result = {
                "prompt_str": f"{action} '{pending_save['summary']}' (skills)—you’re stacking gold!",
                "meat_points": 5,
                "pending_save": {},
                "teach_streak": True,
                "last_topic": pending_save["text"]
            }
            print(f"Node - End Meat Points: {result['meat_points']}")
            return result
        elif "nah" in latest_message.lower() or "no" in latest_message.lower():
            result = {
                "prompt_str": f"Alright, tweak it—whatcha really teaching?",
                "meat_points": meat_points,
                "pending_save": {},
                "teach_streak": True,
                "last_topic": last_topic
            }
            print(f"Node - End Meat Points: {result['meat_points']}")
            return result
        else:
            action = "Updated" if pending_save.get("update", False) else "Saved"
            result = {
                "prompt_str": f"No tweaks? {action} '{pending_save['summary']}' (skills)—what’s next?",
                "meat_points": 5,
                "pending_save": {},
                "teach_streak": True,
                "last_topic": pending_save["text"]
            }
            print(f"Node - End Meat Points: {result['meat_points']}")
            return result

    response = f"You’re saying '{latest_message}'—whatcha getting at, bro?"
    if intent == "low_say":
        response = f"You’re saying '{latest_message}'—just vibing or got more meat coming?"
    elif intent == "mid_say_teach":
        summary = llm.invoke(f"Summarize in 10 words max: '{latest_message}'").content.strip()
        response = f"You’re saying '{summary}'—teaching me a trick there?"
    elif intent == "high_teach":
        summary = llm.invoke(f"Summarize in 10 words max: '{latest_message}'").content.strip()
        if total_meat >= 10:
            if last_topic and any(word in latest_message.lower() for word in last_topic.lower().split()):
                combined_summary = llm.invoke(
                    f"Combine '{last_topic}' and '{latest_message}' into 10 words max."
                ).content.strip()
                response = f"Update '{combined_summary}'—locked that, bro?"
                result = {
                    "prompt_str": response,
                    "meat_points": total_meat,
                    "pending_save": {"text": latest_message, "summary": combined_summary, "vibe": "skills", "update": True},
                    "teach_streak": True,
                    "last_topic": last_topic
                }
                print(f"Node - End Meat Points: {result['meat_points']}")
                return result
            else:
                response = f"You’re teaching me '{summary}'—locked that, bro?"
                result = {
                    "prompt_str": response,
                    "meat_points": total_meat,
                    "pending_save": {"text": latest_message, "summary": summary, "vibe": "skills", "update": False},
                    "teach_streak": True,
                    "last_topic": new_topic
                }
                print(f"Node - End Meat Points: {result['meat_points']}")
                return result
        else:
            response = f"You’re saying '{summary}'—got some juice, huh?"

    result = {"prompt_str": response, "meat_points": total_meat, "teach_streak": teach_streak, "last_topic": new_topic}
    print(f"Node - End Meat Points: {result['meat_points']}")
    return result

graph_builder = StateGraph(State)
graph_builder.add_node("ami", ami_node)
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def convo_stream(user_input, thread_id="test_thread"):
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = checkpointer.get(config)
    state = checkpoint["channel_values"] if checkpoint else {
        "messages": [],
        "meat_points": 0,
        "pending_save": {},
        "teach_streak": False,
        "last_topic": ""
    }
    print(f"Stream - Loaded Meat Points: {state['meat_points']}")

    new_message = HumanMessage(content=user_input)
    state["messages"] = state["messages"] + [new_message]

    new_state = convo_graph.invoke(state, config)
    response = new_state["prompt_str"]
    for chunk in textwrap.wrap(response, width=80):
        yield f"data: {json.dumps({'message': chunk})}\n\n"
    
    ai_message = AIMessage(content=response)
    new_state["messages"] = new_state["messages"] + [ai_message]
    convo_graph.update_state(config, new_state)
    print(f"Stream - Saved Meat Points: {new_state['meat_points']}")
    print(f"AMI: {response}")

if __name__ == "__main__":
    thread_id = "test_thread"
    inputs = [
        "Yo, ghosting’s been a nightmare",
        "Calling them works better than emails",
        "Yeah, wait five days",
        "Yep",
        "Texting works too",
        "Yeah, after five days",
        "Yep",
        "Billing’s a mess now",
    ]
    for user_input in inputs:
        print(f"\nUser: {user_input}")
        for chunk in convo_stream(user_input, thread_id):
            print(chunk)