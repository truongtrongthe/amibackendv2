# ami_core.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 15, 2025
# Purpose: Core Ami logic with stable stage handling, powered by Ami Blue Print 3.3 Mark 3, synced with langgraph

from BP_3_4_Mark1.utilities_3_4_mark1 import detect_intent, extract_knowledge, confirm_knowledge, recall_knowledge, LLM
import json

class AmiCore:
    def __init__(self):
        self.brain = []  # Preset Brain stub—could load later
        self.customer_data = []  # Customer history—future use
        self.user_id = "tfl"  # Default user ID
        self.presets = {
            "situation": lambda x: f"Tell me about {x} in your current setup.",
            "challenger": lambda x: f"Reframe it like this: {x}"
        }
        self.sales_stages = ["profiling", "pitched", "payment", "shipping", "done"]  # Immutable stages
        self.state = {
            "messages": [],
            "prompt_str": "",
            "convo_id": None,
            "active_terms": {},
            "pending_node": {"pieces": [], "primary_topic": "Miscellaneous"},
            "pending_knowledge": {},
            "brain": self.brain,
            "sales_stage": self.sales_stages[0],
            "last_response": ""
        }

    def load_brain_from_pinecone(self):
        # Stub for Preset Brain—could fetch from Pinecone "preset_memory" later
        pass

    def get_pickup_line(self, is_first, intent):
        intent = intent.strip('"')
        if is_first:
            return "Yo, tôi là Ami—vibe mạnh lắm nha! Thử đi, nghi ngờ kiểu gì cũng lật ngược—bạn tính sao?"
        if intent == "teaching":
            return "Kiến thức đỉnh—cho tôi thêm đi bro!"
        if intent == "casual":
            return "Chill vậy—kế tiếp là gì nào?"
        return "Cá là bạn có gì đó xịn—kể nghe coi!"

    def do(self, state=None, is_first=False, confirm_callback=None):
        state = state if state is not None else self.state
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        intent = detect_intent(state) if latest_msg else "greeting"
        response = ""

        default_state = {
            "messages": [],
            "prompt_str": "",
            "convo_id": state.get("convo_id"),
            "active_terms": {},
            "pending_node": {"pieces": [], "primary_topic": "Miscellaneous"},
            "pending_knowledge": {},
            "brain": self.brain,
            "sales_stage": self.sales_stages[0],
            "last_response": ""
        }
        state = {**default_state, **state}

        if intent == "teaching":
            knowledge = extract_knowledge(state, self.user_id)
            confirmed_node = confirm_knowledge(state, self.user_id, confirm_callback=confirm_callback)
            if confirmed_node and state["last_response"] == "yes":
                # Lean, casual confirmation—show she got it
                pieces = confirmed_node["pieces"]
                if pieces:
                    text = pieces[0]["raw_input"]  # First piece’s raw input
                    response = f"{text} -Ami nắm được rồi! Còn gì hay nữa không?"
                else:
                    response = "Ami nắm rồi, bro! Còn gì hay nữa không?"
            elif state["last_response"] == "no":
                response = state["prompt_str"]

        elif intent in ["question", "request"]:
            recall = recall_knowledge(latest_msg, self.user_id)
            response = recall["response"]
            if recall["mode"] == "Autopilot":
                state["sales_stage"] = self.sales_stages[1]

        elif intent in ["greeting", "casual"]:
            pickup = self.get_pickup_line(is_first, intent)
            casual_prompt = f"You're Ami, respond to '{latest_msg}' casually in Vietnamese with this vibe: '{pickup}'"
            response = LLM.invoke(casual_prompt).content
            print(f"DEBUG: Casual response = '{response}'")

        else:
            response = self.get_pickup_line(is_first, intent)

        state["prompt_str"] = f"Ami detected intent: '{intent}'. Here's my response: {response}"
        state["brain"] = self.brain
        self.state = state
        return state
