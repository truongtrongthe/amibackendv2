# ami_core.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 14, 2025
# Purpose: Core Ami logic with stable stage handling

from utilities import detect_intent, extract_knowledge, store_in_pinecone, recall_knowledge,LLM

class AmiCore:
    def __init__(self):
        self.brain = []
        self.customer_data = []
        self.user_id = "tfl"
        self.presets = {
            "situation": lambda x: f"Tell me about {x} in your current setup.",
            "challenger": lambda x: f"Reframe it like this: {x}"
        }
        self.sales_stages = ["profiling", "pitched", "payment", "shipping", "done"]  # Immutable

    def load_brain_from_pinecone(self):
        # Stub - could load Preset Brain later
        pass

    def get_pickup_line(self, is_first, intent):
        if is_first:
            return "Yo, I’m Ami—vibing hard! Trials flip skeptics—what’s your move?"
        if intent == '"teaching"':
            return "Dope info—gimme more!"
        if intent == '"casual"':
            return "Chill vibes—what’s up next?"
        return "Bet you’ve got something slick—spill it!"

    def do(self, state, is_first):
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        intent = detect_intent(state) if latest_msg else '"greeting"'
        response = ""

        if intent in ['"greeting"', '"casual"']:
            casual_prompt = f"You're Ami, respond to '{latest_msg}' casually"  # Fixed typos, used latest_msg
            response = LLM.invoke(casual_prompt).content
            print(f"DEBUG: Casual response = '{response}'")  # Check what’s coming out
        elif intent == '"teaching"':
            teach_prompt = f"You're Ami, teach something cool about '{latest_msg}'"
            response = LLM.invoke(teach_prompt).content

        state["prompt_str"] = f"Ami detected intent: {intent}. Here's my response: {response}"
        state["sales_stage"] = self.sales_stages[0]
        state["brain"] = self.brain
        return state