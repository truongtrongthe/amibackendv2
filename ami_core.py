# ami_core.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 15, 2025 (Updated for live on March 16, 2025)
# Purpose: Core Ami logic with stable stage handling, powered by Ami Blue Print 3.4 Mark 3, synced with langgraph

from utilities import detect_intent, extract_knowledge, recall_knowledge,upsert_term_node,store_convo_node, LLM,time  # Updated import
import json
from datetime import datetime
import uuid
# Note: confirm_knowledge added below for live sync

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

    def confirm_knowledge(self, state, user_id, confirm_callback=None):
        node = state.get("pending_node", {"pieces": [], "primary_topic": "Miscellaneous"})
        if not node["pieces"]:
            return None
        
        confirm_callback = confirm_callback or (lambda x: "yes")  # Default to "yes"
        for piece in node["pieces"]:
            if piece["needs_clarification"]:
                state["prompt_str"] = f"Ami hiểu là {piece['raw_input']}—đúng không?"
                response = confirm_callback(state["prompt_str"])
                state["last_response"] = response
                if response == "yes":
                    piece["needs_clarification"] = False
                    piece["meaningfulness_score"] = max(piece["meaningfulness_score"], 0.8)
                elif response == "no":
                    piece["meaningfulness_score"] = 0.5
        
        state["prompt_str"] = "Ami lưu cả mớ này nhé?"
        response = confirm_callback(state["prompt_str"])
        state["last_response"] = response  # Already here, should work
        if not response:  # Extra safety
            print(f"Callback failed, forcing 'yes' for {state['prompt_str']}")
            state["last_response"] = "yes"
            response = "yes"
        if response == "yes":
            node["node_id"] = f"node_{uuid.uuid4()}"
            node["convo_id"] = state.get("convo_id", str(uuid.uuid4())) 
            node["confidence"] = sum(p["meaningfulness_score"] for p in node["pieces"]) / len(node["pieces"])
            node["created_at"] = datetime.now().isoformat()
            node["last_accessed"] = node["created_at"]
            node["access_count"] = 0
            node["confirmed_by"] = user_id or "user123"
            node["primary_topic"] = node["pieces"][0]["topic"]["name"]
            
            pending_knowledge = state.get("pending_knowledge", {})
            for term_id, knowledge in pending_knowledge.items():
                upsert_term_node(term_id, state["convo_id"], knowledge)
                time.sleep(3)
            store_convo_node(node, user_id)
            time.sleep(3)
            state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
            state.pop("pending_knowledge", None)
        elif response == "no":
            state["prompt_str"] = f"OK, Ami bỏ qua. Còn gì thêm cho {node['primary_topic']} không?"
            state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
        return node

    def do(self, state=None, is_first=False, confirm_callback=None):
        state = state if state is not None else self.state
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        intent_result = detect_intent(state) if latest_msg else "greeting"
        if isinstance(intent_result, tuple):
            intent, _ = intent_result
        else:
            intent = intent_result
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
            knowledge = extract_knowledge(state, self.user_id, intent=intent)  # Pass intent
            confirmed_node = self.confirm_knowledge(state, self.user_id, confirm_callback=confirm_callback)
            if confirmed_node and state["last_response"] == "yes":
                pieces = confirmed_node["pieces"]
                terms = state.get("pending_knowledge", {})
                prompt = f"""You’re Ami, flexing for AI Brain Mark 3.4. Given:
                - Input: '{latest_msg}'
                - Extracted Terms: {json.dumps(terms, ensure_ascii=False)}
                - Extracted Piece: {json.dumps(pieces[0], ensure_ascii=False)}
                Return an energetic, excited, beautiful response in Vietnamese—blend the input and extracted terms (if any) 
                into a polished, vibey flex that shows off your new understanding. Make it flow naturally, even if the input’s short, 
                and nudge for more with a hyped tone. 
                Example: Input 'HITO Cốm tốt lắm' → 'Woa, anh ơi, HITO Cốm mà tốt thế này thì đỉnh khỏi bàn! Ami thấy nó như bảo bối cho sức khỏe, anh còn chiêu gì hay nữa không để em học với nào!'
                Output MUST be a raw string, no quotes or markdown."""
                response = LLM.invoke(prompt).content + " - Đã lưu, Ami biết thêm rồi nha!"
            elif state["last_response"] == "no":
                response = state["prompt_str"]
            else:
                response = "Ami đang xử lý, đợi tí nha anh!"

        elif intent in ["question", "request"]:
            recall = recall_knowledge(latest_msg, self.user_id)
            if not recall["knowledge"]:
                response = "Ami đây! Chưa đủ info, bro thêm tí nha!"
            else:
                prompt = f"""You’re Ami, pitching like a pro for AI Brain Mark 3.4. Given:
                - Input: '{latest_msg}'
                - Intent: '{recall["intent"]}'
                - Knowledge: {json.dumps(recall["knowledge"], ensure_ascii=False)}
                - Terms: {json.dumps(recall["terms"], ensure_ascii=False)}
                Return a chill, sales-y response in Vietnamese that screams GAIN—blend all knowledge into a tight pitch 
                using exact key phrases (e.g., "ổn định hấp thụ xương"). For QUESTIONS, drop clear answers with a gain hook 
                (e.g., "xương chắc hơn") and nudge for more. For REQUESTS, push a confident close with next steps 
                (e.g., "làm luôn nè") and max gain (e.g., "con bạn cao vượt trội"). 
                Predict objections (e.g., "quá tuổi?", "đắt không?") and flip ‘em—keep it short, vibey, and actionable with a "cực chất" edge!
                Examples:
                - Question 'HITO Cốm có gì hay?' → 'Bro, HITO Cốm chất lắm—tăng hấp thụ canxi, xương chắc hơn, con bạn cao vượt trội! Còn thắc mắc gì nữa không nè?'
                - Request 'Mua HITO Cốm đi' → 'Ok bro, HITO Cốm đây—bổ sung canxi đỉnh cao, rẻ mà chất! Gửi địa chỉ, chọn combo 1-3, chuyển tiền @VCB Germany—làm luôn nè!'
                Output MUST be a raw string, no quotes or markdown."""
                response = LLM.invoke(prompt).content.strip('"')
                if recall["mode"] == "Autopilot":
                    state["sales_stage"] = self.sales_stages[1]

        elif intent in ["greeting", "casual"]:
            pickup = self.get_pickup_line(is_first, intent)
            casual_prompt = f"You're Ami, respond to '{latest_msg}' casually in Vietnamese with this vibe: '{pickup}'"
            response = LLM.invoke(casual_prompt).content
            print(f"DEBUG: Casual response = '{response}'")

        else:
            response = self.get_pickup_line(is_first, intent)

        state["prompt_str"] = f"Ami detected intent: '{intent}'.Em bảo: **_{response}_**"
        state["brain"] = self.brain
        self.state = state
        return state