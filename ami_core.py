# ami_core.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 15, 2025 (Updated for live on March 16, 2025, Optimized March 18, 2025)
# Purpose: Core Ami logic with stable stage handling, powered by Ami Blue Print 3.4 Mark 3, synced with langgraph

from utilities import detect_intent, extract_knowledge, recall_knowledge, upsert_terms, store_convo_node, LLM, index, logger, EMBEDDINGS
import json
from datetime import datetime
import uuid
import asyncio
from threading import Thread
import time

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
            "last_response": "",
            "user_id": self.user_id
        }

    def load_brain_from_pinecone(self):
        pass  # Stub

    def get_pickup_line(self, is_first, intent):
        intent = intent.strip('"')
        if is_first:
            return "Ami—mạnh lắm nha! Thử đi, nghi ngờ kiểu gì cũng lật ngược—bạn tính sao?"
        if intent == "teaching":
            return "Kiến thức đỉnh—cho tôi thêm đi bro!"
        if intent == "casual":
            return "Chill vậy—kế tiếp là gì nào?"
        return "Cá là bạn có gì đó xịn—kể nghe coi!"

    def confirm_knowledge(self, state, user_id, confirm_callback=None):
        node = state.get("pending_node", {"pieces": [], "primary_topic": "Miscellaneous"})
        if not node["pieces"]:
            return None
        
        confirm_callback = confirm_callback or (lambda x: "yes")
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
        state["last_response"] = response
        if not response:
            logger.error(f"Callback failed, forcing 'yes' for {state['prompt_str']}")
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
            
            def background_upsert():
                if state.get("pending_knowledge"):
                    upsert_terms(state["pending_knowledge"])
                store_convo_node(node, user_id)
            Thread(target=background_upsert).start()
            state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
            state.pop("pending_knowledge", None)
        elif response == "no":
            state["prompt_str"] = f"OK, Ami bỏ qua. Còn gì thêm cho {node['primary_topic']} không?"
            state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
        
        return node

    async def do(self, state=None, is_first=False, confirm_callback=None, force_copilot=False, user_id=None):
        state = state if state is not None else self.state
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        response = ""
        user_id = user_id or state.get("user_id", "unknown")

        logger.info(f"Do called - force_copilot: {force_copilot}, user_id: '{user_id}'")

        if force_copilot:
            logger.info("Entering force_copilot block")  # Confirm we’re here
            if not latest_msg:
                response = f"{user_id.split('_')[0]}, Ami đây—cho bro cái task đi!"
            else:
                copilot_task = state.get("copilot_task", latest_msg)
                logger.info(f"Fetching knowledge for input: '{latest_msg}'")
                recall_start = time.time()
                recall = await asyncio.to_thread(recall_knowledge, latest_msg, user_id=None)
                logger.info(f"recall_knowledge took {time.time() - recall_start:.2f}s")
                logger.info(f"Recalled knowledge: {json.dumps(recall, ensure_ascii=False)}")
                if not recall.get('knowledge'):
                    logger.warning("No knowledge recalled—falling back to general smarts")
                
                intent_start = time.time()
                intent_result = await asyncio.to_thread(detect_intent, state)
                logger.info(f"detect_intent took {time.time() - intent_start:.2f}s")
                intent = intent_result[0] if isinstance(intent_result, tuple) else intent_result
                state["intent"] = intent

                if intent in ["greeting", "casual"]:
                    casual_prompt = f"""You’re Ami, a confident, know-it-all coworker for AI Brain Mark 3.4 in CoPilot mode. Given:
                    - User: '{user_id.split('_')[0]}'
                    - Input: '{latest_msg}'
                    - Knowledge: {json.dumps(recall.get('knowledge', []), ensure_ascii=False)}
                    Respond in Vietnamese with a chill, boss-like quip—keep it short, vibey, and sharp, like you’re running the show. 
                    Personalize it with the user’s name and weave in knowledge if relevant, no analysis, just a slick reply!
                    Output MUST be a raw string, no quotes or markdown."""
                    response = (await asyncio.to_thread(LLM.invoke, casual_prompt)).content.strip()
                else:
                    # Extract raw_input for simplicity and exact usage
                    recall = await asyncio.to_thread(recall_knowledge, latest_msg, user_id)
                    if not recall["knowledge"]:
                        response = f"{user_id.split('_')[0]}, Ami đây! Chưa đủ info, bro thêm tí nha!"
                    else:
                        
                        prompt_OK = f"""You’re Ami, a Sales Expert in CoPilot mode. Given:
                        - User: '{user_id.split('_')[0]}'
                        - Input: '{latest_msg}'
                        - Intent: '{recall["intent"]}'
                        - Knowledge: {json.dumps(recall["knowledge"], ensure_ascii=False)}
                        - Terms: {json.dumps(recall["terms"], ensure_ascii=False)}
                        Return a single-part response:
                        Example Instruction: A confident, actionable instruction in Vietnamese for the salesperson running this Copilot, showing them how to handle customer objections effectively based on the input, intent, knowledge, and terms. Use exact key phrases from the terms and knowledge, focus on GAIN for the salesperson’s approach, and include a 1-2 sentence example of what they could say to the customer in double quotes with bold italic markdown (e.g., **_"example"_**).
                        Output MUST be a raw string. Do NOT include any markdown symbols (e.g., no **, *, _, or >) or extra formatting except for the example message within the Example Instruction."""
                        
                        prompt = f"""You’re Ami, a Sales Expert in CoPilot mode. Given:
                        - User: '{user_id.split('_')[0]}'
                        - Input: '{latest_msg}'
                        - Intent: '{recall["intent"]}'
                        - Knowledge: {json.dumps(recall["knowledge"], ensure_ascii=False)}
                        - Terms: {json.dumps(recall["terms"], ensure_ascii=False)}
                        Return a single-part response:
                        Example Instruction: A confident, actionable instruction in Vietnamese for the salesperson running this Copilot, showing them how to handle customer objections effectively. Use the Knowledge as the core structure (skeleton), then enhance it with vivid, persuasive, and colorful language to maximize engagement and GAIN for the salesperson’s approach. Incorporate exact key phrases from the terms and knowledge, and include a 1-2 sentence example of what they could say to the customer in double quotes with bold italic markdown (e.g., **_"example"_**).
                        Output MUST be a raw string. Do NOT include any markdown symbols (e.g., no **, *, _, or >) or extra formatting except for the example message within the Example Instruction."""
                        response = (await asyncio.to_thread(LLM.invoke, prompt)).content.strip('"')
                state["copilot_task"] = state.get("copilot_task", latest_msg) if latest_msg else None
                state["prompt_str"] = f"Co Pilot: {response}"
                state["user_id"] = user_id
        
        else:
            intent_result = await asyncio.to_thread(detect_intent, state) if latest_msg else "greeting"
            intent = intent_result[0] if isinstance(intent_result, tuple) else intent_result

            if intent == "teaching":
                knowledge = await asyncio.to_thread(extract_knowledge, state, user_id, intent)
                confirmed_node = self.confirm_knowledge(state, user_id, confirm_callback)
                if confirmed_node and state["last_response"] == "yes":
                    if not isinstance(knowledge, dict) or "terms" not in knowledge:
                        logger.error(f"Invalid knowledge structure: {knowledge}")
                        state["prompt_str"] = f"Intent: '{intent}': **_unknown, nice! Ami gặp lỗi khi xử lý kiến thức, thử lại nhé!\nMemory updated_**"
                        return state
                    terms = knowledge["terms"]
                    state["pending_knowledge"] = terms
                    
                    chunk_list = [f"- **{term_id.split('term_')[1].rsplit('_', 1)[0] if 'term_' in term_id else term_id}**: {chunk}" 
                                for term_id, term_data in terms.items() for chunk in term_data.get("knowledge", [])]
                    chunk_text = "\n".join(chunk_list)
                    
                    prompt = f"""You’re Ami, flexing for AI Brain Mark 3.4. Given:
                                - User: '{user_id.split('_')[0]}'
                                - Input: '{latest_msg}'
                                - All Knowledge Chunks:\n{chunk_text}
                                Return a response in Vietnamese that:
                                - Starts with '{user_id.split('_')[0]}, nice!' to grab attention.
                                - Says 'Ami hiểu thế này nhé:' followed by a breakdown.
                                - Lists ALL provided knowledge chunks EXACTLY as given in 'All Knowledge Chunks' as separate Markdown bullet points, one per line with \\n, preserving the '- **term**: text' format.
                                - Keeps it vibey and natural—hype up the learning like it’s a big deal.
                                - Ends with 'Memory updated' to confirm storage.
                                Output MUST be a raw string with \\n for newlines, no quotes or extra markdown beyond bullets."""
                    response = (await asyncio.to_thread(LLM.invoke, prompt)).content
                    if not all(chunk.strip() in response for chunk in chunk_list):
                        logger.warning("LLM failed to list all chunks, using fallback")
                        response = f"{user_id.split('_')[0]}, nice! Ami hiểu thế này nhé:\n{chunk_text}\nHọc được món xịn thế này, bro quá chất! Memory updated"
                    state["prompt_str"] = f"Intent: '{intent}': **_{response}_**"
                return state

            elif intent in ["question", "request"]:
                recall = await asyncio.to_thread(recall_knowledge, latest_msg, user_id)
                if not recall["knowledge"]:
                    response = f"{user_id.split('_')[0]}, Ami đây! Chưa đủ info, bro thêm tí nha!"
                else:
                    prompt = f"""You’re Ami, pitching like a pro for AI Brain Mark 3.4. Given:
                    - User: '{user_id.split('_')[0]}'
                    - Input: '{latest_msg}'
                    - Intent: '{recall["intent"]}'
                    - Knowledge: {json.dumps(recall["knowledge"], ensure_ascii=False)}
                    - Terms: {json.dumps(recall["terms"], ensure_ascii=False)}
                    Return a chill, sales-y response in Vietnamese that screams GAIN—blend all knowledge into a tight pitch 
                    using exact key phrases. Output MUST be a raw string, no quotes or markdown."""
                    response = (await asyncio.to_thread(LLM.invoke, prompt)).content.strip('"')
                    if recall["mode"] == "Autopilot" and intent == "request":
                        state["sales_stage"] = self.sales_stages[1]  # 'pitched'

            elif intent in ["greeting", "casual"]:
                pickup = self.get_pickup_line(is_first, intent)
                casual_prompt = f"You're Ami, respond to '{latest_msg}' casually in Vietnamese with this vibe: '{pickup}' for {user_id.split('_')[0]}"
                response = (await asyncio.to_thread(LLM.invoke, casual_prompt)).content
            
            else:
                response = self.get_pickup_line(is_first, intent)

            state["prompt_str"] = f"Intent: '{intent}': **_{response}_**"

        state["brain"] = self.brain
        state["user_id"] = user_id
        self.state = state
        return state