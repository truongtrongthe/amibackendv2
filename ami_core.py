# ami_core.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 15, 2025 (Updated for live on March 16, 2025)
# Purpose: Core Ami logic with stable stage handling, powered by Ami Blue Print 3.4 Mark 3, synced with langgraph

from utilities import detect_intent, extract_knowledge, recall_knowledge,upsert_terms,store_convo_node, LLM,index,logger,EMBEDDINGS  # Updated import
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
            "last_response": "",
            "user_id": self.user_id  # Add this
        }

    def load_brain_from_pinecone(self):
        # Stub for Preset Brain—could fetch from Pinecone "preset_memory" later
        pass

    def get_pickup_line(self, is_first, intent):
        intent = intent.strip('"')
        if is_first:
            return "Ami—mạnh lắm nha! Thử đi, nghi ngờ kiểu gì cũng lật ngược—bạn tính sao?"
        if intent == "teaching":
            return "Kiến thức đỉnh—cho tôi thêm đi bro!"
        if intent == "casual":
            return "Chill vậy—kế tiếp là gì nào?"
        return "Cá là bạn có gì đó xịn—kể nghe coi!"

    # ami_core.py
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
            
            # Upsert terms using the new utility
            pending_knowledge = state.get("pending_knowledge", {})
            if pending_knowledge:
                upsert_terms(pending_knowledge)
            
            # Store convo node
            store_convo_node(node, user_id)
            state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
            state.pop("pending_knowledge", None)
        elif response == "no":
            state["prompt_str"] = f"OK, Ami bỏ qua. Còn gì thêm cho {node['primary_topic']} không?"
            state["pending_node"] = {"pieces": [], "primary_topic": node["primary_topic"]}
        
        return node

    def do(self, state=None, is_first=False, confirm_callback=None, force_copilot=False, user_id=None):
        state = state if state is not None else self.state
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        response = ""
        user_id = user_id or state.get("user_id", "unknown")  # Fallback to state if not passed

        logger.info(f"Do called - force_copilot: {force_copilot}, user_id: '{user_id}', latest_msg: '{latest_msg}'")

        if force_copilot:
            if not latest_msg:
                response = f"{user_id.split('_')[0]}, Ami đây—cho bro cái task đi!"
            else:
                copilot_task = state.get("copilot_task", latest_msg)
                recall = recall_knowledge(latest_msg, user_id=None)  # Base convo_nodes
                
                intent_result = detect_intent(state)
                if isinstance(intent_result, tuple):
                    intent, _ = intent_result
                else:
                    intent = intent_result
                logger.info(f"Detected intent in CoPilot mode for {user_id}: '{intent}'")
                state["intent"] = intent  # Store intent in state for logging

                if intent in ["greeting", "casual"]:
                    casual_prompt = f"""You’re Ami, a confident, know-it-all coworker for AI Brain Mark 3.4 in CoPilot mode. Given:
                    - User: '{user_id.split('_')[0]}'
                    - Input: '{latest_msg}'
                    - Knowledge: {json.dumps(recall["knowledge"], ensure_ascii=False)}
                    Respond in Vietnamese with a chill, boss-like quip—keep it short, vibey, and sharp, like you’re running the show. 
                    Personalize it with the user’s name and weave in knowledge if relevant, no analysis, just a slick reply!
                    Examples:
                    - 'Yo Ami, how’s it going?' → 'Yo John, Ami vẫn chất—bro thế nào sau deal Shawn?'
                    - 'Ami, you good?' → 'Pete, tốt vl—sẵn sàng đập deal tiếp, hỏi gì thêm đi!'
                    Output MUST be a raw string, no quotes or markdown."""
                    response = LLM.invoke(casual_prompt).content.strip()
                    logger.info(f"CoPilot casual response for {user_id}: '{response}'")
                else:
                    needs_analysis = any(keyword in latest_msg.lower() for keyword in ["tại sao", "why", "sao lại", "what's", "happen", "going on"]) or len(latest_msg.split()) > 10
                    logger.info(f"Needs analysis for {user_id}: {needs_analysis} (based on keywords or length)")

                    if needs_analysis:
                        analysis_prompt = f"""You’re Ami, a confident, know-it-all coworker for AI Brain Mark 3.4. Given:
                        - User: '{user_id.split('_')[0]}'
                        - Task: '{copilot_task}'
                        - Input: '{latest_msg}'
                        - Knowledge: {json.dumps(recall["knowledge"], ensure_ascii=False)}
                        - Terms: {json.dumps(recall["terms"], ensure_ascii=False)}
                        Analyze the situation in Vietnamese—break it down into concise bullet points:
                        - Identify the most critical aspects (e.g., people, context, risks, opportunities—whatever stands out).
                        - Use a sharp, vibey tone, like you’re briefing {user_id.split('_')[0]}.
                        - Label each point dynamically with a bolded title (e.g., "**Khách hàng**", "**Vấn đề**", "**Cơ hội**") based on what’s relevant.
                        - Keep it tight, no fluff—focus on what drives the next step, weave in knowledge if it fits!
                        Format as bullet points with bold labels (e.g., "- **Label**: detail"), no markdown in the detail text.
                        Output MUST be a raw string, no quotes or markdown beyond the bullet format."""
                        analysis = LLM.invoke(analysis_prompt).content.strip()
                        logger.info(f"CoPilot analysis for {user_id}: '{analysis}'")
                    else:
                        analysis = ""
                        logger.info(f"Skipping analysis for {user_id}—straight to action")

                    action_prompt = f"""You’re Ami, a confident, know-it-all coworker for AI Brain Mark 3.4. Given:
                    - User: '{user_id.split('_')[0]}'
                    - Task: '{copilot_task}'
                    - Input: '{latest_msg}'
                    - Knowledge: {json.dumps(recall["knowledge"], ensure_ascii=False)}
                    - Terms: {json.dumps(recall["terms"], ensure_ascii=False)}
                    - Analysis (if any): '{analysis}'
                    Deliver a direct, firm, brilliant action statement in Vietnamese—act like you’ve got it all figured out for {user_id.split('_')[0]}. 
                    Blend the task, input, analysis (if provided), and knowledge into a sharp, actionable message that drives the goal forward. 
                    No hesitation, just hit hard with a clear next step—keep it short, vibey, and boss-like!
                    Output MUST be a raw string, no quotes or markdown."""
                    action = LLM.invoke(action_prompt).content.strip()
                    logger.info(f"CoPilot action for {user_id}: '{action}'")

                    if analysis:
                        response = f"{analysis}\nKết luận: **_*{action}*_**"
                    else:
                        response = action if action else f"{user_id.split('_')[0]}, Ami xử lý xong—giờ làm gì tiếp bro?"
                    if not action:
                        logger.warning(f"LLM returned empty action for {user_id}, task: {copilot_task}")

            state["copilot_task"] = state.get("copilot_task", latest_msg) if latest_msg else None
            state["prompt_str"] = f"Ami in CoPilot mode: **_{response}_**"
            state["user_id"] = user_id
            logger.info(f"CoPilot set prompt_str for {user_id}: '{state['prompt_str']}'")
        
        else:
            intent_result = detect_intent(state) if latest_msg else "greeting"
            if isinstance(intent_result, tuple):
                intent, _ = intent_result
            else:
                intent = intent_result
            logger.info(f"Detected intent for {user_id}: '{intent}'")
            # ami_core.py, in do()
            if intent == "teaching":
                knowledge = extract_knowledge(state, user_id, intent=intent)
                logger.debug(f"Raw knowledge from extract_knowledge: {json.dumps(knowledge, ensure_ascii=False)}")
                logger.debug(f"Pending knowledge before confirm: {json.dumps(state['pending_knowledge'], ensure_ascii=False)}")  # Add this
                confirmed_node = self.confirm_knowledge(state, user_id, confirm_callback=confirm_callback)
                if confirmed_node and state["last_response"] == "yes":
                    pieces = confirmed_node["pieces"]
                    # Extract terms safely
                    if not isinstance(knowledge, dict) or "terms" not in knowledge:
                        logger.error(f"Invalid knowledge structure: {knowledge}")
                        state["prompt_str"] = f"Intent: '{intent}': **_unknown, nice! Ami gặp lỗi khi xử lý kiến thức, thử lại nhé!\nMemory updated_**"
                        return state
                    terms = knowledge["terms"]
                    state["pending_knowledge"] = terms
                    
                    logger.debug(f"Raw terms from state: {json.dumps(terms, ensure_ascii=False)}")
                    
                    # Pre-process terms into a flat chunk list
                    chunk_list = []
                    for term_id, term_data in terms.items():
                        term_name = term_id.split("term_")[1].rsplit("_", 1)[0] if "term_" in term_id else term_id
                        logger.debug(f"Processing term_id: {term_id}, term_data: {json.dumps(term_data, ensure_ascii=False)}")
                        for chunk in term_data.get("knowledge", []):
                            chunk_list.append(f"- **{term_name}**: {chunk}")
                    chunk_text = "\n".join(chunk_list)
                    logger.debug(f"Pre-processed chunk_text:\n{chunk_text}")
                    
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
                                Rules:
                                - Include EVERY chunk from 'All Knowledge Chunks'—one bullet per line, NO exceptions.
                                - Do NOT summarize, combine, skip, or alter the chunks—use them EXACTLY as provided.
                                - Example: Given 'All Knowledge Chunks: - **HITO Cốm**: giúp tăng chiều cao\n- **canxi hữu cơ**: xịn' → 'John, nice! Ami hiểu thế này nhé:\\n- **HITO Cốm**: giúp tăng chiều cao\\n- **canxi hữu cơ**: xịn\\nHọc được món xịn thế này, bro quá chất! Memory updated'
                                Output MUST be a raw string with \\n for newlines, no quotes or extra markdown beyond bullets."""
                    logger.debug(f"Executing prompt:\n{prompt}")
                    response = LLM.invoke(prompt).content
                    logger.debug(f"LLM response:\n{response}")
                    if not all(chunk.strip() in response for chunk in chunk_list):
                        logger.warning("LLM failed to list all chunks, using fallback")
                        response = f"{user_id.split('_')[0]}, nice! Ami hiểu thế này nhé:\n{chunk_text}\nHọc được món xịn thế này, bro quá chất! Memory updated"
                    state["prompt_str"] = f"Intent: '{intent}': **_{response}_**"
                return state

            elif intent in ["question", "request"]:
                recall = recall_knowledge(latest_msg, user_id)
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
                    using exact key phrases (e.g., "ổn định hấp thụ xương"). For QUESTIONS, drop clear answers with a gain hook 
                    (e.g., "xương chắc hơn") and nudge for more. For REQUESTS, push a confident close with next steps 
                    (e.g., "làm luôn nè") and max gain (e.g., "con bạn cao vượt trội"). 
                    Predict objections (e.g., "quá tuổi?", "đắt không?") and flip ‘em—keep it short, vibey, and actionable with a "cực chất" edge!
                    Examples:
                    - Question 'HITO Cốm có gì hay?' → 'John, HITO Cốm chất lắm—tăng hấp thụ canxi, xương chắc hơn, con bạn cao vượt trội! Còn thắc mắc gì nữa không nè?'
                    - Request 'Mua HITO Cốm đi' → 'Pete, HITO Cốm đây—bổ sung canxi đỉnh cao, rẻ mà chất! Gửi địa chỉ, chọn combo 1-3, chuyển tiền @VCB Germany—làm luôn nè!'
                    Output MUST be a raw string, no quotes or markdown."""
                    response = LLM.invoke(prompt).content.strip('"')
                    if recall["mode"] == "Autopilot" and intent == "request":
                        state["sales_stage"] = self.sales_stages[1]  # 'pitched'

            elif intent in ["greeting", "casual"]:
                pickup = self.get_pickup_line(is_first, intent)
                casual_prompt = f"You're Ami, respond to '{latest_msg}' casually in Vietnamese with this vibe: '{pickup}' for {user_id.split('_')[0]}"
                response = LLM.invoke(casual_prompt).content
                print(f"DEBUG: Casual response for {user_id}: '{response}'")
            
            else:
                response = self.get_pickup_line(is_first, intent)
                print(f"DEBUG: Casual response for {user_id}: '{response}'")

            state["prompt_str"] = f"Intent: '{intent}': **_{response}_**"
            logger.info(f"Intent set prompt_str for {user_id}: '{state['prompt_str']}'")

        state["brain"] = self.brain
        state["user_id"] = user_id
        self.state = state
        logger.info(f"Returning state with prompt_str for {user_id}: '{state['prompt_str']}'")
        return state