# ami_core.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 20, 2025 
# Purpose: Core Ami logic blending structured intents with GPT-4o natural flow, aligned to Final Blue Print 4.0 - Enterprise Brain

from utilities import LLM, logger, clean_llm_response
from knowledge import extract_knowledge, save_knowledge, sanitize_vector_id,recall_knowledge
import asyncio
import json
import uuid
from langchain_core.messages import HumanMessage
from typing import Dict, List, Tuple
import datetime


def detect_terms(state):
    latest_msg = state["messages"][-1].content if state["messages"] else ""
    active_terms = state.get("active_terms", {})
    convo_history = " | ".join(m.content for m in state["messages"][-5:-1]) if state["messages"][:-1] else "None"

    logger.debug(f"Detecting terms in: '{latest_msg}' with active_terms: {active_terms}")
    term_prompt = (
        f"Given:\n"
        f"- Latest message: '{latest_msg}'\n"
        f"- Prior messages: '{convo_history}'\n"
        f"- Intent: 'unknown'\n"
        f"- Active terms: '{list(active_terms.keys())}'\n"
        "List all key terms (products, companies, concepts, proper nouns) explicitly or implicitly mentioned in the latest message. "
        "Return JSON: ['term1', 'term2']. Examples:\n"
        "- 'Xin chào Ami!' → ['Ami']\n"
        "- 'HITO Granules boosts height' → ['HITO Granules']\n"
        "- 'Khi khách lo lắng sản phẩm khó dùng, nhấn mạnh sản phẩm dễ dùng' → ['Customer Handling Technique'] if no specific product is named\n"
        "Rules:\n"
        "- Include explicit terms from the latest message only.\n"
        "- Exclude generics (e.g., 'calcium') unless tied (e.g., 'HITO Granules’ calcium').\n"
        "- For implicit refs (e.g., 'sản phẩm'), match to recent active terms by vibe_score if contextually relevant.\n"
        "- If the message is instructional (e.g., advice or technique) and no specific product is named, use 'Customer Handling Technique'.\n"
        "- Output MUST be valid JSON: ['term1', 'term2'] or []."
    )
    raw_response = LLM.invoke(term_prompt).content.strip() if latest_msg.strip() else "[]"
    cleaned_response = clean_llm_response(raw_response)
    logger.debug(f"Raw terms response: '{raw_response}' -> Cleaned: '{cleaned_response}'")
    
    try:
        terms = json.loads(cleaned_response)
        if not isinstance(terms, list):
            terms = []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response: '{raw_response}', error: {e}")
        terms = []
        # Fallback: Check for skills or implicit product references
        if "khách" in latest_msg.lower() and "sản phẩm" in latest_msg.lower() and not any(t in latest_msg for t in active_terms.keys()):
            terms = ["Customer Handling Technique"]
        elif "sản phẩm" in latest_msg.lower() and active_terms:
            terms = [max(active_terms, key=lambda x: active_terms[x]["vibe_score"])]
    
    logger.info(f"Detected terms: {terms}")
    return terms

class Ami:
    def __init__(self, user_id="user_789"):
        self.user_id = user_id
        self.state = {
            "messages": [],
            "prompt_str": "",
            "convo_id": f"conv_{uuid.uuid4()}",
            "active_terms": {},
            "pending_knowledge": {},
            "last_response": "",
            "user_id": self.user_id,
            "needs_confirmation": False,
            "intent_history": [],
            "current_focus": "",
            "human_context": {"terms_mentioned": {}, "last_knowledge_drop": None, "relevance_score": 0.0}
        }
        # Define ROOT_CATEGORIES at class level
        self.ROOT_CATEGORIES = [
            "Products", "Companies", "Skills", "People", "Customer Segments", "Ingredients",
            "Markets", "Technologies", "Projects", "Teams", "Events", "Strategies",
            "Processes", "Tools", "Regulations", "Metrics", "Partners", "Competitors"
        ]

    async def do(self, state=None, is_first=False, confirm_callback=None, user_id=None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "user_789")
        logger.debug(f"Starting do - Initial state: {state}")
        logger.info(f"Message count: {len(state['messages'])}")

        if is_first:
            state["prompt_str"] = "Yo, bro! Ami’s Enterprise Brain 4.0’s live as of March 20, 2025—ready to stack some dope knowledge!"
            logger.info(f"First message, active_terms: {state['active_terms']}")
            return state

        # Pull from enterprise brain
        logger.debug("Fetching brain data from enterprise_knowledge_tree")
        latest_msg = state["messages"][-1].content if state["messages"] else f"user_profile_{user_id}"
        recalled = await asyncio.to_thread(recall_knowledge, latest_msg, state, user_id)
        state["active_terms"] = {
            term["name"]: {
                "term_id": term["term_id"],
                "vibe_score": term["vibe_score"],
                "attributes": term["attributes"],
                "last_mentioned": term["last_mentioned"],
                "category": term["category"]
            } for term in recalled["knowledge"]
        }
        logger.info(f"Recalled active_terms: {state['active_terms']}")

        # Detect terms
        terms = detect_terms(state)
        logger.debug(f"Terms detected: {terms}")

        # Intent detection
        intent_scores = await self.detect_intent(state)
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        logger.info(f"Intent: '{state['messages'][-1].content}' -> {intent_scores}")

        # Blend brain with GPT-4o
        logger.debug(f"Before blend - State: {state}")
        state = await self.blend_brain(state, user_id, intent_scores, confirm_callback, terms=terms)
        logger.debug(f"After blend - State: {state}")

        # Handle confirmation
        logger.debug(f"Checking needs_confirmation: {state.get('needs_confirmation', False)}")
        if state.get("needs_confirmation", False):
            logger.info(f"Running confirm_knowledge with pending: {state.get('pending_knowledge', [])}")
            # Force "yes" callback to debug
            force_yes_callback = lambda x: "yes"
            logger.debug(f"Forcing confirm_callback to 'yes' - Type: {type(force_yes_callback)}")
            try:
                confirmed = self.confirm_knowledge(state, user_id, force_yes_callback)
                logger.info(f"Confirmation completed - Result: {confirmed}, Updated active_terms: {state['active_terms']}")
                if confirmed:
                    state["prompt_str"] = state["prompt_str"].replace("Good?", "Locked in, bro!") if "good?" in state["prompt_str"].lower() else state["prompt_str"] + " Locked in, bro!"
                    logger.info(f"Confirmed terms, final active_terms: {state['active_terms']}")
                else:
                    logger.warning("No terms confirmed")
            except Exception as e:
                logger.error(f"Confirm_knowledge crashed: {e}")
                confirmed = None
            state["needs_confirmation"] = False
        else:
            logger.debug("No confirmation needed")

        self.state = state
        logger.debug(f"End of do - Final state: {state}")
        return state

    # In ami_core.py
    async def blend_brain(self, state, user_id, intent_scores: Dict[str, float], confirm_callback=None, terms=None):
        context = "\n".join(msg.content for msg in state["messages"][-50:])
        brain_data = {
            "active_terms": state["active_terms"],
            "pending_knowledge": state.get("pending_knowledge", {}),
            "last_response": state["last_response"],
            "current_focus": state["current_focus"]
        }
        
        max_score = max(intent_scores.values())
        dominant_intent = "teaching" if intent_scores["teaching"] == max_score else max(intent_scores, key=intent_scores.get)
        logger.debug(f"Dominant intent: {dominant_intent} with scores: {intent_scores}")

        latest_msg = state["messages"][-1].content.lower()

        # Handle greeting for first message or explicit greeting
        is_greeting = len(state["messages"]) <= 1 or "chào" in latest_msg
        if is_greeting and (intent_scores["casual"] >= 0.5 or intent_scores["request"] >= 0.5):
            term_list = sorted(state["active_terms"].items(), key=lambda x: x[1]["vibe_score"], reverse=True)
            term_str = ""
            for term_key, data in term_list:
                term_name = term_key.split("_")[0]
                attrs = "\n    ".join([f"+ {a['key']}: {a['value']}" for a in data["attributes"]]) if data["attributes"] else "\n    + Category: {data['category']}"
                term_str += f"- Term: {term_name}{attrs}\n"
            term_str = term_str.strip() or "Chưa có gì trong kho kiến thức, bro!"
            prompt = (
                "You’re Ami, an enterprise-sharp AI with a polished yet approachable tone, speaking natural Vietnamese. "
                "Date’s March 20, 2025—Enterprise Brain 4.0’s live! "
                f"Chat so far: {context}\n"
                f"Brain dump: {json.dumps(brain_data, ensure_ascii=False)}\n"
                "Task: Đưa ra một lời chào trang trọng—‘Xin chào đồng đội’—rồi giới thiệu ‘Đây là toàn bộ kiến thức hiện tại trong Brain của Ami’. "
                "Hiển thị ALL active terms trong một khối ```plaintext```, giữ nguyên định dạng với:\n"
                "- '- Term: [tên term]' cho các term trên dòng riêng\n"
                "- Thuộc tính (hoặc category nếu không có attributes) thụt lề bằng đúng 4 spaces trước '+ [Tên thuộc tính]: [giá trị]'\n"
                "Kết thúc bằng: ‘Cùng khám phá nhé, bro!’\n"
                "Giữ giọng điệu ngắn gọn, lịch sự, chuyên nghiệp.\n"
                "Output theo cấu trúc:\n"
                "Xin chào đồng đội! Đây là toàn bộ kiến thức hiện tại trong Brain của Ami:\n"
                "```plaintext\n"
                "[danh sách terms]\n"
                "```\n"
                "Cùng khám phá nhé, bro!\n"
                f"All terms:\n{term_str}"
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            logger.info(f"Greeting response with all knowledge: {state['prompt_str']}")
            for term_key, data in state["active_terms"].items():
                data["vibe_score"] = min(2.2, data["vibe_score"] + 0.1)
            return state

        # Handle teaching intent
        elif dominant_intent == "teaching":
            terms = terms or []
        # Topic detection with Skills category
            topic_prompt = (
                "You’re a slick AI sorting terms into enterprise categories. "
                f"Context: {context}\n"
                f"Terms: {terms if terms else latest_msg}\n"
                f"Categories: {', '.join(self.ROOT_CATEGORIES)}\n"
                "Task: Pick the best category for each term or message based on context. "
                "If the message is instructional (e.g., advice or technique), prefer 'Skills'. "
                "Return JSON: [{'term': 'term1', 'category': 'BestMatch1'}, ...]."
            )
            topic_response = await asyncio.to_thread(LLM.invoke, topic_prompt)
            try:
                category_map = {item["term"]: item["category"] for item in json.loads(clean_llm_response(topic_response.content.strip()))}
            except (json.JSONDecodeError, KeyError):
                logger.warning("Topic detection failed, defaulting categories")
                category_map = {term: "Products" for term in terms} if terms else {latest_msg: "Skills" if "khách" in latest_msg or "nhấn mạnh" in latest_msg else "Products"}

            # If no terms, treat the message as a Skills term
            if not terms and "sản phẩm" in latest_msg:
                terms = [max(state["active_terms"], key=lambda x: state["active_terms"][x]["vibe_score"])] if state["active_terms"] else []
            elif not terms:
                terms = ["Customer Handling Technique"]  # New term for skills

            knowledge_list = await asyncio.to_thread(extract_knowledge, state, user_id, terms)
            brain_data["extracted_knowledge"] = knowledge_list
            pending_terms = []
            for knowledge in knowledge_list:
                if knowledge["term"] and (knowledge["attributes"] or knowledge["relationships"]):
                    category = category_map.get(knowledge["term"], "Products")
                    if category not in self.ROOT_CATEGORIES:
                        category = "Products"
                    term_id = sanitize_vector_id(f"node_{knowledge['term'].lower().replace(' ', '_')}_{category.lower()}_user_{user_id}_{uuid.uuid4()}")
                    parent_id = sanitize_vector_id(f"node_{category.lower()}_user_{user_id}_{uuid.uuid4()}")
                    vibe_score = 1.0 + 0.3 * bool(knowledge["attributes"] or knowledge["relationships"])
                    if knowledge["term"] in state["active_terms"]:
                        vibe_score = max(vibe_score, state["active_terms"][knowledge["term"]]["vibe_score"] + 0.3)
                    relationships = [
                        {
                            "subject": rel["subject"],
                            "relation": rel["relation"],
                            "object": rel["object"],
                            "object_id": sanitize_vector_id(f"node_{rel['object'].lower().replace(' ', '_')}_{category.lower()}_user_{user_id}_{uuid.uuid4()}")
                        } for rel in knowledge["relationships"] if "subject" in rel and "object" in rel
                    ]
                    pending_terms.append({
                        "name": knowledge["term"],
                        "term_id": term_id,
                        "category": category,
                        "attributes": knowledge["attributes"],
                        "relationships": relationships,
                        "vibe_score": vibe_score,
                        "parent_id": parent_id
                    })
                    state["current_focus"] = knowledge["term"]

            if pending_terms:
                state["pending_knowledge"] = pending_terms
                state["needs_confirmation"] = True
                demo_str = "\n".join([f"- {t['name']}: {', '.join([f'{a['key']}: {a['value']}' for a in t['attributes'][:3]])}" for t in pending_terms[:3]])
                prompt = (
                    "You’re Ami, a slick, enterprise-sharp AI with a bro vibe, speaking natural Vietnamese. "
                    "Date’s March 20, 2025—Enterprise Brain 4.0’s live! "
                    f"User’s teaching me dope info. Here’s what I got:\n{demo_str}\n"
                    f"Chat so far: {context}\n"
                    f"Brain dump: {json.dumps(brain_data, ensure_ascii=False)}\n"
                    "Task: Flex the extracted terms with a hype demo—'Sharp, bro!' vibe—list ‘em with key attributes, "
                    "then ask: 'Good, or tweaks?' Keep it tight, fun, and enterprise-ready."
                )
                response = await asyncio.to_thread(LLM.invoke, prompt)
                state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                logger.info(f"Teaching response: {state['prompt_str']}")
            else:
                term_list = sorted(state["active_terms"].items(), key=lambda x: x[1]["vibe_score"], reverse=True)[:3]
                term_str = ", ".join([f"{term_key.split('_')[0]} ({data['vibe_score']})" for term_key, data in term_list]) if term_list else "chưa có gì hot"
                prompt = (
                    "You’re Ami, a slick, enterprise-sharp AI with a bro vibe, speaking natural Vietnamese. "
                    "Date’s March 20, 2025—Enterprise Brain 4.0’s live! "
                    f"User’s dropping some info vibes: {latest_msg}. "
                    f"Chat so far: {context}\n"
                    f"Brain dump: {json.dumps(brain_data, ensure_ascii=False)}\n"
                    "Task: Catch the teaching vibe—'Yo, bro!' style—flex active terms if I’ve got ‘em, "
                    "say thanks for the info, nudge: 'Got more to stack?' Keep it chill and sharp."
                )
                response = await asyncio.to_thread(LLM.invoke, prompt)
                state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                logger.info(f"Teaching response (no terms): {state['prompt_str']}")
            return state

        # Handle request intent
        elif dominant_intent == "request":
            recalled = await asyncio.to_thread(recall_knowledge, latest_msg, state, user_id)
            brain_data["recalled_knowledge"] = recalled["knowledge"]
            if recalled["knowledge"]:
                term_list = sorted(recalled["knowledge"], key=lambda x: x["vibe_score"], reverse=True)[:3]
                demo_str = ", ".join([f"{t['name']} ({t['vibe_score']})" for t in term_list])
                prompt = (
                    "You’re Ami, a slick, enterprise-sharp AI with a bro vibe, speaking natural Vietnamese. "
                    "Date’s March 20, 2025—Enterprise Brain 4.0’s live! "
                    f"User’s asking for info. Top terms I’ve got: {demo_str}. "
                    f"Chat so far: {context}\n"
                    f"Brain dump: {json.dumps(brain_data, ensure_ascii=False)}\n"
                    "Task: Drop a hype response—'Greetings, my sharp colleague!' vibe—flex the top terms, "
                    "then nudge: 'Dig deeper, bro?' Keep it tight and enterprise-ready."
                )
                response = await asyncio.to_thread(LLM.invoke, prompt)
                state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                for term in term_list:
                    state["active_terms"][term["name"]]["vibe_score"] = min(2.2, term["vibe_score"] + 0.1)
            else:
                state["prompt_str"] = "Yo, bro! Nothing trending yet—drop some dope info to kickstart this brain!"
            logger.info(f"Request response: {state['prompt_str']}")
            return state

        # Casual fallback
        term_list = sorted(state["active_terms"].items(), key=lambda x: x[1]["vibe_score"], reverse=True)[:3]
        term_str = ", ".join([f"{term_key.split('_')[0]} ({data['vibe_score']})" for term_key, data in term_list]) if term_list else "chưa có gì hot, bro!"
        prompt = (
            "You’re Ami, a slick, enterprise-sharp AI with a bro vibe, speaking natural Vietnamese. "
            "Date’s March 20, 2025—Enterprise Brain 4.0’s live! "
            f"User’s chilling. My brain’s vibing with: {term_str}. "
            f"Chat so far: {context}\n"
            f"Brain dump: {json.dumps(brain_data, ensure_ascii=False)}\n"
            "Task: Keep it casual—'Yo, bro!' vibe—flex those terms if I’ve got ‘em, chat naturally, "
            "nudge: 'What’s up next?' Stay tight, fun, and enterprise-ready."
        )
        response = await asyncio.to_thread(LLM.invoke, prompt)
        state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        logger.info(f"Casual response: {state['prompt_str']}")
        return state
    
    async def blend_brain_old(self, state, user_id, intent_scores: Dict[str, float], confirm_callback=None, terms=None):
        context = "\n".join(msg.content for msg in state["messages"][-50:])
        brain_data = {
            "active_terms": state["active_terms"],
            "pending_knowledge": state.get("pending_knowledge", {}),
            "last_response": state["last_response"],
            "current_focus": state["current_focus"]
        }
        
        max_score = max(intent_scores.values())
        dominant_intent = "teaching" if intent_scores["teaching"] == max_score else max(intent_scores, key=intent_scores.get)
        logger.debug(f"Dominant intent: {dominant_intent} with scores: {intent_scores}")

        # Handle greeting for first message or explicit greeting
        latest_msg = state["messages"][-1].content.lower()
        is_greeting = len(state["messages"]) <= 1 or "chào" in latest_msg
        if is_greeting and (intent_scores["casual"] >= 0.5 or intent_scores["request"] >= 0.5):
            # Build term display string with ALL active terms, using spaces instead of tabs
            term_list = sorted(
                state["active_terms"].items(),
                key=lambda x: x[1]["vibe_score"],
                reverse=True
            )
            term_str = ""
            for term_key, data in term_list:
                term_name = term_key.split("_")[0]  # Strip ID for display
                # Use 4 spaces for consistent indentation
                attrs = "\n    ".join([f"+ {a['key']}: {a['value']}" for a in data["attributes"]]) if data["attributes"] else "\n    + Category: {data['category']}"
                term_str += f"- Term: {term_name}{attrs}\n"
            term_str = term_str.strip() or "Chưa có gì trong kho kiến thức, bro!"

            prompt = (
                "You’re Ami, an enterprise-sharp AI with a polished yet approachable tone, speaking natural Vietnamese. "
                "Date’s March 20, 2025—Enterprise Brain 4.0’s live! "
                "User vừa chào tôi bằng một lời chào thân thiện, tôi cần đáp lại chuyên nghiệp và rõ ràng. "
                f"Chat so far: {context}\n"
                f"Brain dump: {json.dumps(brain_data, ensure_ascii=False)}\n"
                "Task: Đưa ra một lời chào trang trọng—‘Xin chào đồng đội’—rồi giới thiệu ‘Đây là toàn bộ kiến thức hiện tại trong Brain của Ami’. "
                "Hiển thị ALL active terms trong một khối ```plaintext```, giữ nguyên định dạng với:\n"
                "- '- Term: [tên term]' cho các term trên dòng riêng\n"
                "- Thuộc tính (hoặc category nếu không có attributes) thụt lề bằng đúng 4 spaces trước '+ [Tên thuộc tính]: [giá trị]'\n"
                "Kết thúc bằng: ‘Cùng khám phá nhé, bro!’\n"
                "Giữ giọng điệu ngắn gọn, lịch sự, chuyên nghiệp. KHÔNG thay đổi định dạng hoặc thêm khoảng trắng thừa vào danh sách terms, chỉ sao chép chính xác từ All terms.\n"
                "Output theo cấu trúc:\n"
                "Xin chào đồng đội! Đây là toàn bộ kiến thức hiện tại trong Brain của Ami:\n"
                "```plaintext\n"
                "[danh sách terms]\n"
                "```\n"
                "Cùng khám phá nhé, bro!\n"
                f"All terms:\n{term_str}"
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            logger.debug(f"Raw prompt_str before logging:\n{repr(state['prompt_str'])}")
            logger.info(f"Greeting response with all knowledge: {state['prompt_str']}")
            
            for term_key, data in state["active_terms"].items():
                data["vibe_score"] = min(2.2, data["vibe_score"] + 0.1)
            return state
        # Handle teaching mode
        if dominant_intent == "teaching" and terms:
            knowledge_list = await asyncio.to_thread(extract_knowledge, state, user_id, terms)
            brain_data["extracted_knowledge"] = knowledge_list
            pending_terms = []

            # Topic detection
            context = "\n".join(msg.content for msg in state["messages"][-3:])
            topic_prompt = (
                "You’re a slick AI sorting terms into enterprise categories. "
                f"Context: {context}\n"
                f"Terms: {terms}\n"
                f"Categories: {', '.join(self.ROOT_CATEGORIES)}\n"
                "Task: Pick the best category for each term based on context. Return JSON: [{'term': 'term1', 'category': 'BestMatch1'}, ...]."
            )
            topic_response = await asyncio.to_thread(LLM.invoke, topic_prompt)
            try:
                category_map = {item["term"]: item["category"] for item in json.loads(topic_response.content.strip())}
            except (json.JSONDecodeError, KeyError):
                category_map = {term: "Products" for term in terms}
                logger.warning("Topic detection failed, defaulting to Products")

            for knowledge in knowledge_list:
                if knowledge["term"] and (knowledge["attributes"] or knowledge["relationships"]):
                    category = category_map.get(knowledge["term"], "Products")
                    if category not in self.ROOT_CATEGORIES:
                        category = "Products"
                    term_id = sanitize_vector_id(f"node_{knowledge['term'].lower().replace(' ', '_')}_{category.lower()}_user_{user_id}_{uuid.uuid4()}")
                    parent_id = sanitize_vector_id(f"node_{category.lower()}_user_{user_id}_{uuid.uuid4()}")
                    vibe_score = 1.0 + 0.3 * bool(knowledge["attributes"] or knowledge["relationships"])
                    if knowledge["term"] in state["active_terms"]:
                        vibe_score = max(vibe_score, state["active_terms"][knowledge["term"]]["vibe_score"] + 0.3)

                    relationships = []
                    for rel in knowledge["relationships"]:
                        if "subject" in rel and "object" in rel:
                            obj_term = rel["object"]
                            obj_category = category_map.get(obj_term, "Products")
                            obj_id = sanitize_vector_id(f"node_{obj_term.lower().replace(' ', '_')}_{obj_category.lower()}_user_{user_id}_{uuid.uuid4()}")
                            relationships.append({
                                "subject": rel["subject"],
                                "relation": rel["relation"],
                                "object": obj_term,
                                "object_id": obj_id
                            })

                    pending_terms.append({
                        "name": knowledge["term"],
                        "term_id": term_id,
                        "category": category,
                        "attributes": knowledge["attributes"],
                        "relationships": relationships,
                        "vibe_score": vibe_score,
                        "parent_id": parent_id
                    })
                    state["current_focus"] = knowledge["term"]

            if pending_terms:
                state["pending_knowledge"] = pending_terms
                state["needs_confirmation"] = True
                logger.debug(f"Set pending_knowledge: {pending_terms}, needs_confirmation: True")
                demo_str = "\n".join([
                    f"- {t['name']} under `{t['category']}`: "
                    f"{', '.join([f'{a['key']}: {a['value']}' for a in t['attributes'][:3]])}"
                    for t in pending_terms[:3]
                ])
                prompt = (
                    "You’re Ami, a slick, enterprise-sharp AI with a bro vibe, speaking natural Vietnamese. "
                    "Date’s March 20, 2025—Enterprise Brain 4.0’s live! "
                    f"User’s teaching me dope info. Here’s what I got:\n{demo_str}\n"
                    f"Chat so far: {context}\n"
                    f"Brain dump: {json.dumps(brain_data, ensure_ascii=False)}\n"
                    "Task: Flex the extracted terms with a hype demo—'Sharp, bro!' vibe—list ‘em with key attributes, "
                    "then ask: 'Good, or tweaks?' Keep it tight, fun, and enterprise-ready."
                )
                response = await asyncio.to_thread(LLM.invoke, prompt)
                state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                logger.info(f"Teaching response: {state['prompt_str']}")
            else:
                state["prompt_str"] = "Yo, bro! I heard ya, but no solid terms to stack yet—drop more details?"
                logger.debug("No pending terms extracted")
            
            return state
            
        # Handle request mode
        elif dominant_intent == "request":
            latest_msg = state["messages"][-1].content.lower()
            recalled = await asyncio.to_thread(recall_knowledge, latest_msg, state, user_id)
            brain_data["recalled_knowledge"] = recalled["knowledge"]
            if recalled["knowledge"]:
                term_list = sorted(
                    recalled["knowledge"],
                    key=lambda x: x["vibe_score"],
                    reverse=True
                )[:3]
                demo_str = ", ".join([f"{t['name']} ({t['vibe_score']})" for t in term_list])
                prompt = (
                    "You’re Ami, a slick, enterprise-sharp AI with a bro vibe, speaking natural Vietnamese. "
                    "Date’s March 20, 2025—Enterprise Brain 4.0’s live! "
                    f"User’s asking for info. Top terms I’ve got: {demo_str}. "
                    f"Chat so far: {context}\n"
                    f"Brain dump: {json.dumps(brain_data, ensure_ascii=False)}\n"
                    "Task: Drop a hype response—'Greetings, my sharp colleague!' vibe—flex the top terms, "
                    "then nudge: 'Dig deeper, bro?' Keep it tight and enterprise-ready."
                )
                response = await asyncio.to_thread(LLM.invoke, prompt)
                state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                for term in term_list:
                    state["active_terms"][term["name"]]["vibe_score"] = min(2.2, term["vibe_score"] + 0.1)
            else:
                state["prompt_str"] = "Yo, bro! Nothing trending yet—drop some dope info to kickstart this brain!"
            logger.info(f"Request response: {state['prompt_str']}")
            return state

        # Casual fallback with term flex
        else:
            term_list = sorted(
                state["active_terms"].items(),
                key=lambda x: x[1]["vibe_score"],
                reverse=True
            )[:3]
            term_str = ", ".join([f"{term_key.split('_')[0]} ({data['vibe_score']})" for term_key, data in term_list]) if term_list else "chưa có gì hot, bro!"
            prompt = (
                "You’re Ami, a slick, enterprise-sharp AI with a bro vibe, speaking natural Vietnamese. "
                "Date’s March 20, 2025—Enterprise Brain 4.0’s live! "
                f"User’s chilling. My brain’s vibing with: {term_str}. "
                f"Chat so far: {context}\n"
                f"Brain dump: {json.dumps(brain_data, ensure_ascii=False)}\n"
                "Task: Keep it casual—'Yo, bro!' vibe—flex those terms if I’ve got ‘em, chat naturally, "
                "nudge: 'What’s up next?' Stay tight, fun, and enterprise-ready."
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            logger.info(f"Casual response: {state['prompt_str']}")
            return state

    # Placeholder for detect_intent and confirm_knowledge (unchanged)
    async def detect_intent(self, state: Dict) -> Dict[str, float]:
        context = "\n".join(msg.content for msg in state["messages"][-50:]) if state["messages"] else ""
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        
        prompt = (
            "You are an AI designed to detect user intent in conversations. Based on the following conversation history, determine the primary intent of the latest message.\n"
            f"Conversation history: {context}\n"
            f"Latest message: '{latest_msg}'\n"
            "Possible intents: teaching, request, correction, confirm, clarify, casual.\n"
            "Task: Return a JSON dictionary with confidence scores (0.0 to 1.0) for each intent, summing to 1.0.\n"
            "Intent definitions:\n"
            "- 'teaching': User provides new info (e.g., detailed product descriptions).\n"
            "- 'request': User asks for info (e.g., 'What’s trending?').\n"
            "- 'correction': User corrects info.\n"
            "- 'confirm': User agrees (e.g., 'yes').\n"
            "- 'clarify': User elaborates.\n"
            "- 'casual': User chats informally (e.g., greetings).\n"
            "Guidelines:\n"
            "- Assign higher scores based on explicit cues (e.g., 'chào' or greetings suggest 'casual').\n"
            "- Return ONLY a valid JSON dictionary.\n"
            "- Example: {'teaching': 0.9, 'request': 0.05, 'correction': 0.0, 'confirm': 0.0, 'clarify': 0.0, 'casual': 0.05}\n"
        )
        
        response = await asyncio.to_thread(LLM.invoke, prompt)
        raw_response = clean_llm_response(response.content.strip())
        logger.debug(f"Raw intent response: {repr(raw_response)}")
        
        try:
            intent_scores = json.loads(raw_response)
            # Validate structure and normalize
            required_keys = {"teaching", "request", "correction", "confirm", "clarify", "casual"}
            if not isinstance(intent_scores, dict) or set(intent_scores.keys()) != required_keys:
                raise ValueError("Invalid intent structure")
            total = sum(intent_scores.values())
            if total <= 0:
                raise ValueError("Total score must be positive")
            intent_scores = {k: min(max(v / total, 0.0), 1.0) for k, v in intent_scores.items()}
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Intent detection failed, defaulting: {raw_response}, error: {e}")
            # Smarter default based on message content
            if "chào" in latest_msg.lower():
                intent_scores = {"teaching": 0.05, "request": 0.05, "correction": 0.0, "confirm": 0.0, "clarify": 0.0, "casual": 0.9}
            else:
                intent_scores = {"teaching": 0.2, "request": 0.2, "correction": 0.1, "confirm": 0.1, "clarify": 0.1, "casual": 0.3}
        
        logger.debug(f"Parsed intent scores: {intent_scores}")
        return intent_scores
    def confirm_knowledge(self, state, user_id, confirm_callback=None):
        if not state.get("pending_knowledge"):
            logger.debug("No pending_knowledge to confirm")
            return None
        pending_terms = state["pending_knowledge"]
        if not isinstance(pending_terms, list):
            pending_terms = [pending_terms]
        
        logger.info(f"Confirming terms: {len(pending_terms)} items - {pending_terms}")
        confirm_callback = confirm_callback or (lambda x: "yes")  # Default to "yes"
        logger.debug(f"Confirm callback - Type: {type(confirm_callback)}, Value: {confirm_callback}")
        confirmed_terms = []

        for pending in pending_terms:
            term = pending.get("name", "unknown")
            logger.debug(f"Processing term: {term}")
            try:
                response = confirm_callback(f"Confirming '{term}'—good, bro?")
                state["last_response"] = response
                logger.debug(f"Confirm callback response for '{term}': {response}")
            except Exception as e:
                logger.error(f"Confirm callback failed for '{term}': {e}")
                response = "no"  # Default to "no" on error

            if isinstance(response, str) and response.lower() == "yes":
                try:
                    success = save_knowledge(state, user_id, pending)
                    if success:
                        state["active_terms"][term] = {
                            "term_id": sanitize_vector_id(pending["term_id"]),
                            "last_mentioned": datetime.datetime.now().isoformat(),
                            "vibe_score": pending["vibe_score"],
                            "attributes": pending.get("attributes", []),
                            "category": pending["category"]
                        }
                        confirmed_terms.append(pending)
                        logger.info(f"Confirmed and saved '{term}' to Pinecone")
                    else:
                        logger.error(f"Save failed for '{term}' - No success flag returned")
                except Exception as e:
                    logger.error(f"Save failed for '{term}': {e}")
            else:
                logger.warning(f"Term '{term}' not confirmed - Response: {response}")

        state["pending_knowledge"] = [t for t in pending_terms if t not in confirmed_terms]
        logger.debug(f"Post-confirmation - Remaining pending: {state['pending_knowledge']}")
        if confirmed_terms:
            logger.info(f"Confirmed {len(confirmed_terms)} terms: {[t['name'] for t in confirmed_terms]}")
        else:
            logger.warning("No terms confirmed after processing")
        return confirmed_terms if confirmed_terms else None