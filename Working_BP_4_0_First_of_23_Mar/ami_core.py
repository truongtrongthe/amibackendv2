# ami_core_4_0.py
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 21, 2025 
# Purpose: Core Ami logic for human teaching AI, with dual-language intent detection

from utilities import detect_intent, extract_knowledge, recall_knowledge, save_knowledge, LLM, index, logger, EMBEDDINGS
import json
import re
from datetime import datetime
import uuid
import asyncio
from threading import Thread
from langchain_core.messages import HumanMessage

class Ami:
    def __init__(self):
        self.brain = []
        self.customer_data = []
        self.user_id = "tfl"
        self.presets = {
            "situation": lambda x: f"Tell me about {x} in your current setup.",
            "challenger": lambda x: f"Reframe it like this: {x}"
        }
        self.state = {
            "messages": [],
            "prompt_str": "",
            "convo_id": None,
            "active_terms": {},
            "pending_knowledge": {},
            "last_response": "",
            "user_id": self.user_id,
            "needs_confirmation": False,
            "intent_history": []
        }

    def load_brain_from_pinecone(self):
        pass

    
    def confirm_knowledge(self, state, user_id, confirm_callback=None):
        if "pending_knowledge" not in state or not state["pending_knowledge"]:
            logger.info("No pending knowledge to confirm")
            return None
        
        pending = state["pending_knowledge"]
        term = pending.get("name", "unknown")
        confirm_callback = confirm_callback or (lambda x: "yes")

        state["prompt_str"] = f"Ami hiểu '{term}' thế này nhé—lưu không bro?"
        response = confirm_callback(state["prompt_str"])
        state["last_response"] = response

        if not response:
            logger.error(f"Callback failed, forcing 'yes' for {state['prompt_str']}")
            response = "yes"

        if response == "yes":
            if "term_id" not in pending or "category" not in pending:
                logger.error(f"Pending knowledge missing required fields: {pending}")
                return None
            
            if term in state["active_terms"]:
                pending["vibe_score"] = state["active_terms"][term]["vibe_score"]
            else:
                pending.setdefault("vibe_score", 1.0)
            pending.setdefault("parent_id", f"node_{pending['category'].lower()}_user_{user_id}_{uuid.uuid4()}")
            
            try:
                save_knowledge(state, user_id)
                logger.info(f"Knowledge saved for term: '{term}'")
            except Exception as e:
                logger.error(f"Failed to save knowledge: {e}")
                return None
            
        elif response == "no":
            state["prompt_str"] = f"OK, bỏ qua '{term}'. Còn gì thêm không bro?"
            state.pop("pending_knowledge", None)
        
        return pending
    
    async def handle_casual(self, state, user_id, window_size=5):
        logger.info(f"Entering handle_casual - User: {user_id}, Messages: {len(state['messages'])}")
        
        # Limit context to last `window_size` messages
        context_messages = state["messages"][-window_size:] if len(state["messages"]) > window_size else state["messages"]
        convo_history = "\n".join([msg.content for msg in context_messages])
        latest_msg = state["messages"][-1].content
        
        # Improved prompt with pronoun guidance
        prompt = (
            "Bạn là Ami, một AI thân thiện và tự nhiên, nói tiếng Việt. "
            "Dựa trên đoạn trò chuyện gần đây sau đây, hãy trả lời câu cuối cùng một cách thoải mái, tự nhiên, như con người:\n"
            f"{convo_history}\n"
            "Chọn cách xưng hô phù hợp (như 'mình', 'tôi', 'tớ', 'em', 'bạn', v.v.) dựa trên giọng điệu và ngữ cảnh của người dùng. "
            "Ví dụ: nếu người dùng nói thoải mái ('ê', 'này'), dùng 'mình' hoặc 'tớ'; nếu trang trọng, dùng 'tôi'; nếu thân mật, có thể dùng 'em'. "
            "Trả lời ngắn gọn, hợp ngữ cảnh, và giữ giọng điệu gần gũi."
        )
        
        try:
            # Call LLM from utilities
            response = await asyncio.to_thread(LLM, prompt)
            # Extract content from AIMessage and strip
            response_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            logger.info(f"Casual response generated: {response_text}")
            return response_text
        except Exception as e:
            logger.error(f"Failed to generate casual response: {e}", exc_info=True)
            return "Ờm, mình hơi lơ mơ—bạn nói gì tiếp đi nào!"  # Default to "mình" for fallback
    
    async def handle_confirm(self, state, user_id):
        if state["intent"] == "confirm":
            logger.info("Confirming knowledge")
            self.confirm_knowledge(state, user_id, lambda x: "yes")
            state["needs_confirmation"] = False
            return f"{user_id.split('_')[0]}, nice! Ami lưu xong—kiến thức đỉnh thật!"
        elif state["intent"] == "clarify":
            logger.info("Clarifying, discarding pending knowledge")
            state.pop("pending_knowledge", None)
            state["needs_confirmation"] = False
            return "OK, tell me again—what’s the term?"
        else:
            return "Bro, just say 'yes' or 'no' to confirm!"

    async def handle_teaching(self, state, user_id, confirm_callback):
        logger.info(f"Entering handle_teaching - User: {user_id}, Messages: {len(state['messages'])}")
        logger.debug(f"Initial state - Active Terms: {state['active_terms']}, Pending Knowledge: {state.get('pending_knowledge', 'None')}")

        initial_active_terms = state["active_terms"].copy()
        logger.debug(f"Captured initial active terms: {initial_active_terms}")

        logger.info("Calling extract_knowledge for teaching intent")
        knowledge = await asyncio.to_thread(extract_knowledge, state, user_id, "teaching")
        logger.info(f"Extracted knowledge: {knowledge}")
        logger.debug(f"Post-extraction state - Active Terms: {state['active_terms']}, Pending Knowledge: {state.get('pending_knowledge', 'None')}")

        term = knowledge["term"]
        if term is None:
            response = "Tôi không hiểu khái niệm này! Bạn có thể giải thích kỹ hơn không"
            logger.info(f"No term detected - Response: {response}")
            return response

        is_fresh = term not in initial_active_terms
        logger.debug(f"Checking freshness - Term: '{term}', Is Fresh: {is_fresh}, Initial Active Terms: {initial_active_terms.keys()}, Current Active Terms: {state['active_terms'].keys()}")

        if is_fresh:
            logger.info(f"Fresh term detected: '{term}'")
            if knowledge["confidence"] < 0.8:
                state["needs_confirmation"] = True
                response = f"Có phải bạn đang nói về '{term}' không? Xác nhận hay làm rõ hơn đi!"
                logger.info(f"Low confidence ({knowledge['confidence']}) for fresh term - Response: {response}")
            else:
                logger.debug(f"High confidence ({knowledge['confidence']}) for fresh term - Saving")
                self.confirm_knowledge(state, user_id, confirm_callback)
                state["needs_confirmation"] = False
                updates = []
                if knowledge["attributes"]:
                    attrs_str = ", ".join(f"{a['key']} là {a['value']}" for a in knowledge["attributes"][:3])
                    updates.append(f"Thuộc tính: {attrs_str}")
                if knowledge["relationships"]:
                    rels_str = ", ".join(f"{r['subject']} {r['relation']} {r['object']}" for r in knowledge["relationships"][:2])
                    updates.append(f"Mối quan hệ: {rels_str}")
                update_str = "; ".join(updates) if updates else "thông tin có giá trị"
                response = f"Đã cập nhật '{term}' vào bộ nhớ: {update_str} rồi nhé!"
                logger.info(f"Fresh term saved - Response: {response}")
            return response

        logger.info(f"Existing term detected: '{term}'")
        term_data = initial_active_terms.get(term, {})
        existing_attrs = term_data.get("attributes", [])  # Default to empty list if missing
        if "attributes" not in term_data:
            logger.warning(f"Term '{term}' missing 'attributes' in initial_active_terms: {term_data}")
            state["active_terms"][term]["attributes"] = existing_attrs

        new_attrs = knowledge["attributes"]
        new_rels = knowledge["relationships"]

        existing_attr_set = {(a["key"], a["value"]) for a in existing_attrs}
        new_attr_set = {(a["key"], a["value"]) for a in new_attrs}
        added_attrs = [a for a in new_attrs if (a["key"], a["value"]) not in existing_attr_set]
        logger.debug(f"Diffing attributes - Existing: {existing_attrs}, New: {new_attrs}, Added: {added_attrs}")

        existing_rel_set = {(r["subject"], r["relation"], r["object"]) for r in term_data.get("relationships", [])}
        new_rel_set = {(r["subject"], r["relation"], r["object"]) for r in new_rels}
        added_rels = [r for r in new_rels if (r["subject"], r["relation"], r["object"]) not in existing_rel_set]
        logger.debug(f"Diffing relationships - Existing: {existing_rel_set}, New: {new_rels}, Added: {added_rels}")

        updates = []
        if added_attrs:
            updates.append(f"Thuộc tính: {', '.join(f'{a['key']} là {a['value']}' for a in added_attrs)}")
        if added_rels:
            updates.append(f"Mối quan hệ: {', '.join(f'{r['subject']} {r['relation']} {r['object']}' for r in added_rels)}")
        
        if updates:
            update_str = "; ".join(updates)
            response = f"Đã cập nhật '{term}' vào bộ nhớ: {update_str} rồi nhé!"
            logger.debug(f"Updates detected - Calling confirm_knowledge with: {state['pending_knowledge']}")
            self.confirm_knowledge(state, user_id, confirm_callback)
            state["needs_confirmation"] = False
            logger.info(f"Updates saved - Response: {response}")
        else:
            response = f"Tôi đã biết cái này là nói về '{term}' rồi. Còn thêm gì nữa không?"
            logger.info(f"No new updates for existing term - Response: {response}")

        logger.info(f"Exiting handle_teaching - Response: {response}")
        return response
    
    async def handle_request(self, state, user_id):
        latest_msg = state["messages"][-1].content.lower()
        recalled = await asyncio.to_thread(recall_knowledge, latest_msg, state, user_id)
        term = next((t for t in state["active_terms"].keys() if t.lower() in latest_msg), None)
        
        def get_attrs(source, key_filter=None):
            attrs = source.get("attributes", [])
            return [a for a in attrs if key_filter is None or a["key"].lower() == key_filter] or attrs
        
        if term:
            attrs = get_attrs(state["active_terms"].get(term, {}), "price" if "price" in latest_msg or "giá" in latest_msg else None)
            if attrs:
                return f"Đây là gì tôi biết: {term}: {attrs}"
            elif recalled["knowledge"]:
                attrs = get_attrs(recalled["knowledge"][0], "price" if "price" in latest_msg or "giá" in latest_msg else None)
                if attrs:
                    return f"Đây là thông tin tôi biết: {term}: {attrs}"
        return "Chưa có thông tin—nói thêm cho tôi về {lastest_msg} đi"

    async def handle_correction(self, state, user_id):
        logger.info("Processing correction")
        latest_msg = state["messages"][-1].content
        active_terms = state.get("active_terms", {})
        
        corrected_term = None
        for term in active_terms.keys():
            if term.lower() in latest_msg.lower():
                corrected_term = term
                break
        
        if not corrected_term:
            return "What are you correcting, bro? Gimme a hint!"

        knowledge = await asyncio.to_thread(extract_knowledge, state, user_id, "teaching")
        if knowledge["term"] and knowledge["attributes"]:
            state["pending_knowledge"] = {
                "term_id": active_terms[corrected_term]["term_id"],
                "name": corrected_term,
                "category": "Products",
                "attributes": knowledge["attributes"],
                "relationships": knowledge["relationships"],
                "vibe_score": active_terms[corrected_term]["vibe_score"],
                "parent_id": f"node_products_user_{user_id}_{uuid.uuid4()}"
            }
            self.confirm_knowledge(state, user_id)
            state["needs_confirmation"] = False
            return "Got it, fixed—anything else to tweak?"
        else:
            return "Hmm, couldn’t catch that correction—try again?"
    
    async def load_brain_from_pinecone(self, user_id, top_k=5):
        """Load knowledge from Pinecone for the given user."""
        logger.info(f"Loading brain from Pinecone for user: {user_id}")
        namespace = f"enterprise_knowledge_tree_{user_id}"
        
        try:
            stats = index.describe_index_stats()
            node_count = stats["namespaces"].get(namespace, {}).get("vector_count", 0)
            if node_count == 0:
                logger.info("No nodes found in Pinecone namespace")
                return {"node_count": 0, "terms": []}

            # Query Pinecone for top_k nodes
            response = await asyncio.to_thread(
                index.query,
                vector=[0] * 1536,  # Dummy vector for broad fetch
                top_k=min(node_count, top_k),
                include_metadata=True,
                namespace=namespace
            )
            if not response["matches"]:
                logger.info("No matches returned from Pinecone query")
                return {"node_count": node_count, "terms": []}

            # Extract terms from metadata
            terms = [match["metadata"] for match in response["matches"]]
            logger.info(f"Loaded {len(terms)} terms from Pinecone: {[t['name'] for t in terms]}")
            return {"node_count": node_count, "terms": terms}

        except Exception as e:
            logger.error(f"Failed to load brain from Pinecone: {e}", exc_info=True)
            return {"node_count": 0, "terms": [], "error": str(e)}
        
    async def do(self, state=None, is_first=False, confirm_callback=None, user_id=None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "tfl")
        logger.info(f"Navigating - Messages: {len(state['messages'])}, User: {user_id}, Needs Confirmation: {state['needs_confirmation']}")

        try:
            # Startup brain flex for first message
            if len(state["messages"]) == 1:
                logger.info("First message detected - Loading brain")
                brain_data = await self.load_brain_from_pinecone(user_id)
                if brain_data["terms"]:
                    names_str = ", ".join(f"‘{term['name']}’" for term in brain_data["terms"])
                    state["prompt_str"] = f"Ami đang có {brain_data['node_count']} kiến thức trong đầu—Ví dụ như {names_str}! {user_id}muốn thử không?"
                else:
                    state["prompt_str"] = f"Hi, {user_id}, đầu Ami đang chưa có gì!"
                logger.info(f"Startup response: {state['prompt_str']}")
                self.state = state
                return state
            # Normal flow for subsequent messages
            intent = await asyncio.to_thread(detect_intent, state)
            state["intent"] = intent
            logger.info(f"Intent detected: '{intent}' for message: '{state['messages'][-1].content if state['messages'] else 'None'}'")
            if state["needs_confirmation"] and intent in ["confirm", "clarify"]:
                state["prompt_str"] = await self.handle_confirm(state, user_id)
                if intent == "confirm":
                    state["needs_confirmation"] = False
            elif intent == "teaching":
                state["prompt_str"] = await self.handle_teaching(state, user_id, confirm_callback)
            elif intent == "request":
                state["prompt_str"] = await self.handle_request(state, user_id)
            elif intent == "correction":
                state["prompt_str"] = await self.handle_correction(state, user_id)
            else:  # Fallback to casual
                state["prompt_str"] = await self.handle_casual(state, user_id)

            logger.info(f"Final state - Prompt: '{state['prompt_str']}', Active Terms: {state['active_terms']}")
            self.state = state

        except Exception as e:
            logger.error(f"Error in do(): {e}", exc_info=True)
            state["prompt_str"] = "Oops, something broke—try again!"
            state["pickup_line"] = "Let’s reset and roll again, bro!"
            self.state = state

        logger.info("Exiting do()")
        return state
# Test with Teaching and Correction Flow
