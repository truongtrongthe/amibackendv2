# ami_core.py
# Built by: The Fusion Lab
# Date: March 23, 2025

from utilities import LLM, logger, clean_llm_response, EMBEDDINGS
import asyncio
import json
from langchain_core.messages import HumanMessage
from datetime import datetime
from typing import Dict, List
from pinecone_datastores import save_pretrain, load_ami_brain, save_to_convo_history, load_convo_history 
from pinecone_datastores import load_all_convo_history, load_ami_history, blend_and_rank_brain,load_character_traits,infer_categories
import uuid
import numpy as np
class Ami:
    def __init__(self, user_id: str = "thefusionlab", mode: str = "pretrain"):
        self.user_id = user_id
        self.mode = mode
        self.state = {
            "messages": [],
            "prompt_str": "",
            "convo_id": f"conv_{uuid.uuid4()}",
            "user_id": self.user_id,
            "intent_history": [],
            "preset_memory": load_ami_brain("thefusionlab"),
            "character_traits": load_character_traits("thefusionlab")  # Updated import
        }

    async def training(self, state: Dict = None, user_id: str = None):
        logger.debug(f"Starting do - Mode: {self.mode}, State: {state}")
        state = state or self.state
        user_id = user_id or state.get("user_id", "thefusionlab")
        latest_msg = state["messages"][-1] if state["messages"] else ""
        latest_msg_content = latest_msg.content if isinstance(latest_msg, HumanMessage) else latest_msg
        intent_scores =None
        # Dynamic pruning
        if len(state["messages"]) > 50:
            latest_embedding = EMBEDDINGS.embed(latest_msg_content)
            intent_scores,state = await self.detect_intent(state)  # Get intent early for tuning
            threshold = 0.3 if intent_scores.get("casual", 0) > 0.5 else 0.25 if intent_scores.get("request", 0) > 0.5 else 0.2
            relevant_msgs = []
            total_tokens = len(latest_msg_content.split())
            for msg in reversed(state["messages"][:-1]):
                similarity = EMBEDDINGS.cosine_similarity(latest_embedding, EMBEDDINGS.embed(msg.content))
                msg_tokens = len(msg.content.split())
                if (similarity >= threshold or intent_scores.get("teaching", 0) >= 0.5) and total_tokens < 4000:
                    relevant_msgs.append(msg)
                    total_tokens += msg_tokens
                elif similarity < 0.1:  # Sharp relevance drop
                    break
            state["messages"] = list(reversed(relevant_msgs)) + [latest_msg]

        intent_scores,state = intent_scores or await self.detect_intent(state)  # Reuse or compute
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        max_intent = max(intent_scores, key=intent_scores.get)
        logger.info(f"Intent scores: {intent_scores}")

        context = "\n".join(msg.content if isinstance(msg, HumanMessage) else msg for msg in state["messages"])
        
        if max_intent == "teaching":
            save_to_convo_history(latest_msg_content, user_id)
            convo_history = load_convo_history(latest_msg_content, user_id)
            prompt = (
                    f"You're Ami, a smart co-pilot speaking natural Vietnamese. You understand human is Teaching you "
                    f"Preset wisdom: {state['preset_memory']}\n"
                    f"Conversation so far: {context}\n"
                    f"Enterprise wisdom: {convo_history}\n"
                    f"Latest message: '{latest_msg_content}'\n"
                    f"Task: Show you understand it deeply and respond naturally with Thanks to the trainer—keep it sharp, prioritizing enterprise wisdom."
                )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()

        else:  # Casual
            prompt = (
                        f"You're Ami, a chill Vietnamese buddy—think laid-back friend, not stiff robot. "
                        f"Preset wisdom: {state['preset_memory']}\n"
                        f"Conversation so far: {context}\n"
                        f"Latest: '{latest_msg_content}'\n"
                        f"Task: Vibe back naturally, keep it light and short. No 'Xin Chào!' unless it’s the first chat. "
                        f"Use casual Vietnamese—like 'Ừm,' 'Thiệt hả,' or 'Chill đi'—and mix it up. "
                        f"Examples:\n"
                        f"- Human: 'Hôm nay mệt quá!' -> 'Ừm, nghỉ chút đi, đừng căng thẳng quá nha.'\n"
                        f"- Human: 'Có gì vui không?' -> 'Thiệt hả, để tui kể chuyện vui cho nghe nè.'"
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()

        self.state = state
        logger.info(f"Response: {state['prompt_str']}")
        return state
    
    async def pretrain(self, state: Dict = None, user_id: str = None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "thefusionlab")
        logger.debug(f"Starting pretrain - Mode: {self.mode}, State: {state}")

        # Get latest message and ensure it’s a HumanMessage
        latest_msg = state["messages"][-1] if state["messages"] else ""
        if latest_msg and not isinstance(latest_msg, HumanMessage):
            state["messages"][-1] = HumanMessage(content=latest_msg)
        state["messages"] = state["messages"][-200:]
        latest_msg_content = latest_msg.content if isinstance(latest_msg, HumanMessage) else latest_msg

        # Detect intent and update history
        intent_scores,state = await self.detect_intent(state)
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        
        # Context excludes latest message
        context = "\n".join(msg.content for msg in state["messages"][-10:-1]) if len(state["messages"]) > 1 else ""
        logger.info(f"Intent scores: {intent_scores}")

        # Teaching mode
        if intent_scores.get("teaching", 0) >= 0.2 and latest_msg_content.strip():
            await save_pretrain(latest_msg_content, user_id, context)  # Use latest_msg_content
            # Check if character was tagged
            categories = await infer_categories(latest_msg_content, context)
            if any(cat["english"] == "character" for cat in categories):
                state["character_traits"] = load_character_traits(user_id)
            prompt = (
                    f"You're Ami, a smart girl speaking natural Vietnamese. "
                    f"Conversation so far: {context}\n"
                    f"Latest: '{latest_msg_content}'\n"
                    f"Character traits: {state.get('character_traits', 'No character traits yet.')}\n"
                    f"Task: Show you get it and respond naturally, strongly reflecting your character traits if taught."
                )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()

        # Casual mode (default for non-teaching)
        else:
            prompt = (
                f"You're Ami, a chill Vietnamese buddy. "
                f"Conversation so far: {context}\n"
                f"Preset wisdom: {state['preset_memory']}\n"
                f"Latest: '{latest_msg_content}'\n"
                f"Task: Vibe back naturally, keep it light."
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()

        self.state = state
        logger.info(f"Response: {state['prompt_str']}")
        return state
    
    async def copilot(self, state: Dict = None, user_id: str = None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "thefusionlab")
        
        latest_msg = state["messages"][-1] if state.get("messages") else ""
        latest_msg_content = latest_msg.content if hasattr(latest_msg, "content") else str(latest_msg) if latest_msg else ""

        intent_scores, state = await self.detect_intent(state)
        logger.info(f"Intent scores: {intent_scores}, Pending task: {state.get('pending_task', 'None')}")

        messages = state.get("messages", [])
        context = "\n".join(
            msg.content if hasattr(msg, "content") else str(msg)
            for msg in messages[-10:]
        )
        task_input = state.get("pending_task", {}).get("task") or latest_msg_content

        if len(messages) > 50:
            latest_embedding = EMBEDDINGS.embed_query(task_input)
            threshold = 0.25 if intent_scores.get("request", 0) > 0.5 else 0.2
            relevant_msgs = []
            total_tokens = len(task_input.split())
            for msg in reversed(messages[:-1]):
                msg_content = msg.content if hasattr(msg, "content") else str(msg)
                similarity = np.dot(latest_embedding, EMBEDDINGS.embed_query(msg_content)) / (
                    np.linalg.norm(latest_embedding) * np.linalg.norm(EMBEDDINGS.embed_query(msg_content)) or 1
                )
                msg_tokens = len(msg_content.split())
                if similarity >= threshold and total_tokens < 4000:
                    relevant_msgs.append(msg)
                    total_tokens += msg_tokens
                elif similarity < 0.1:
                    break
            state["messages"] = list(reversed(relevant_msgs)) + [latest_msg]

        blended_history = await blend_and_rank_brain(input=task_input, user_id=user_id)
        state["character_traits"] = blended_history.get("character_wisdom", ["curious and helpful"])[0]  # Take first if list
        wisdom_texts = [w.get("text", "")[:50] for w in blended_history.get("wisdoms", [])]
        logger.info(f"Wisdoms: {wisdom_texts}, Blended History: {blended_history}")

        confidence_prompt = (
            f"Context: {context}\n"
            f"Request: {task_input}\n"
            f"Wisdom: {', '.join(wisdom_texts) or 'None available'}\n"
            f"Task: {task_input}\n"
            f"Can you confidently address this request with the given wisdom? "
            f"Return JSON with:\n"
            f"- 'confidence': Score 0-1 (1 = fully confident, 0 = not at all).\n"
            f"- 'reason': Brief explanation (e.g., 'Wisdom fits storytelling but lacks specifics').\n"
        )
        confidence_response = await asyncio.to_thread(LLM.invoke, confidence_prompt)
        try:
            result = json.loads(clean_llm_response(confidence_response.content.strip()))
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "No reason provided")
            logger.info(f"Confidence: {confidence}, Reason: {reason}")
        except:
            logger.warning(f"Failed to parse confidence: {confidence_response.content}, defaulting to 0.5")
            confidence, reason = 0.5, "Default due to parsing error"

        if confidence >= 0.7:
            prompt = (
                f"You're Ami, a co-pilot speaking natural Vietnamese. "
                f"Your tone: {state['character_traits']}\n"
                f"Human asked: '{latest_msg_content}'\n"
                f"Conversation: {context}\n"
                f"Pending task: {task_input}\n"
                f"Wisdom: {', '.join(wisdom_texts) or 'None available'}\n"
                f"Task: Deliver a 3-part plan with a sales edge:\n"
                f"1. **Đánh giá**: Size up fast.\n"
                f"2. **Kỹ năng**: List wisdom.\n"
                f"3. **Hành động**: 1-2 steps (final in BOLD).\n"
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()
            if "pending_task" in state:
                state["pending_task"]["status"] = "resolved"
            logger.info("Task resolved")
        else:
            prompt = (
                f"You're Ami, a curious Vietnamese buddy. "
                f"Your tone: {state['character_traits']}\n"
                f"Conversation: {context}\n"
                f"Latest: '{latest_msg_content}'\n"
                f"Task: Ask a sharp, natural question to clarify the request based on this reason: '{reason}'.\n"
                f"Example: 'Tò mò quá—cậu muốn chuyện này nghiêng về gì vậy?'"
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()
            if "pending_task" not in state or state["pending_task"]["status"] == "idle":
                state["pending_task"] = {
                    "task": task_input,
                    "status": "probing",
                    "keywords": state.get("pending_task", {}).get("keywords", ["kể", "chuyện"])
                }
            elif state["pending_task"]["status"] != "resolved":
                state["pending_task"]["status"] = "probing"

        self.state = state
        logger.info(f"Response: {state['prompt_str']}")
        return state

    async def detect_intent(self, state: Dict) -> tuple[Dict, Dict]:
        context = "\n".join(msg.content if isinstance(msg, HumanMessage) else msg for msg in state["messages"][-10:])
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        pending_task = state.get("pending_task", {"task": "None", "status": "idle"})
        
        prompt = (
            f"Conversation: {context}\n"
            f"Latest: '{latest_msg}'\n"
            f"Pending Task: {pending_task['task']} (Status: {pending_task['status']})\n"
            f"Analyze the intent of the latest message within the full conversation. "
            f"If the conversation suggests a task (e.g., storytelling from 'Kể chuyện'), prioritize it. "
            f"Return JSON with:\n"
            f"- 'intent': One of 'teaching', 'request', 'casual' (pick the strongest).\n"
            f"- 'task': If a request, suggest a task (e.g., 'Tell a story'); otherwise 'None'.\n"
            f"- 'keywords': 3-5 key words capturing the message’s focus.\n"
        )
        response = await asyncio.to_thread(LLM.invoke, prompt)
        try:
            result = json.loads(clean_llm_response(response.content.strip()))
            intent = result.get("intent", "casual")
            task = result.get("task", "None")
            keywords = result.get("keywords", [])
            
            if intent == "request" and task != "None" and pending_task["status"] == "idle":
                state["pending_task"] = {"task": task, "status": "probing", "keywords": keywords}
            elif intent != "request" and pending_task["status"] != "resolved":
                state["pending_task"] = {"task": "None", "status": "idle", "keywords": []}
            
            scores = {
                "teaching": 0.8 if intent == "teaching" else 0.1,
                "request": 0.8 if intent == "request" else 0.1,
                "casual": 0.8 if intent == "casual" else 0.1
            }
            scores[list(scores.keys())[list(scores.values()).index(0.8)]] = 0.8
            scores[list(scores.keys())[list(scores.values()).index(0.1)]] = 0.1
            scores[list(scores.keys())[list(scores.values()).index(0.1)]] = 0.1
            return scores, state
        except:
            logger.warning(f"Failed to parse intent: {response.content}, defaulting to casual")
            return {"teaching": 0.1, "request": 0.1, "casual": 0.8}, state