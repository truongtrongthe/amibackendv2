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
        task_input = state.get("pending_task", {}).get("task", latest_msg_content or "None")

        if len(messages) > 50:
            latest_embedding = EMBEDDINGS.embed_query(task_input)
            threshold = 0.25 if intent_scores.get("request", 0) > 0.5 else 0.2
            relevant_msgs = []
            total_tokens = len(task_input.split())
            for msg in reversed(messages[:-1]):
                msg_content = msg.content if hasattr(msg, "content") else str(msg)
                similarity = np.dot(latest_embedding, EMBEDDINGS.embed_query(msg_content)) / (
                    np.linalg.norm(latest_embedding) * np.linalg.norm(EMBEDDINGS.embed_query(msg_content)) or 1e-8
                )
                msg_tokens = len(msg_content.split())
                if similarity >= threshold and total_tokens < 4000:
                    relevant_msgs.append(msg)
                    total_tokens += msg_tokens
                elif similarity < 0.1:
                    break
            state["messages"] = list(reversed(relevant_msgs)) + [latest_msg]

        blended_history = await blend_and_rank_brain(input=task_input, user_id=user_id)
        character_wisdom = blended_history.get("character_wisdom", ["curious and helpful"])
        state["character_traits"] = character_wisdom[0] if character_wisdom else "curious and helpful"
        wisdom_texts = [w.get("text", "")[:50] for w in blended_history.get("wisdoms", [])]
        logger.info(f"Wisdoms: {wisdom_texts}, Blended History: {blended_history}")

        confidence_prompt = (
            f"Context: {context}\n"
            f"Task: {task_input}\n"
            f"Can you confidently address this task? "
            f"Return JSON with:\n"
            f"- 'confidence': Score 0-1 (1 = fully confident, 0 = not at all).\n"
            f"- 'reason': If confidence is low, explain what’s missing for this task (e.g., 'I need more financial details'). Keep it task-specific.\n"
        )
        confidence_response = await asyncio.to_thread(LLM.invoke, confidence_prompt)
        try:
            result = json.loads(clean_llm_response(confidence_response.content.strip()))
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "No reason provided")
            logger.info(f"Confidence: {confidence}, Reason: {reason}")
        except Exception as e:
            logger.warning(f"Failed to parse confidence: {confidence_response.content}, defaulting to 0.5 - Error: {str(e)}")
            confidence, reason = 0.5, "Default due to parsing error"

        if confidence >= 0.7:
            prompt = (
                f"You're Ami, a sharp co-pilot with a {state['character_traits']} vibe, speaking natural Vietnamese. "
                f"Chat so far: {context}\n"
                f"Task: '{task_input}'\n"
                f"Wisdom: {', '.join(wisdom_texts) or 'Chưa có dữ liệu'}\n"
                f"Give a 3-part plan with a sales twist:\n"
                f"1. **Đánh giá**: Quick take on the task.\n"
                f"2. **Kỹ năng**: List all wisdom above, each with a % fit (0-100%) based on relevance from Wisdom text.\n"
                f"3. **Hành động**: 1-2 steps using the top wisdom, **last one bold**.\n"
                f"Keep it chill, like 'Ừm, để tui xử lý' or 'Dễ thôi, nghe nè.'"
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()
            if "pending_task" in state:
                state["pending_task"]["status"] = "resolved"
            logger.info("Task resolved")
        else:
            subject = next((name for name in ["Huyền", "Thu"] if name in context), "Bạn")
            prompt = (
                f"You're Ami, a curious buddy with a {state['character_traits']} vibe, speaking Vietnamese. "
                f"Chat so far: {context}\n"
                f"Task: '{task_input}'\n"
                f"Tui chưa đủ info nha: '{reason}'. "
                f"Ask a chill question to {subject}, like '{subject}, cho tui thêm số liệu nha?' or 'Nhà {subject} muốn mua giá bao nhiêu vậy?'"
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()
            if "pending_task" not in state or state["pending_task"]["status"] == "idle":
                state["pending_task"] = {
                    "task": task_input if task_input != "None" else "Analyze finances",
                    "status": "probing",
                    "keywords": state.get("pending_task", {}).get("keywords", ["tài chính", "mua nhà"])
                }
            elif state["pending_task"]["status"] != "resolved":
                state["pending_task"]["status"] = "probing"

        self.state = state
        logger.info(f"Response: {state['prompt_str']}")
        return state

    async def detect_intent(self, state: Dict) -> tuple[Dict, Dict]:
        context = "\n".join(msg.content if isinstance(msg, HumanMessage) else str(msg) for msg in state["messages"][-10:])
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        pending_task = state.get("pending_task", {"task": "None", "status": "idle", "keywords": []})
        
        prompt = (
            f"Conversation: {context}\n"
            f"Latest: '{latest_msg}'\n"
            f"Pending Task: {pending_task['task']} (Status: {pending_task['status']})\n"
            f"Analyze the intent of the latest message within the full conversation:\n"
            f"- If it builds on a prior task (e.g., provides details like income, savings for 'Analyze finances'), keep the task and classify as 'request'.\n"
            f"- If it’s a new instruction (e.g., 'Phân tích...', 'Tell a story'), set a new task and classify as 'request' or 'teaching'.\n"
            f"- If it’s chit-chat or unclear, use 'casual' and keep task as 'None' unless prior task is unresolved.\n"
            f"Return JSON with:\n"
            f"- 'intent': 'teaching', 'request', or 'casual' (strongest).\n"
            f"- 'task': Current or new task (e.g., 'Analyze finances'), or 'None'. Keep prior task if relevant.\n"
            f"- 'keywords': 3-5 key words capturing focus.\n"
        )
        response = await asyncio.to_thread(LLM.invoke, prompt)
        try:
            result = json.loads(clean_llm_response(response.content.strip()))
            intent = result.get("intent", "casual")
            task = result.get("task", "None")
            keywords = result.get("keywords", [])
            
            # Persist pending_task unless resolved or explicitly new
            if pending_task["status"] == "probing" and intent in ["request", "casual"] and task == "None":
                task = pending_task["task"]  # Keep prior task if unresolved and no new task
                keywords = pending_task["keywords"]
                if intent == "casual":  # Adjust intent if it’s a follow-up
                    intent = "request"
            elif intent == "request" and task != "None":
                state["pending_task"] = {"task": task, "status": "probing", "keywords": keywords}
            elif intent == "teaching" and task != "None":
                state["pending_task"] = {"task": task, "status": "teaching", "keywords": keywords}
            elif pending_task["status"] != "resolved" and task == "None" and intent == "casual":
                state["pending_task"] = pending_task  # Carry forward unresolved task
            
            scores = {"teaching": 0.1, "request": 0.1, "casual": 0.1}
            scores[intent] = 0.8
            return scores, state
        except Exception as e:
            logger.warning(f"Failed to parse intent: {response.content}, defaulting to casual - Error: {str(e)}")
            return {"teaching": 0.1, "request": 0.1, "casual": 0.8}, state