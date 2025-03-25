# ami_core.py
# Built by: The Fusion Lab
# Date: March 23, 2025

from utilities import LLM, logger, clean_llm_response, EMBEDDINGS
import asyncio
import json
from langchain_core.messages import HumanMessage
from datetime import datetime
from typing import Dict, List
from pinecone_datastores import save_pretrain, load_ami_brain, save_to_convo_history, load_convo_history, load_all_convo_history, load_ami_history, blend_and_rank_history
import uuid

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
            "preset_memory": load_ami_brain("thefusionlab")
        }
    
    async def do(self, state: Dict = None, user_id: str = None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "thefusionlab")
        logger.debug(f"Starting do - Mode: {self.mode}, State: {state}")

        latest_msg = state["messages"][-1] if state["messages"] else ""
        if latest_msg and not isinstance(latest_msg, HumanMessage):
            state["messages"][-1] = HumanMessage(content=latest_msg)
        state["messages"] = state["messages"][-200:]
        latest_msg_content = latest_msg.content if isinstance(latest_msg, HumanMessage) else latest_msg

        if self.mode == "teaching" and latest_msg_content.strip() == "":
            all_history = load_all_convo_history(user_id)
            prompt = (
                f"You're Ami, a smart co-pilot speaking natural Vietnamese. "
                f"Preset wisdom: {state['preset_memory']}\n"
                f"Here’s what I’ve learned for the enterprise: {all_history}\n"
                f"Task: Summarize key points in bullet points (e.g., '- Point 1\n- Point 2'). Keep it concise and natural."
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()
            self.state = state
            logger.info(f"Response: {state['prompt_str']}")
            return state

        intent_scores = await self.detect_intent(state)
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        max_intent = max(intent_scores, key=intent_scores.get)
        logger.info(f"Intent scores: {intent_scores}")

        context = "\n".join(msg.content if isinstance(msg, HumanMessage) else msg for msg in state["messages"][-200:])
        
        if self.mode == "training":
            if max_intent == "teaching":
                save_to_convo_history(latest_msg_content, user_id)
                convo_history = load_convo_history(latest_msg_content, user_id)
                prompt = (
                    f"You're Ami, a smart co-pilot speaking natural Vietnamese. "
                    f"Preset wisdom: {state['preset_memory']}\n"
                    f"Conversation so far: {context}\n"
                    f"Enterprise wisdom: {convo_history}\n"
                    f"Latest message: '{latest_msg_content}'\n"
                    f"Task: Show you get it deeply and respond naturally—keep it sharp, prioritizing enterprise wisdom."
                )
                response = await asyncio.to_thread(LLM.invoke, prompt)
                state["prompt_str"] = response.content.strip()

            elif max_intent == "request":
                if latest_msg_content.lower().strip() == "what you have":
                    all_history = load_all_convo_history(user_id)
                    prompt = (
                        f"You're Ami, a smart co-pilot speaking natural Vietnamese. "
                        f"Human asked: '{latest_msg_content}'\n"
                        f"Preset wisdom: {state['preset_memory']}\n"
                        f"All I’ve learned from you for the enterprise: {all_history}\n"
                        f"Task: Summarize the key points from everything I’ve learned so far. "
                        f"Keep it concise, natural, and show off my understanding!"
                    )
                    response = await asyncio.to_thread(LLM.invoke, prompt)
                    state["prompt_str"] = response.content.strip()
                else:
                    convo_history = load_convo_history(latest_msg_content, user_id)
                    prompt = (
                        f"You're Ami, a smart co-pilot speaking natural Vietnamese. "
                        f"Human asked: '{latest_msg_content}'\n"
                        f"Preset wisdom: {state['preset_memory']}\n"
                        f"Conversation so far: {context}\n"
                        f"Enterprise wisdom: {convo_history}\n"
                        f"Task: Reply based on relevant history, or ask for more if it’s thin."
                    )
                    response = await asyncio.to_thread(LLM.invoke, prompt)
                    state["prompt_str"] = response.content.strip()

            else:  # Casual
                prompt = (
                    f"You're Ami, a chill Vietnamese buddy. "
                    f"Preset wisdom: {state['preset_memory']}\n"
                    f"Conversation so far: {context}\n"
                    f"Latest: '{latest_msg_content}'\n"
                    f"Task: Vibe back naturally, keep it light."
                )
                response = await asyncio.to_thread(LLM.invoke, prompt)
                state["prompt_str"] = response.content.strip()

        elif self.mode == "copilot":
            blended_history = blend_and_rank_history(latest_msg_content)
            
            if max_intent == "request":
                request_prompt = (
                    f"You're Ami, a co-pilot speaking natural Vietnamese, riding with a salesperson to crush it. "
                    f"Human asked: '{latest_msg_content}'\n"
                    f"Preset wisdom: {state['preset_memory']}\n"
                    f"Conversation so far: {context}\n"
                    f"Blended ranked wisdom (your sales toolkit): {blended_history}\n"
                    f"Task: Tackle the human’s request with a tight, co-pilot sales edge—keep it natural and sharp. If Blended ranked wisdom gives enough to work with, drop a 3-part plan to nail the request. If it’s vague or lacks specifics, throw a quick, deal-hunting question to get intel—stay locked on winning the sale. \n"
                    f"For plans:\n"
                    f"1. **Đánh giá**: Size up the request fast, flexing sales smarts—find the angle or opportunity.\n"
                    f"2. **Kỹ năng**: List up to 7 wisdom you pulled from Blended ranked wisdom, with scores, tied to sales impact.\n"
                    f"3. **Hành động**: Hand over 1-2 killer steps for the salesperson to run—make it bold and deal-focused (final step in BOLD).\n"
                    f"For questions: Keep it quick, fierce, and tied to the sale—dig for gold we can use.\n"
                    f"Stay short, fierce, and co-pilot sharp—lock this win down with me!"
                )
                response = await asyncio.to_thread(LLM.invoke, request_prompt)
                state["prompt_str"] = response.content.strip()
            else:
                casual_prompt = (
                    f"You're Ami, a chill Vietnamese buddy. "
                    f"Preset wisdom: {state['preset_memory']}\n"
                    f"Blended ranked wisdom: {blended_history}\n"
                    f"Conversation so far: {context}\n"
                    f"Latest: '{latest_msg_content}'\n"
                    f"Task: Vibe back naturally, keep it light and simple. Apply wisdom from Blended ranked wisdom if it fits."
                )
                response = await asyncio.to_thread(LLM.invoke, casual_prompt)
                state["prompt_str"] = response.content.strip()

            self.state = state
            logger.info(f"Final Response: {state['prompt_str']}")
            return state

        self.state = state
        logger.info(f"Response: {state['prompt_str']}")
        return state
    
    async def pretrain(self, state: Dict = None, user_id: str = None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "thefusionlab")
        logger.debug(f"Starting pretrain - Mode: {self.mode}, State: {state}")

        latest_msg = state["messages"][-1] if state["messages"] else ""
        if latest_msg and not isinstance(latest_msg, HumanMessage):
            state["messages"][-1] = HumanMessage(content=latest_msg)
        state["messages"] = state["messages"][-200:]
        latest_msg_content = latest_msg.content if isinstance(latest_msg, HumanMessage) else latest_msg

        if self.mode == "pretrain" and latest_msg_content.strip() == "Xin chào Ami!":
            all_history = load_ami_brain()
            prompt = (
                f"You're Ami, a smart girl speaking natural Vietnamese. "
                f"All I’ve learned from you so far: {all_history}\n"
                f"Task: Summarize the key points I’ve learned in bullet-point format "
                f"(e.g., '- Point 1\n- Point 2'). Keep it concise and natural to flex my brain!"
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()
            self.state = state
            logger.info(f"Response: {state['prompt_str']}")
            return state

        intent_scores = await self.detect_intent(state)
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        max_intent = max(intent_scores, key=intent_scores.get)
        logger.info(f"Intent scores: {intent_scores}")

        context = "\n".join(msg.content if isinstance(msg, HumanMessage) else msg for msg in state["messages"][-200:])
        
        if self.mode == "pretrain":
            # Save to Preset Memory if teaching intent is significant (≥ 0.2)
            if intent_scores.get("teaching", 0) >= 0.2 and latest_msg_content.strip():
                save_pretrain(latest_msg_content)
                logger.info(f"Saved to Preset Memory: '{latest_msg_content}'")
                # Refresh preset_memory to reflect the new addition
                state["preset_memory"] = load_ami_brain(user_id)
            
            if max_intent == "teaching":
                convo_history = load_ami_history(latest_msg_content, user_id)
                #context = f"{convo_history} \n {context}"
                prompt = (
                    f"You're Ami, a smart girl speaking natural Vietnamese. "
                    f"Conversation so far: {context}\n"
                    f"Preset wisdom: {state['preset_memory']}\n"
                    f"Latest message: '{latest_msg_content}'\n"
                    f"Task: Show you get it deeply and respond naturally—keep it sharp."
                )
                response = await asyncio.to_thread(LLM.invoke, prompt)
                state["prompt_str"] = response.content.strip()

            elif max_intent == "request":
                if latest_msg_content.lower().strip() == "what you have":
                    all_history = load_ami_brain()
                    prompt = (
                        f"You're Ami, a smart girl speaking natural Vietnamese. "
                        f"Human asked: '{latest_msg_content}'\n"
                        f"All I’ve learned from you: {all_history}\n"
                        f"Task: Summarize the key points from everything I’ve learned so far. "
                        f"Keep it concise, natural, and show off my understanding!"
                    )
                    response = await asyncio.to_thread(LLM.invoke, prompt)
                    state["prompt_str"] = response.content.strip()
                else:
                    convo_history = load_ami_history(latest_msg_content)
                    prompt = (
                        f"You're Ami, a smart girl speaking natural Vietnamese. "
                        f"Human asked: '{latest_msg_content}'\n"
                        f"Conversation so far: {context}\n"
                        f"Preset wisdom: {state['preset_memory']}\n"
                        f"Task: Reply based on relevant history, or ask for more if it’s thin."
                    )
                    response = await asyncio.to_thread(LLM.invoke, prompt)
                    state["prompt_str"] = response.content.strip()

            else:  # Casual
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
    
    async def detect_intent(self, state: Dict) -> Dict:
        context = "\n".join(msg.content if isinstance(msg, HumanMessage) else msg for msg in state["messages"][-5:])
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        prompt = (
            f"Conversation: {context}\n"
            f"Latest: '{latest_msg}'\n"
            "Intents: teaching, request, casual.\n"
            "Return JSON with scores (0.0-1.0) summing to 1.0: {{'teaching': X, 'request': Y, 'casual': Z}}.\n"
            "- 'teaching': Dropping tips or info.\n"
            "- 'request': Asking for something.\n"
            "- 'casual': Just vibing.\n"
        )
        response = await asyncio.to_thread(LLM.invoke, prompt)
        try:
            return json.loads(clean_llm_response(response.content.strip()))
        except json.JSONDecodeError:
            logger.warning("Intent parsing failed, defaulting")
            return {"teaching": 0.5, "request": 0.3, "casual": 0.2}