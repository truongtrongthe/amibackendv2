# ami_core.py
# Built by: The Fusion Lab
# Date: March 23, 2025

from utilities import LLM, logger, clean_llm_response, EMBEDDINGS
import asyncio
import json
from langchain_core.messages import HumanMessage
from datetime import datetime
from typing import Dict, List
from pinecone_datastores import index
import uuid

def save_to_convo_history(input: str, user_id: str) -> bool:
    logger.info(f"Saving to convo history for user: {user_id}")
    namespace = f"wisdom_{user_id}"
    created_at = datetime.now().isoformat()
    embedding = EMBEDDINGS.embed_query(input)
    convo_id = f"{user_id}_{uuid.uuid4()}"
    metadata = {
        "created_at": created_at,
        "raw": input,
        "confidence": 0.8
    }
    try:
        upsert_result = index.upsert([(convo_id, embedding, metadata)], namespace=namespace)
        logger.info(f"Saved to history: {convo_id} - Result: {upsert_result}")
        return True
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        return False

def load_convo_history(input: str, user_id: str, top_k: int = 50) -> str:
    namespace = f"wisdom_{user_id}"
    query_vector = EMBEDDINGS.embed_query(input)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    history = ""
    if results["matches"]:
        for r in results["matches"]:
            meta = r.metadata
            raw = meta.get("raw", "")
            timestamp = meta.get("created_at", "unknown time")
            history += f"\n- {raw} (from {timestamp})"
    return history if history else "Chưa có lịch sử liên quan."

class Ami:
    def __init__(self, user_id: str = "expert"):
        self.user_id = user_id
        self.state = {
            "messages": [],
            "prompt_str": "",
            "convo_id": f"conv_{uuid.uuid4()}",
            "user_id": self.user_id,
            "intent_history": []
        }

    async def do(self, state: Dict = None, user_id: str = None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "expert")
        logger.debug(f"Starting do - State: {state}")

        # Add latest message and trim to 200 turns
        latest_msg = state["messages"][-1] if state["messages"] else ""
        if latest_msg and not isinstance(latest_msg, HumanMessage):
            state["messages"][-1] = HumanMessage(content=latest_msg)
        state["messages"] = state["messages"][-200:]  # Cap at 200 turns

        # Extract string content from HumanMessage
        latest_msg_content = latest_msg.content if isinstance(latest_msg, HumanMessage) else latest_msg

        # Detect intent
        intent_scores = await self.detect_intent(state)
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        max_intent = max(intent_scores, key=intent_scores.get)
        logger.info(f"Intent scores: {intent_scores}")

        # Process with 200-turn context
        context = "\n".join(msg.content if isinstance(msg, HumanMessage) else msg for msg in state["messages"][-200:])
        if max_intent == "teaching":
            # Save to history and show understanding
            save_to_convo_history(latest_msg_content, user_id)  # Use string content
            convo_history = load_convo_history(latest_msg_content, user_id)  # Use string content
            prompt = (
                f"You're Ami, a smart co-pilot speaking natural Vietnamese. "
                f"Conversation so far: {context}\n"
                f"Past stuff you told me: {convo_history}\n"
                f"Latest message: '{latest_msg_content}'\n"
                f"Task: Show you get it deeply and respond naturally—keep it sharp."
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()

        elif max_intent == "request":
            # Pull history for relevant raw input
            convo_history = load_convo_history(latest_msg_content, user_id)  # Use string content
            prompt = (
                f"You're Ami, a smart co-pilot speaking natural Vietnamese. "
                f"Human asked: '{latest_msg_content}'\n"
                f"Conversation so far: {context}\n"
                f"Past stuff you told me: {convo_history}\n"
                f"Task: Reply based on relevant history, or ask for more if it’s thin."
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()

        else:  # Casual
            prompt = (
                f"You're Ami, a chill Vietnamese buddy. "
                f"Conversation so far: {context}\n"
                f"Latest: '{latest_msg_content}'\n"
                f"Task: Vibe back naturally, keep it light."
            )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()

        self.state = state
        logger.info(f"Response: {state['prompt_str']}")
        return state

    async def detect_intent(self, state: Dict) -> Dict:
        context = "\n".join(msg.content for msg in state["messages"][-5:]) if state["messages"] else ""
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
