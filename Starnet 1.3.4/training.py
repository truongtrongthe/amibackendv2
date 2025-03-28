# pretrain.py
# Built by: The Fusion Lab
# Date: March 28, 2025

from langchain_openai import ChatOpenAI
from utilities import logger, clean_llm_response, EMBEDDINGS
import asyncio
import json
from langchain_core.messages import HumanMessage
from datetime import datetime
from typing import Dict, List, Tuple
from pinecone_datastores import load_ami_brain, save_training, load_convo_history
from pinecone_datastores import load_character_traits, infer_categories
import uuid
import numpy as np


def cosine_similarity(vec1, vec2):
    """Fallback cosine similarity function."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class Training:
    def __init__(self, user_id: str = "thefusionlab", mode: str = "pretrain"):
        self.user_id = user_id
        self.mode = mode
        self.MyLLM = ChatOpenAI(model="gpt-4o", streaming=False)  # Non-streaming LLM
        traits = load_character_traits("thefusionlab")
        if isinstance(traits, str):
            try:
                traits = json.loads(traits)
            except (json.JSONDecodeError, TypeError):
                traits = {}
        self.state = {
            "messages": [],
            "prompt_str": "",
            "convo_id": f"conv_{uuid.uuid4()}",
            "user_id": self.user_id,
            "intent_history": [],
            "preset_memory": load_ami_brain("thefusionlab"),
            "character_traits": traits if isinstance(traits, dict) else {},
            "embedding_cache": {},
            "pending_task": {"task": "None", "status": "idle", "keywords": [], "definitions": {}}
        }

    async def training(self, state: Dict = None, user_id: str = None) -> Dict:
        """Main training method with full context and structured trait saving."""
        logger.debug(f"Starting training - Mode: {self.mode}")
        state = state or self.state
        user_id = user_id or state.get("user_id", "thefusionlab")
        
        latest_msg = state["messages"][-1] if state["messages"] else ""
        latest_msg_content = latest_msg.content if isinstance(latest_msg, HumanMessage) else latest_msg
        
        if latest_msg_content and latest_msg_content not in state["embedding_cache"]:
            state["embedding_cache"][latest_msg_content] = EMBEDDINGS.embed_query(latest_msg_content)
        latest_embedding = state["embedding_cache"].get(latest_msg_content, None)
        if not latest_embedding and state["messages"]:
            state["messages"] = state["messages"][-50:]

        if len(state["messages"]) > 50 and latest_embedding:
            intent_scores, state = await self.detect_training_intent(state)
            threshold = (
                0.3 if intent_scores.get("casual", 0) > 0.5 else
                0.25 if intent_scores.get("request", 0) > 0.5 else 0.2
            )
            relevant_msgs = []
            total_tokens = len(latest_msg_content.split())
            for msg in state["messages"][-50:-1]:
                msg_content = msg.content if isinstance(msg, HumanMessage) else msg
                if msg_content not in state["embedding_cache"]:
                    state["embedding_cache"][msg_content] = EMBEDDINGS.embed_query(msg_content)
                similarity = cosine_similarity(latest_embedding, state["embedding_cache"][msg_content])
                msg_tokens = len(msg_content.split())
                if (similarity >= threshold or intent_scores.get("teaching", 0) >= 0.5) and total_tokens < 4000:
                    relevant_msgs.append(msg)
                    total_tokens += msg_tokens
                elif similarity < 0.1:
                    break
            state["messages"] = relevant_msgs + [latest_msg]

        intent_scores, state = await self.detect_training_intent(state)
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        max_intent = max(intent_scores, key=intent_scores.get)
        logger.info(f"Intent scores: {intent_scores}", extra={"user_id": user_id})

        context_lines = [msg.content if isinstance(msg, HumanMessage) else msg for msg in state["messages"]]
        context = "\n".join(context_lines)

        if max_intent == "teaching":
            await save_training(latest_msg_content, user_id)
            convo_history = load_convo_history(latest_msg_content, user_id)
            categories = await infer_categories(latest_msg_content, context)
            
            if not isinstance(state["character_traits"], dict):
                logger.warning(f"character_traits was {type(state['character_traits'])}, resetting to dict")
                state["character_traits"] = {}
            
            if state["pending_task"]["task"] in ["Define trait", "Apply trait"]:
                if state["pending_task"]["definitions"]:
                    for trait_key, trait_value in state["pending_task"]["definitions"].items():
                        if trait_key not in state["character_traits"]:
                            state["character_traits"][trait_key] = {"definition": "", "instruction": ""}
                        if state["pending_task"]["task"] == "Define trait":
                            state["character_traits"][trait_key]["definition"] = trait_value
                        elif state["pending_task"]["task"] == "Apply trait":
                            state["character_traits"][trait_key]["instruction"] = trait_value
                        logger.info(f"Updated character trait: {trait_key} = {state['character_traits'][trait_key]}")
                    state["pending_task"]["definitions"] = {}

            prompt_parts = [
                "You're Ami, a smart co-pilot speaking natural Vietnamese. You understand human is Teaching you",
                f"Character traits: {json.dumps(state.get('character_traits', 'No character traits yet.'), ensure_ascii=False)}",
                f"Preset wisdom: {state['preset_memory']}",
                f"Conversation so far: {context}",
                f"Enterprise wisdom: {convo_history}",
                f"Latest message: '{latest_msg_content}'",
                "Task: Show you understand the trait deeply, incorporating both its definition and how to apply it, and respond naturally with Thanks to the trainer—keep it sharp, prioritizing enterprise wisdom."
            ]
            prompt = "\n".join(prompt_parts)
        else:
            prompt_parts = [
                "You're Ami, a chill Vietnamese buddy—think laid-back friend, not stiff robot.",
                f"Preset wisdom: {state['preset_memory']}",
                f"Conversation so far: {context}",
                f"Latest: '{latest_msg_content}'",
                "Task: Vibe back naturally, keep it light and short. No 'Xin Chào!' unless it’s the first chat. "
                "Use casual Vietnamese—like 'Ừm,' 'Thiệt hả,' or 'Chill đi'—and mix it up."
            ]
            prompt = "\n".join(prompt_parts)

        try:
            response = await asyncio.to_thread(self.MyLLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()
        except Exception as e:
            logger.error(f"Failed to invoke MyLLM: {str(e)}", extra={"user_id": user_id})
            state["prompt_str"] = "Mình gặp lỗi khi xử lý, thử lại nhé!"
        
        logger.info(f"Response: {state['prompt_str']}", extra={"user_id": user_id})
        self.state = state
        return state

    async def detect_training_intent(self, state: Dict) -> Tuple[Dict, Dict]:
        """Detect intent using MyLLM, ensuring full definition extraction."""
        recent_msgs = state["messages"][-5:] if len(state["messages"]) >= 5 else state["messages"]
        context = "\n".join(msg.content if isinstance(msg, HumanMessage) else str(msg) for msg in recent_msgs)
        latest_msg = state["messages"][-1].content if state["messages"] else ""
        pending_task = state.get("pending_task", {"task": "None", "status": "idle", "keywords": [], "definitions": {}})

        prompt_parts = [
            f"Conversation: {context}",
            f"Latest: '{latest_msg}'",
            "Analyze the intent of the latest message within the full conversation:",
            "- If it defines a term that could be a character trait (e.g., 'Curiosity is asking actively about unknown topics' or a detailed explanation of a curious person), classify as 'teaching' and extract the term and the full definition from the conversation.",
            "- If it instructs the AI to adopt a behavior or trait (e.g., 'Be curious', 'Act bold', or 'Always show curiosity') and a prior definition exists or it implies a trait, classify as 'teaching'.",
            "- If it builds on a prior task (e.g., provides details like income for 'Analyze finances'), classify as 'request'.",
            "- If it's chit-chat or unclear, classify as 'casual'.",
            "Return JSON with:",
            "- 'intent': 'teaching', 'request', or 'casual' (strongest).",
            "- 'task': 'Define trait' for definitions, 'Apply trait' for instructions, or 'None'.",
            "- 'keywords': 3-5 key words capturing focus.",
            "- 'definition': {term: full_definition_or_instruction} if a term is defined or implied (use the full relevant text from the conversation), else {}."
        ]
        prompt = "\n".join(prompt_parts)

        try:
            response = await asyncio.to_thread(self.MyLLM.invoke, prompt)
            result = json.loads(clean_llm_response(response.content.strip()))
        except Exception as e:
            logger.warning(f"Failed to detect intent: {str(e)}")
            return {"teaching": 0.1, "request": 0.1, "casual": 0.8}, state

        intent = result.get("intent", "casual")
        task = result.get("task", "None")
        keywords = result.get("keywords", [])
        definition = result.get("definition", {})

        if intent == "teaching":
            if task in ["Define trait", "Apply trait"] and definition:
                state["pending_task"]["task"] = task
                state["pending_task"]["definitions"].update(definition)
            # Ensure first message is processed as definition if not yet set
            elif task == "None" and not state["character_traits"].get("curiosity", {}).get("definition"):
                if "tò mò" in latest_msg.lower() or "ham học hỏi" in latest_msg.lower():
                    state["pending_task"]["task"] = "Define trait"
                    state["pending_task"]["definitions"] = {"curiosity": latest_msg}

        if pending_task["status"] == "probing" and intent in ["request", "casual"] and task == "None":
            task = pending_task["task"]
            keywords = pending_task["keywords"]
            if intent == "casual":
                intent = "request"
        elif intent in ["request", "teaching"] and task != "None":
            state["pending_task"]["status"] = "teaching" if intent == "teaching" else "probing"
            state["pending_task"]["keywords"] = keywords

        scores = {"teaching": 0.1, "request": 0.1, "casual": 0.1}
        scores[intent] = 0.8
        return scores, state


# Example usage
if __name__ == "__main__":
    async def main():
        trainer = Training()
        trainer.state["messages"] = [
            HumanMessage(content="Một người tò mò, ham học hỏi là người luôn có khao khát tìm hiểu, khám phá những điều mới và không ngừng nâng cao kiến thức, kỹ năng của mình. Họ không chấp nhận những câu trả lời hời hợt mà luôn muốn đào sâu vấn đề để hiểu rõ bản chất.\n\nBiểu hiện của người tò mò, ham học hỏi:\n🔎 1. Luôn đặt câu hỏi \"Tại sao?\" và \"Như thế nào?\"\nHọ không dễ dàng chấp nhận mọi thứ theo cách nó vốn có, mà luôn muốn hiểu sâu hơn.\n\nVí dụ: Khi thấy một công nghệ mới, họ không chỉ hỏi “Nó hoạt động như thế nào?” mà còn hỏi “Tại sao nó lại hiệu quả hơn cái cũ?”\n\n📖 2. Chủ động tìm kiếm kiến thức mới\nHọ không chờ ai đó dạy mà tự mình khám phá, đọc sách, học hỏi từ nhiều nguồn khác nhau.\n\nHọ thích thử nghiệm những điều mới, không ngại bước ra khỏi vùng an toàn.\n\n🎯 3. Học từ thất bại, không ngại thử thách\nKhi gặp khó khăn hoặc thất bại, họ không nản lòng mà xem đó là cơ hội để học hỏi.\n\nHọ phân tích lỗi sai, tìm cách cải thiện và làm tốt hơn vào lần sau.\n\n👂 4. Luôn lắng nghe và cởi mở với ý kiến mới\nHọ không bảo thủ mà sẵn sàng tiếp nhận quan điểm khác, dù nó trái ngược với suy nghĩ ban đầu của họ.\n\n🚀 5. Đặt mình vào môi trường có nhiều thách thức\nHọ chủ động tham gia vào những công việc hoặc dự án khó để học hỏi nhanh hơn.\n\n💡 Ví dụ thực tế:\nMột nhân viên bán hàng ham học hỏi không chỉ làm theo kịch bản có sẵn mà sẽ:\n\nQuan sát cách khách hàng phản ứng để điều chỉnh cách giao tiếp.\n\nĐọc thêm về tâm lý khách hàng để hiểu nhu cầu sâu hơn.\n\nHọc hỏi từ những người bán hàng giỏi hơn để nâng cao kỹ năng của mình.\n\n📌 Tóm lại: Người tò mò, ham học hỏi luôn muốn hiểu sâu vấn đề, không ngại khó khăn, sẵn sàng học từ bất kỳ ai và luôn tìm cách phát triển bản thân."),
            HumanMessage(content="Luôn luôn thể hiện mình là người tò mò trong lúc giao tiếp với người khác")
        ]
        state = await trainer.training()
        print(f"Response: {state['prompt_str']}")
        print(f"Character traits: {state['character_traits']}")

    asyncio.run(main())