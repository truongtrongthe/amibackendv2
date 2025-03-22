import logging
import ast
import re
from langchain_core.messages import HumanMessage
from utilities import detect_intent, clean_llm_response
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM = ChatOpenAI(model="gpt-4o", streaming=False)

def extract_knowledge(state, user_id=None, intent=None):
    messages = state["messages"][-5:] if state["messages"] else []
    latest_msg = messages[-1].content if messages else ""
    active_terms = state.get("active_terms", {})
    focus_history = state.get("focus_history", [])
    convo_id = state.get("convo_id", "test_convo")

    # Step 1: Identify the main term with focus chain and confidence
    logger.info(f"Step 1 - Identifying term for: '{latest_msg}'")
    intent = intent or detect_intent(state)
    term = None
    confidence = 1.0

    if latest_msg.strip():
        convo_history = " | ".join(m.content for m in messages[:-1]) if len(messages) > 1 else "None"
        focus_str = ", ".join([f"{t['term']} (score: {t['score']:.1f})" for t in focus_history]) if focus_history else "None"
        term_prompt = f"""Given:
- Latest message: '{latest_msg}'
- Prior messages: '{convo_history}'
- Intent: '{intent}'
- Focus history (term, relevance score): '{focus_str}'
What’s the main term or phrase this is about? Return just the term, nothing else. Examples:
- 'yo, got some dope info!' → 'None'
- 'hito granules—height growth booster!' → 'HITO Granules'
- 'Good—made by novahealth!' → 'HITO Granules' (if HITO Granules is high in focus)
- 'Thành Phần chính của nó là: Aquamin F...' → 'HITO Granules' (if HITO Granules is in focus)
- 'Ngoài ra còn có Collagen...' → 'HITO Granules' (if HITO Granules is in focus, implies extension)
Rules:
- Pick the subject or focus—usually a product, company, or key concept.
- Favor the highest-scored term in focus history unless the message explicitly defines a new subject (e.g., 'X is Y').
- If intent is 'teaching', assume continuity with the focus term unless overridden.
- If intent is 'request', return the top focus term or 'None' if empty.
- Only return 'None' if no term makes sense—err on picking something if possible."""
        
        try:
            term_response = LLM.invoke(term_prompt).content.strip()
            logger.info(f"Raw term response: '{term_response}'")
            term = clean_llm_response(term_response)
            logger.info(f"Cleaned term: '{term}'")
            if term.lower() == "none" or not term:
                term = None
            if intent == "request" and not term and focus_history:
                term = max(focus_history, key=lambda x: x["score"])["term"]
            if term and term.lower() not in [t["term"].lower() for t in focus_history] and intent == "teaching":
                words = latest_msg.lower().split()
                if term.lower() not in latest_msg.lower() or ("là" not in words and "is" not in words and focus_history):
                    term = max(focus_history, key=lambda x: x["score"])["term"]
                    confidence = 0.7
            if term and term.lower() in [t["term"].lower() for t in focus_history]:
                if "của nó" not in latest_msg.lower() and "là" not in latest_msg.lower() and term.lower() not in latest_msg.lower():
                    confidence = 0.7
            if term:
                found = False
                for focus in focus_history:
                    if focus["term"].lower() == term.lower():
                        focus["score"] = min(focus["score"] + 1.0, 5.0)
                        found = True
                        break
                if not found:
                    focus_history.append({"term": term, "score": 1.0})
                for focus in focus_history:
                    if focus["term"] != term:
                        focus["score"] = max(focus["score"] - 0.3, 0.0)
                focus_history = [f for f in focus_history if f["score"] > 0]
        except Exception as e:
            logger.error(f"LLM failed in Step 1: {e}")
            term = None if not focus_history else max(focus_history, key=lambda x: x["score"])["term"]
            confidence = 0.5

    logger.info(f"Identified term: '{term}' (intent: {intent}, confidence: {confidence:.1f}, focus_history: {focus_history})")

    # Step 2: Extract attributes and relationships dynamically
    attributes = []
    relationships = []
    if term and latest_msg.strip():
        focus_score = max([f["score"] for f in focus_history if f["term"] == term], default=0)
        if confidence < 0.8:
            return {
                "term": term,
                "confidence": confidence,
                "needs_confirmation": f"Is this still about '{term}'? Confirm or clarify the dominant term."
            }
        attr_prompt = f"""Given:
- Latest message: '{latest_msg}'
- Main term: '{term}' (focus score: {focus_score:.1f})
- Intent: '{intent}'
- Focus history (term, relevance score): '{focus_str}'
List descriptive phrases or properties directly about '{term}' in original Vietnamese. Return ONLY a Python list string, e.g., `["bổ sung canxi", "hỗ trợ phát triển chiều cao"]`. Rules:
- If '{term}' has a focus score >2.0, assume ALL message content describes it unless explicitly contradicted (e.g., 'X là Y' defines a new subject).
- Include anything that could describe '{term}'—e.g., features ("sản phẩm cao cấp"), ingredients as nouns ("canxi cá tuyết", "Collagen Type II"), benefits ("củng cố hệ xương"), details ("đường tự nhiên trong sữa mẹ"), or website URLs ("website: https://example.com").
- Use context and focus history—high score means '{term}' is the convo’s core, so link properties to it unless clearly about another term.
- Exclude verb-based relationships with external entities (e.g., "dành cho X", "được Y tin dùng").
- Keep it concise, combine related details (e.g., "Aquamin F (32% canxi)" instead of separate entries).
- Use original Vietnamese from the message, no translation.
- If nothing fits, return `[]`.
- No markdown, no code blocks, no text—just the list."""
        try:
            attr_response = LLM.invoke(attr_prompt).content.strip()
            logger.info(f"Raw attributes response: '{attr_response}'")
            cleaned_attr = clean_llm_response(attr_response)
            logger.info(f"Cleaned attributes: '{cleaned_attr}'")
            if cleaned_attr:
                try:
                    attributes = ast.literal_eval(cleaned_attr)
                    logger.info(f"Parsed attributes: {attributes}")
                    if not isinstance(attributes, list):
                        logger.warning(f"Attributes not a list: {attributes}")
                        attributes = []
                except (ValueError, SyntaxError):
                    match = re.search(r'\[(.*?)\]', cleaned_attr, re.DOTALL)
                    if match:
                        try:
                            attributes = ast.literal_eval(f'[{match.group(1)}]')
                            logger.info(f"Fallback parsed attributes: {attributes}")
                        except:
                            logger.warning(f"Failed fallback parsing: '{cleaned_attr}'")
                            attributes = []
                    else:
                        logger.warning(f"Failed to parse attributes: '{cleaned_attr}'")
                        attributes = []
            else:
                logger.info("No attributes after cleaning")
                attributes = []
        except Exception as e:
            logger.error(f"LLM failed in attributes extraction: {e}")
            attributes = []

        rel_prompt = f"""Given:
- Latest message: '{latest_msg}'
- Main term: '{term}' (focus score: {focus_score:.1f})
- Intent: '{intent}'
- Focus history (term, relevance score): '{focus_str}'
List relationships as a Python list of triples, e.g., `[["{term}", "dành cho", "Việt kiều 20-30"]]`. Return ONLY the list string—no markdown, no code blocks, no text. Rules:
- Identify entities EXTERNAL to '{term}' connected via verbs or prepositions (e.g., "dành cho X", "được Y tin dùng").
- Use original Vietnamese verbs: e.g., "hỗ trợ", "dành cho", "được".
- Exclude internal benefits or properties (e.g., "hỗ trợ phát triển chiều cao", "củng cố hệ xương").
- Format: `[subject, relation, object]`.
- If nothing fits, return `[]`."""
        try:
            rel_response = LLM.invoke(rel_prompt).content.strip()
            logger.info(f"Raw relationships response: '{rel_response}'")
            cleaned_rel = clean_llm_response(rel_response)
            logger.info(f"Cleaned relationships: '{cleaned_rel}'")
            if cleaned_rel:
                try:
                    relationships = ast.literal_eval(cleaned_rel)
                    logger.info(f"Parsed relationships: {relationships}")
                    if not isinstance(relationships, list) or not all(isinstance(r, list) and len(r) == 3 for r in relationships):
                        logger.warning(f"Relationships not a valid list of triples: {relationships}")
                        relationships = []
                except (ValueError, SyntaxError):
                    match = re.search(r'\[(.*?)\]', cleaned_rel, re.DOTALL)
                    if match:
                        try:
                            relationships = ast.literal_eval(f'[{match.group(1)}]')
                            logger.info(f"Fallback parsed relationships: {relationships}")
                        except:
                            logger.warning(f"Failed fallback parsing: '{cleaned_rel}'")
                            relationships = []
                    else:
                        logger.warning(f"Failed to parse relationships: '{cleaned_rel}'")
                        relationships = []
            else:
                logger.info("No relationships after cleaning")
                relationships = []
        except Exception as e:
            logger.error(f"LLM failed in relationships extraction: {e}")
            relationships = []

    logger.info(f"Extracted - Attributes: {attributes}, Relationships: {relationships}")

    state["focus_history"] = focus_history
    if term and term not in active_terms:
        active_terms[term] = {"term_id": f"node_{len(active_terms) + 1}", "vibe_score": 1.0}

    return {"term": term, "attributes": attributes, "relationships": relationships}

# Test with all 4 messages
if __name__ == "__main__":
    state = {
        "messages": [],
        "active_terms": {},
        "focus_history": [],
        "convo_id": "test_123",
        "intent_history": []
    }

    turns = [
        "HITO là sản phẩm bổ sung canxi hỗ trợ phát triển chiều cao (từ 2 tuổi trở lên, đăc biệt dành cho người trưởng thành),Đối tượng KH: Việt kiều 20-30 tuổi (cốm viên) Và mẹ có con từ 12-18 tuổi ở VN (sữa, thạch). Sản phẩm cao cấp, công thức toàn diện. Được đội ngũ chuyên viên đồng hành, cung cấp thông tin chuyên khoa, cá nhân hóa. Bộ tứ canxi hữu cơ kết hợp giúp hệ xương phát tri. ển toàn diện: Canxi cá tuyết, canxi tảo đỏ, canxi Gluconate, bột nhung hươu, ở trên bảng thành phần sp A+. Sản phẩm được CLB hàng đầu VN tín nhiệm và đưa vào chế độ dinh dưỡng cho các lứa cầu thủ chuyên nghiệp. Sản phẩm canxi duy nhất được CLB Bóng đá Hoàng Anh Gia Lai tin dùng. Website: https://hitovietnam.com/. Canxi cá tuyết: cá tuyết sống ở mực nước sâu hàng nghìn mét dưới mực nước biển nên có hệ xương vững chắc, mật độ xương cao. Theo chuyên gia Hito thì xương cá tuyết có cầu tạo gần giống hệ xương người, dồi dào canxi hữu cơ (gấp 9-10 lần canxi so với các nguồn khác), tương thích sinh học cao, tăng hấp thụ tối đa canxi vào xương",
        "Thành Phần chính của nó là: Aquamin F( 32% canxi , canxi từ tảo biển đỏ): Bổ sung canxin hữu cơ dễ hấp thu mà còn không lắng cặn, không bị nóng trong hay táo bón như canxin vô cơ .Củng cố hệ xương, bổ sung canxi giúp xương chắc khỏe, dẻo dai. bảo vệ và tham gia vào quá trình hình thành dịch nhầy ở khớp, giúp khớp chuyển động linh hoạt, thoải mái hơn.giúp ngăn ngừa việc hình thành khối u ở gan, polyp trực tràng. Đồng thời bảo vệ sức khỏe đường tiêu hóa",
        "Ngoài ra còn có Collagen Type II: Giúp tăng tế bào não , tăng vận động cho các khớp nối, giảm đau với bệnh viêm khớp mạn tính, phòng ngừa  và làm giảm viêm khớp dạng thấp, bảo vệ tim mạch, chống ăn mòn hoặc  chống đông máu mạnh mẽ ngăn ngừa các cục máu đông  giảm tỉ lệ đột quỵ.",
        "Phụ Liệu : Lactose , polyvinylpyrrolidone K30, Bột talc, Kali sorbat, Hương sữa vừa đủ 1 gói. Lactose là đường tự nhiên có trong thành phần của sữa mẹ, sữa bò, sữa động vật nên an toàn tuyệt đối cho sức khỏe.polyvinylpyrrolidone K30 là chất kết dính cho dạng hạt tồn tại dưới dạng màu trăng màu vàng nhạt có khả năng hấp thụ tốt"
    ]

    for i, msg in enumerate(turns, 1):
        state["messages"].append(HumanMessage(msg))
        intent = detect_intent(state)  # Define intent here for each turn
        result = extract_knowledge(state, intent=intent)
        focus_history = state.get("focus_history", [])  # Your tweak
        if result.get("needs_confirmation"):
            print(f"Turn {i}: {result['needs_confirmation']}")
            # Simulate human confirmation
            confirmation = input(f"Confirm '{result['term']}'? (yes/no): ").lower()
            if confirmation == "yes":
                result = extract_knowledge(state, intent=intent)  # Rerun with confirmed term and intent
            else:
                new_term = input("Enter the correct term: ")
                result = {"term": new_term, "attributes": [], "relationships": []}
                focus_history.append({"term": new_term, "score": 1.0})
                state["focus_history"] = focus_history
        print(f"Turn {i}: {result}")