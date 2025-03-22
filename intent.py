def detect_intent(state, user_id=None):
    messages = state["messages"][-3:] if state["messages"] else []
    convo_history = " | ".join([m.content for m in messages[:-1]]) if len(messages) > 1 else ""
    latest_msg = messages[-1].content if messages else ""
    last_ai_msg = state.get("prompt_str", "")
    active_terms = state.get("active_terms", {})
    intent_history = state.get("intent_history", [])

    if not latest_msg.strip():
        logger.info("Empty input, defaulting to casual")
        return "casual"

    latest_lower = latest_msg.lower()

    # Dual-language rule-based checks
    if "Is this still about" in last_ai_msg or "Confirm or clarify" in last_ai_msg:
        if latest_lower in ["yes", "yep", "correct", "có", "đúng", "phải"]:
            logger.info(f"Rule-based: Detected 'confirm' for '{latest_msg}'")
            return "confirm"
        if latest_lower in ["no", "nope", "wrong", "không", "sai"]:
            logger.info(f"Rule-based: Detected 'clarify' for '{latest_msg}'")
            return "clarify"

    if any(kw in latest_lower for kw in ["bye", "see ya", "later", "goodbye", "tạm biệt", "hẹn gặp"]):
        logger.info(f"Rule-based: Detected 'goodbye' for '{latest_msg}'")
        return "goodbye"

    if any(kw in latest_lower for kw in ["save", "forget", "delete", "add", "do it", "stop", "lưu", "xóa", "thêm", "làm", "dừng"]):
        logger.info(f"Rule-based: Detected 'command' for '{latest_msg}'")
        return "command"

    # Move teaching up to catch declarative statements first
    if not latest_lower.endswith("?") and any(term.lower() in latest_lower for term in active_terms.keys()):
        if any(kw in latest_lower for kw in ["là", "is", "có"]) and not any(kw in latest_lower for kw in ["không", "no", "sai"]):  # Declarative, not negation
            logger.info(f"Rule-based: Detected 'teaching' for '{latest_msg}'")
            return "teaching"

    if latest_lower.endswith("?") or "không" in latest_lower or "à" in latest_lower or "hả" in latest_lower:
        # Request
        if any(kw in latest_lower for kw in ["tell", "about", "what is", "describe", "explain", "what’s", "how’s", "nói", "về", "là gì", "mô tả", "giải thích", "thế nào", "ra sao", "bao nhiêu"]):
            if any(term.lower() in latest_lower for term in active_terms.keys()) or any(kw in latest_lower for kw in ["weather", "time", "news", "thời tiết", "giờ", "tin tức"]):
                logger.info(f"Rule-based: Detected 'request' for '{latest_msg}'")
                return "request"
        # Confirmation (tighter rules)
        if any(kw in latest_lower for kw in ["does", "can", "are", "will", "có thể", "được", "sẽ"]) or "sure" in latest_lower or "chắc" in latest_lower:
            if any(term.lower() in latest_lower for term in active_terms.keys()):
                logger.info(f"Rule-based: Detected 'confirmation' for '{latest_msg}'")
                return "confirmation"
        # Asking (AI)
        if any(kw in latest_lower for kw in ["you", "can you", "what can", "how do you", "are you", "bạn", "có thể", "bạn làm gì", "làm sao", "bạn có"]):
            if not any(term.lower() in latest_lower for term in active_terms.keys()):
                logger.info(f"Rule-based: Detected 'asking' for '{latest_msg}'")
                return "asking"

    if any(kw in latest_lower for kw in ["no", "not", "wrong", "actually", "không", "chẳng", "sai", "thật ra"]) and active_terms and last_ai_msg:
        logger.info(f"Rule-based: Detected 'correction' for '{latest_msg}'")
        return "correction"

    if any(kw in latest_lower for kw in ["maybe", "might", "có thể", "có lẽ"]):
        logger.info(f"Rule-based: Detected 'teaching' for '{latest_msg}'")
        return "teaching"

    if latest_msg.endswith("!") and not latest_lower.endswith("?") and any(kw in latest_lower for kw in ["wow", "cool", "great", "hate", "love", "ôi", "tuyệt", "hết sức", "ghét", "thích"]):
        logger.info(f"Rule-based: Detected 'emotional' for '{latest_msg}'")
        return "emotional"

    # LLM fallback (unchanged)
    prompt = f"""Given:
    - Latest message: '{latest_msg}'
    - Last 2 messages: '{convo_history}'
    - Last AI reply: '{last_ai_msg}'
    - Active terms: {list(active_terms.keys())}
    - Intent history: {intent_history[-2:]}
    Classify as 'teaching', 'request', 'asking', 'casual', 'correction', 'confirmation', 'command', 'emotional', or 'goodbye'. Return ONLY JSON: {{"intent": "teaching", "confidence": 0.95}}.
    Rules:
    - 'teaching': Adds knowledge (e.g., "HITO boosts height", "Nó có thể từ Nhật"). Declarative, tentative ("maybe", "có thể" → 0.5-0.7).
    - 'request': Asks for info (e.g., "Tell me about HITO", "Thời tiết thế nào?"). Questions with topics or general (weather, time).
    - 'asking': Queries AI (e.g., "What can you do?", "Bạn làm gì được?"). "You"/"bạn" focus, no topics.
    - 'casual': Chit-chat (e.g., "Hey!", "Ngày đẹp nhỉ"). Neutral tone.
    - 'correction': Fixes info (e.g., "No, HITO’s from Korea"). Negation after AI reply.
    - 'confirmation': Yes/no check (e.g., "Is HITO good?"). "Is"/"có" questions or "yes"/"no" after confirmation prompt.
    - 'command': Orders AI (e.g., "Save this"). Imperatives.
    - 'emotional': Feelings (e.g., "Wow, cool!"). Exclamations.
    - 'goodbye': Ends convo (e.g., "Bye")."""
    
    response = clean_llm_response(LLM.invoke(prompt).content)
    try:
        result = json.loads(response)
        intent = result["intent"]
        confidence = result["confidence"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Intent parse error: {e}. Raw: '{response}'")
        json_match = re.search(r'\{.*"intent":\s*"([^"]+)".*"confidence":\s*([0-9.]+).*\}', response, re.DOTALL)
        if json_match:
            intent = json_match.group(1)
            confidence = float(json_match.group(2))
        else:
            intent = "casual"
            confidence = 0.5

    logger.info(f"Final intent: '{intent}' (confidence: {confidence})")
    state["intent_history"] = intent_history + [intent]
    return intent