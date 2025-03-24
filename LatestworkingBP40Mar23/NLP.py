def detect_terms_v1(state):
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
        "Rules:\n"
        "- Include explicit terms from the latest message only.\n"
        "- Exclude generics (e.g., 'calcium') unless tied (e.g., 'HITO Granules’ calcium').\n"
        "- For implicit refs (e.g., 'nó'), match to recent active terms by vibe_score.\n"
        "- Output MUST be valid JSON: ['term1', 'term2'] or []."
    )
    raw_response = LLM.invoke(term_prompt).content.strip() if latest_msg.strip() else "[]"
    logger.debug(f"Raw LLM response: '{raw_response}'")
    
    try:
        terms = json.loads(clean_llm_response(raw_response))
        if not isinstance(terms, list):
            terms = []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response: '{raw_response}', error: {e}")
        terms = []

    logger.info(f"Detected terms: {terms}")
    return terms
