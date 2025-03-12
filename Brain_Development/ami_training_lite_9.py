# ami_training_harvard_dynamic_v9
# Built by: The Fusion Lab with xAI's Grok 3
# Date: March 12, 2025
# Purpose: Test Ami's Training Path—dynamic LLM intent/topic/extraction, blueprint categories

import json
from langchain_openai import ChatOpenAI

# Setup
llm = ChatOpenAI(model="gpt-4o")
brain = []  # Mock storage as a list
convo_history = []  # Track convo for context

# Pickup Line Based on Brain State
def get_pickup_line(brain):
    if not brain:
        return "Hey, I’m Ami—sharp but green. What’s the one sales move I need to steal from you? "
    return "Hey, I’m Ami—got some tricks up my sleeve. What’s your next-level play? "

# Detect Intent (Dynamic LLM)
def detect_intent(msg):
    intent_prompt = f"""
    Analyze this message and return ONLY the intent as a quoted phrase:
    - Message: '{msg}'
    Possible Intents:
    - "casual": Greetings, farewells, or light chit-chat with no clear sales angle.
    - "teaching": Sharing sales-related info—like offering a sales technique, stating a product fact, or sharing a sales result.
    - If it’s about sales (e.g., suggesting a move, detailing a tool, or noting an outcome), it’s "teaching"—even if phrased casually.
    - Focus on the latest message; prior context only informs tone, not intent.
    - Keep it simple, no explanations—just the intent.
    """
    intent = llm.invoke(intent_prompt).content.strip()
    print(f"Debug: Intent detected = {intent}")
    return intent

# Detect Topic (Dynamic LLM)
def detect_topic(content, category):
    topic_prompt = f"""
    Infer a concise sales-related topic from this content and category, return ONLY the topic as a quoted phrase:
    - Content: '{content}'
    - Category: '{category}'
    Rules:
    - "skill": Techniques, especially questions or pitches (e.g., "ask" → "sales scripts").
    - "knowledge": Product facts (e.g., "CRM" → "CRM").
    - "lesson": Sales outcomes (e.g., "deals" → "sales performance").
    - Return "unknown" if unclear.
    - Keep it short, sales-focused, no fluff.
    """
    topic = llm.invoke(topic_prompt).content.strip()
    print(f"Debug: Topic detected for {category} = {topic}")
    return topic

# Extract Knowledge (Dynamic LLM)
def extract_knowledge(msg, history):
    extract_prompt = f"""
    Analyze *only this latest message* for sales-related info and categorize it:
    - Latest Message: '{msg}'
    - Convo History (last 3 turns, for context only): {history[-3:] if history else 'None'}
    Return a JSON object with:
    {{
        "skills": ["Sales techniques or questions, e.g., full phrasing of a suggested move"],
        "knowledge": ["Concrete product facts or details, e.g., how a tool works"],
        "lessons": ["Sales outcomes or insights, e.g., results or what worked"]
    }}
    - Focus on the latest message—don’t pull from history.
    - Categorize naturally: suggestions are skills, facts are knowledge, outcomes are lessons.
    - For skills, preserve the full technique phrasing when it’s a suggestion (e.g., "Ask 'How’s your team doing?'").
    - Empty lists if no match.
    - Return plain JSON, no ```json markers.
    """
    raw_response = llm.invoke(extract_prompt).content.strip()
    print(f"Debug: Raw extract response = {raw_response}")
    cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"Debug: JSON parse error = {e}")
        return {"skills": [], "knowledge": [], "lessons": []}

# Ami’s Reply
def ami_reply(msg, brain, convo_history, is_first=False):
    intent = detect_intent(msg)
    pickup = get_pickup_line(brain) if is_first else ""
    convo_history.append(msg)  # Add to history

    if intent == '"teaching"':
        extracted = extract_knowledge(msg, convo_history)
        # Store each category with its own topic
        for skill in extracted["skills"]:
            topic = detect_topic(skill, "skill")
            brain.append({"type": "skill", "topic": topic.strip('"'), "content": skill})
        for knowledge in extracted["knowledge"]:
            topic = detect_topic(knowledge, "knowledge")
            brain.append({"type": "knowledge", "topic": topic.strip('"'), "content": knowledge})
        for lesson in extracted["lessons"]:
            topic = detect_topic(lesson, "lesson")
            brain.append({"type": "lesson", "topic": topic.strip('"'), "content": lesson})
        
        # Build reply based on *all* extracted items
        reply_parts = []
        for skill in extracted["skills"]:
            reply_parts.append(f"Whoa, {skill}? That’s a slick move!")
        for knowledge in extracted["knowledge"]:
            reply_parts.append(f"Got it, {knowledge}—that’s key!")
        for lesson in extracted["lessons"]:
            reply_parts.append(f"Damn, {lesson}? That’s clutch!")
        reply = " ".join(reply_parts) or "Spill more—I’m hooked!"
        reply += " How’d you figure that out?"
    else:  # Casual chat
        casual_prompt = f"""
        Respond to '{msg}' like a sharp, charming Harvard girl—confident, curious, punchy.
        Keep it short, energetic, and toss in a fun follow-up question.
        No emojis—just words.
        """
        reply = llm.invoke(casual_prompt).content

    return f"{pickup}{reply}" if is_first else reply

# Test Loop
def run_convo(inputs):
    global convo_history
    convo_history = []  # Reset history
    print(f"Ami: {ami_reply('', brain, convo_history, is_first=True)}")
    for msg in inputs:
        print(f"You: {msg}")
        reply = ami_reply(msg, brain, convo_history, is_first=False)
        print(f"Ami: {reply}")
        print(f"Brain: {brain}")

# Test It
if __name__ == "__main__":
    test_inputs = [
        "Hey, good to meet you!",
        "Let’s ask 'How’s your team doing?' instead of 'How are you?'",
        "The CRM syncs in real-time, and last week we closed 10 deals!",
        "Gotta run, later!"
    ]
    run_convo(test_inputs)