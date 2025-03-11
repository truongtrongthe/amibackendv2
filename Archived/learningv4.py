def ami_node(state: State):
    latest_message = state["messages"][-1].content
    meat_points = state.get("meat_points", 0)
    pending_save = state.get("pending_save", {})
    teach_streak = state.get("teach_streak", False)

    if len(state["messages"]) == 1:
        return {"prompt_str": "Yo, I’m AMI—whatcha vibing on today?", "meat_points": 0, "teach_streak": False}

    # Handle confirmation
    if pending_save and "summary" in pending_save:
        if "yep" in latest_message.lower() or "yeah" in latest_message.lower():
            return {
                "prompt_str": f"Saved '{pending_save['summary']}' (skills)—feeding me gold, huh?",
                "meat_points": 0,  # Reset after save
                "pending_save": {},
                "teach_streak": True
            }
        elif "nah" in latest_message.lower() or "no" in latest_message.lower():
            return {
                "prompt_str": f"Alright, tweak it—whatcha really saying?",
                "meat_points": meat_points,
                "pending_save": {},
                "teach_streak": True
            }
        else:
            return {
                "prompt_str": f"No tweaks? Saved '{pending_save['summary']}' (skills)—what’s next?",
                "meat_points": 0,  # Reset after save
                "pending_save": {},
                "teach_streak": True
            }

    intent, total_meat, teach_streak = detect_intent(state)

    # Topic change check (simple heuristic for now)
    last_user = next((msg for msg in reversed(state["messages"][:-1]) if msg.type == "human"), None)
    topic_shift = last_user and not any(word in latest_message.lower() for word in last_user.content.lower().split())
    if topic_shift and total_meat > 0:
        total_meat = total_meat // 2  # Halve meat on topic shift (tweakable)

    # Boost post-reset
    if meat_points == 0 and not pending_save:
        total_meat += 2

    if intent == "low_say":
        response = f"You’re saying '{latest_message}'—just vibing or got more meat coming?"
    elif intent == "mid_say_teach":
        summary = llm.invoke(f"Summarize in 10 words max: '{latest_message}'").content.strip()
        response = f"You’re saying '{summary}'—teaching me a trick there?"
    elif intent == "high_teach":
        summary = llm.invoke(f"Summarize in 10 words max: '{latest_message}'").content.strip()
        if total_meat >= 8:
            response = f"You’re teaching me '{summary}'—locked that, bro?"
            return {
                "prompt_str": response,
                "meat_points": total_meat,
                "pending_save": {"text": latest_message, "summary": summary, "vibe": "skills"},
                "teach_streak": True
            }
        else:
            response = f"You’re saying '{summary}'—got some juice, huh?"

    return {"prompt_str": response, "meat_points": total_meat, "teach_streak": teach_streak}