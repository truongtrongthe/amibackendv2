
async def training(self, state: Dict = None, user_id: str = None):
        state = state or self.state
        user_id = user_id or state.get("user_id", "thefusionlab")
        logger.debug(f"Starting training - Mode: {self.mode}, State: {state}")

        # Get latest message and ensure it’s a HumanMessage
        latest_msg = state["messages"][-1] if state["messages"] else ""
        if latest_msg and not isinstance(latest_msg, HumanMessage):
            state["messages"][-1] = HumanMessage(content=latest_msg)
        state["messages"] = state["messages"][-200:]
        latest_msg_content = latest_msg.content if isinstance(latest_msg, HumanMessage) else latest_msg

        # Detect intent and update history
        intent_scores = await self.detect_intent(state)
        state["intent_history"].append(intent_scores)
        if len(state["intent_history"]) > 5:
            state["intent_history"].pop(0)
        
        ##LOAD INSTINCT AND it's knowledge here

        # Context excludes latest message
        context = "\n".join(msg.content for msg in state["messages"][-10:-1]) if len(state["messages"]) > 1 else ""
        logger.info(f"Intent scores: {intent_scores}")

        # Teaching mode
        if intent_scores == "teaching" and latest_msg_content.strip():
            await save_pretrain(latest_msg_content, user_id, context)  # Use latest_msg_content
            # Check if character was tagged
            prompt = (
                    f"You're {NAME}, a smart girl speaking natural Vietnamese. "
                    f"Character traits: {state.get('instinct', 'No character traits yet.')}\n"
                    f"Conversation so far: {context}\n"
                    f"Latest: '{latest_msg_content}'\n"
                    f"Task: Show your trainer understand and respond naturally, strongly reflecting your character traits if taught."
                )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()
        
        elif intent_scores =="request":
             #Response to request with knowledge and AI instinct
            relevant_knowledge = query_knowledge(latest_msg_content)
            prompt = (
                    f"You're {NAME}, a smart sales buddy speaking natural Vietnamese. "
                    f"Character traits: {state.get('instinct')}\n"
                    f"Conversation so far: {context}\n"
                    f"Latest: '{latest_msg_content}'\n"
                    f"Your knowledge {relevant_knowledge}"
                    f"Answer human request {latest_msg_content}"
                )
            response = await asyncio.to_thread(LLM.invoke, prompt)
            state["prompt_str"] = response.content.strip()
        # Casual mode (default for non-teaching)
        else:
            prompt = (
                f"You're {AINAME}."
                f"Character traits: {state.get('instinct', 'No character traits yet.')}\n"
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
