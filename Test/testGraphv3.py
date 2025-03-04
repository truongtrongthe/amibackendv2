
# Test function for multi-turn conversation

def test_conversation(inputs: list[str]):
    state = {"messages": [], "user_id": USER_ID}
    print(f"Initial state: {state}")
    for user_input in inputs:
        print(f"\nUser: {user_input}")
        state = convo_graph.invoke(
            {"messages": [HumanMessage(content=user_input)], "user_id": USER_ID},
            config={"configurable": {"thread_id": "brian_thread"}}
        )
        print(f"Agent: {state['messages'][-1].content}")

if __name__ == "__main__":
    test_inputs = [
        "Hi, I like coffee.",
        "What do I like?",
        "I also prefer tea.",
        "What are my preferences now?",
        "Just saying hi again."
    ]
    test_conversation(test_inputs)