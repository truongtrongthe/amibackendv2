from graph import graph

config = {"configurable": {"thread_id": "1"}}


def loop():
    def stream_graph_updates(user_input: str):
        config = {"configurable": {"thread_id": "1"}}
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
            )
        for event in events:
            message = event["messages"][-1]
            print(message["content"])  # Use safe content extraction

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break

loop()