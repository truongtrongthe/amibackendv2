import json
from langchain.schema import AIMessage
from graphv2 import g_app  # Assuming this is your compiled LangGraph chatbot
config = {"configurable": {"thread_id": "global_ami"}}

def event_stream(user_input):
    print(f"Sending user input to AI model: {user_input}")
    events = g_app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": "global_thread"}},
        stream_mode="values",
    )
    for event in events:
        print("Received event:", event)
        if "messages" in event and isinstance(event["messages"], list):
            for msg in event["messages"]:
                print("Message type:", type(msg), "Content:", msg.content)
            ai_messages = [msg for msg in event["messages"] if isinstance(msg, AIMessage)]
            print("AI messages found:", ai_messages)
            if ai_messages:
                latest_ai_content = ai_messages[-1].content
                print("Latest AI message:", latest_ai_content)
                yield f"data: {json.dumps({'message': latest_ai_content})}\n\n"
            else:
                print("No AI message in this event")
        else:
            print("No messages key or not a list")