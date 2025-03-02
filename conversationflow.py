import json
from flask import Response
from langchain.schema import AIMessage
from graph import g_app  # Assuming this is your compiled LangGraph chatbot
import uuid

def event_stream(user_input):
    config = {"configurable": {"thread_id": 1}}
    print(f"Sending user input to AI model: {user_input}")
    events = g_app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )

    latest_ai_content = None
    for event in events:
        print("Received event:", event)
        if "messages" in event and isinstance(event["messages"], list):
            # Get all AIMessages and pick the last one
            ai_messages = [msg for msg in event["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                latest_ai_content = ai_messages[-1].content  # Take the last AIMessage

    if latest_ai_content:
        print("Latest AI message:", latest_ai_content)
        yield f"data: {json.dumps({'message': latest_ai_content})}\n\n"
    else:
        yield f"data: {json.dumps({'message': 'No response generated'})}\n\n"
