from graph import app  # Assuming this is your compiled LangGraph chatbot
import json
from flask import Response
config = {"configurable": {"thread_id": "1"}}


def event_stream(user_input):
    """
    Streams chatbot responses using Server-Sent Events (SSE).
    """
    events = app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    
    for event in events:
        print("Received event:", event)
        
        if "messages" in event and event["messages"]:
            last_message = event["messages"][-1]
            # Try to access content as an attribute first
            if hasattr(last_message, "content"):
                message = last_message.content
                yield f"data: {json.dumps({'message': message})}\n\n"
            else:
                # Fallback in case it's a dict
                try:
                    message = last_message["content"]
                    yield f"data: {json.dumps({'message': message})}\n\n"
                except (TypeError, KeyError):
                    print("Warning: 'content' key missing in last_message:", last_message)
        else:
            print("Warning: 'messages' key missing or empty in event:", event)
        
        #time.sleep(0.1)  # Simulate real-time delay
