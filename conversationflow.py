import json
from langchain_core.messages import AIMessage,HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from graphv2 import g_app  # Assuming this is your compiled LangGraph chatbot
from graph3 import convo_graph
llm = ChatOpenAI(model="gpt-4o", streaming=True)

def event_stream(user_input, user_id, thread_id="global_thread"):
    
    print(f"Sending user input to AI model: {user_input}")
    # Update graph state with user input
    state = convo_graph.invoke(
        {"messages": [HumanMessage(content=user_input)], "user_id": user_id},
        {"configurable": {"thread_id": thread_id}}
    )
    #print("Graph state updated:", state)
    prompt = state["prompt_str"]
    print(f"Prompt to AI model:", prompt)
    
    # Stream LLM response directly
    try:
        for chunk in llm.stream(prompt):
            if chunk.content.strip():
                #print(f"Yielding chunk: {chunk.content}")
                yield f"data: {json.dumps({'message': chunk.content})}\n\n"
    except Exception as e:
        print(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

