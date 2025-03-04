import json

from langchain_core.messages import AIMessage,HumanMessage
from faissaccess import initialize_vector_store
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from graphv2 import g_app  # Assuming this is your compiled LangGraph chatbot
from graph3 import convo_graph
llm = ChatOpenAI(model="gpt-4o", streaming=True)
vector_store = initialize_vector_store("tfl")
def event_stream(user_input, user_id, thread_id="global_thread"):
    
    print(f"Sending user input to AI model: {user_input}")
    # Update graph state with user input
    state = convo_graph.invoke(
        {"messages": [HumanMessage(content=user_input)], "user_id": user_id},
        {"configurable": {"thread_id": thread_id}}
    )
    print("Graph state updated:", state)

    # Get context from state for streaming
    convo_history = "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])
    relevant_docs = vector_store.similarity_search(user_input, k=5, filter={"user_id": user_id})
    seen = set()
    memory_context_lines = []
    for doc in relevant_docs:
        content = doc.page_content.strip()
        if content and content not in seen and len(content) > 3:
            memory_context_lines.append(f"[{doc.metadata.get('timestamp','unknown')}]{content}")
            seen.add(content)
    memory_context = "\n".join(memory_context_lines) or "No significant memories yet."
    prompt = f"""
    You are Ami, an assistant with total recall...
    Long-term memories: {memory_context}
    Recent conversation: {convo_history}
    User: {user_input}
    """

    # Stream LLM response directly
    try:
        for chunk in llm.stream(prompt):
            if chunk.content.strip():
                print(f"Yielding chunk: {chunk.content}")
                yield f"data: {json.dumps({'message': chunk.content})}\n\n"
    except Exception as e:
        print(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"