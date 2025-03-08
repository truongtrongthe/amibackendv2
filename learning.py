import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone_datastores import pinecone_index
import re
from experts import extract_expert_insights,retrieve_insights,store_insights_in_pinecone
# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

# Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    user_id: str

def detect_intent(message):
    """Use LLM to classify whether the user input is casual or learning-related."""
    classification_prompt = f"""
    Classify the following message as either "learning" or "casual":
    
    Message: "{message}"
    
    Respond with only one word: "learning" or "casual".
    """
    response = llm.invoke(classification_prompt)
    intent = response.content.strip().lower()

    if intent not in ["learning", "casual"]:
        intent = "learning"  # Default to learning if uncertain

    return intent

def learning_node(state: State):
    latest_message = state["messages"][-1]
    user_id = state["user_id"]
    current_convo_history = "\n".join(
        f"{m.type}: {m.content}" for m in state["messages"]
    )
    intent = detect_intent(latest_message.content)

    if intent == "casual":
        if intent == "casual":
            casual_prompt = f"""
            You're Ami, a Sales copilot. You're young, passionate, and confident.
            You always bring an Energetic and Positive vibe into conversation.
            
            The user just said: "{latest_message.content}"
            
            Respond in a natural, friendly way, keeping it engaging and human-like.
            """
            casual_response = casual_prompt.strip()
            return {"prompt_str": casual_response, "user_id": user_id}
    # Step 1: Detect surface intent
    insights = extract_expert_insights(latest_message)
     # Ensure insights is always a dictionary
    if not isinstance(insights, dict):
        print("Warning: extract_expert_insights() did not return a dictionary.")
        insights = {"knowledge": [], "skills": [], "experiences": []}

    # Step 2: Store extracted insights in Pinecone
    store_insights_in_pinecone(insights)
    
    # Step 3: Get insights for each category
    knowledge = insights.get("knowledge", [])
    skill = insights.get("skills", [])
    exp = insights.get("experiences", [])

    if knowledge or skill or exp:
    # Step 4: Construct confirmation message
        prompt = f"""You're AMI, an active learner. You tell your trainer that you have absorbed their message as follows:
        
        **Knowledge Acquired:** 
        {', '.join(knowledge) if knowledge else "None"}
        
        **Skills Learned:** 
        {', '.join(skill) if skill else "None"}
        
        **Experience Understood:** 
        {', '.join(exp) if exp else "None"}

        Please let me know if my understanding is correct!
        """
    else:
        prompt = f"""
        You're AMI, an active learner. The user just shared: "{latest_message.content}"
        You value their input and want to learn.

        Acknowledge what they said and ask a **natural** follow-up to better understand the key takeaways or skills.
        """
    return {"prompt_str": prompt, "user_id": user_id}


# Build and compile the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", learning_node)
graph_builder.add_edge(START, "chatbot")
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

def learning_stream(user_input, user_id, thread_id="learning_thread"):
    print(f"Sending user input to AI model: {user_input}")
    # Retrieve the existing conversation history from MemorySaver
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    history = checkpoint["channel_values"].get("messages", []) if checkpoint else []
    
    # Append the new user input to the history
    new_message = HumanMessage(content=user_input)
    updated_messages = history + [new_message]
    
    try:
        # Invoke the graph with the full message history
        state = convo_graph.invoke(
            {"messages": updated_messages, "user_id": user_id},
            {"configurable": {"thread_id": thread_id}}
        )
        prompt = state["prompt_str"]
        print(f"Prompt to AI model: {prompt}")
        
        # Stream the LLM response
        full_response = ""
        for chunk in llm.stream(prompt):
            if chunk.content.strip():
                full_response += chunk.content
                yield f"data: {json.dumps({'message': chunk.content})}\n\n"
        
        # Save the AI response back to the graph
        ai_message = AIMessage(content=full_response)
        convo_graph.invoke(
            {"messages": updated_messages+[ai_message], "user_id": user_id},
            {"configurable": {"thread_id": thread_id}}
        )
        print(f"AI response saved to graph: {full_response[:50]}...")
        
    except Exception as e:
        error_msg = f"Error in event stream: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

# Optional: Utility function to inspect history (for debugging)
def get_conversation_history(thread_id="global_thread"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    if checkpoint:
        state = checkpoint["channel_values"]
        return state.get("messages", [])
    return []
