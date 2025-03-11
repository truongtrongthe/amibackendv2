import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from pinecone_datastores import pinecone_index
import textwrap
# Initialize OpenAI, Embeddings, and Pinecone
llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

PINECONE_INDEX = pinecone_index

# Define the State without meat_points
class State(TypedDict):
    messages: Annotated[list, add_messages]
    prompt_str: str
    last_topic: str
    mode: str  # New: "casual" or "learner"
def detect_intent(state: State):
    """
    Detect the intent of the human by analyzing the conversation.
    If the topic changes and the previous topic isn't saved to Pinecone, store it in MemorySaver.
    """
    latest_message = state["messages"][-1].content
    prior_messages = state["messages"][:-1]

    # Detect intent based on the latest message
    intent_prompt = f"""
    Analyze the following conversation and determine the intent of the latest message:
    Prior Messages: {[m.content for m in prior_messages]}
    Latest Message: {latest_message}
    Possible Intents:
    - Teaching a new skill
    - Sharing a lesson
    - Switching topics
    - Asking a question
    - Other

    Return the intent as a single word or phrase.
    """
    intent = llm.invoke(intent_prompt).content.strip()

    # Check if the topic has changed
    if state["last_topic"] and intent == "Switching topics":
        # If the previous topic isn't saved to Pinecone, store it in MemorySaver
        if not is_topic_saved(state["last_topic"]):  # Placeholder function
            summary = generate_summary(prior_messages)
            save_to_memory(state["last_topic"], summary)  # Placeholder function

    # Update the last topic
    state["last_topic"] = intent if intent != "Switching topics" else ""

    return intent
# Generate a summary for storage

def generate_summary(messages):
    """
    Generate a summary of the conversation for storage.
    """
    conversation = "\n".join([m.content for m in messages])
    summary_prompt = f"""
    Summarize the key points of this conversation in one or two sentences:
    {conversation}
    """
    summary = llm.invoke(summary_prompt).content.strip()
    return summary
def ami_node(state: State):
    """
    Drive Ami's behavior based on the detected intent.
    """
    intent = detect_intent(state)
    latest_message = state["messages"][-1].content

    if intent == "Teaching a new skill" or intent == "Sharing a lesson":
        # Extract knowledge from the conversation
        knowledge_prompt = f"""
        Extract actionable insights, skills, or lessons from the following message:
        {latest_message}
        Return the extracted knowledge as a JSON object with the following structure:
        {{
            "topic": "The topic of the knowledge",
            "insights": ["List of actionable insights or skills"],
            "lessons": ["List of lessons or best practices"]
        }}
        """
        try:
            # Attempt to parse the LLM response as JSON
            extracted_knowledge = json.loads(llm.invoke(knowledge_prompt).content.strip())
        except json.JSONDecodeError:
            # Handle cases where the LLM returns invalid JSON
            print("Error: LLM returned invalid JSON. Falling back to default response.")
            extracted_knowledge = {
                "topic": "Unknown",
                "insights": [],
                "lessons": []
            }

        # Store the knowledge in Pinecone or MemorySaver
        if is_topic_saved(extracted_knowledge["topic"]):  # Placeholder function
            save_to_pinecone(extracted_knowledge)  # Placeholder function
        else:
            save_to_memory(extracted_knowledge["topic"], extracted_knowledge)  # Placeholder function

        # Ask a follow-up question to deepen understanding
        follow_up_prompt = f"""
        Based on the following knowledge, generate a follow-up question to deepen understanding:
        {extracted_knowledge}

        Return the question as a string.
        """
        follow_up_question = llm.invoke(follow_up_prompt).content.strip()

        response = follow_up_question

    elif intent == "Switching topics":
        response = "I noticed we're switching topics. Let me summarize what we've discussed so far."

    else:
        response = "I'm here to learn from you. Please share your expertise!"

    return {"prompt_str": response}

def is_topic_saved(topic: str) -> bool:
    """
    Check if a topic is already saved in Pinecone.
    """
    # Placeholder implementation
    return False

def save_to_pinecone(knowledge: dict):
    """
    Save knowledge to Pinecone.
    """
    # Placeholder implementation
    print(f"Saved to Pinecone: {knowledge}")

def save_to_memory(topic: str, summary: str):
    """
    Save a topic and its summary to MemorySaver.
    """
    # Placeholder implementation
    print(f"Saved to MemorySaver: {topic} - {summary}")
# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("ami", ami_node)
graph_builder.add_edge(START, "ami")
graph_builder.add_edge("ami", END)
checkpointer = MemorySaver()
convo_graph = graph_builder.compile(checkpointer=checkpointer)

# Streaming function
def convo_stream(user_input, user_id, thread_id="learning_thread"):
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})
    history = checkpoint["channel_values"].get("messages", []) if checkpoint else []
    new_message = HumanMessage(content=user_input)
    updated_messages = history + [new_message]
    
    try:
        # Initialize the state with default values
        state = {
            "messages": updated_messages,
            "prompt_str": "",
            "last_topic": "",  # Initialize last_topic
        }
        
        # Invoke the graph with the initialized state
        state = convo_graph.invoke(
            state,
            {"configurable": {"thread_id": thread_id}}
        )
        response = state["prompt_str"]
        
        for chunk in textwrap.wrap(response, width=100):
            yield f"data: {json.dumps({'message': chunk})}\n\n"
        
        #ai_message = AIMessage(content=response)
        #state["messages"] = updated_messages + [ai_message]
        convo_graph.update_state({"configurable": {"thread_id": thread_id}}, state)
        print(f"AI response saved: {response}...")
        
    except Exception as e:
        error_msg = f"Error in stream: {str(e)}"
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
# Test it out
if __name__ == "__main__":
    thread_id = "test_thread"
    expert_inputs = [
        "Hey Ami",
    ]
    for user_input in expert_inputs:
        print(f"\nExpert: {user_input}")
        for chunk in convo_stream(user_input, thread_id):
            print(chunk)