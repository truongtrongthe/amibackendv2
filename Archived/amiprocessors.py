from langchain_openai import OpenAI
import os
from supabase import create_client
from langmem import LangMem

llm = OpenAI(model="gpt-4o")

def extract_customer_info(message):
    """Extracts customer name, interest, and sentiment from chat."""
    prompt = f"Extract customer details from this message: {message}"
    response = llm.invoke(prompt)
    return response


from langchain_openai import OpenAIEmbeddings

# Initialize LangMem with OpenAI embeddings
memory = LangMem(embedding_model=OpenAIEmbeddings(), storage_type="local")

def store_interaction(salesperson_id, customer_name, message, sentiment):
    """Stores a sales conversation in LangMem."""
    memory.store(
        data=message,
        metadata={"salesperson_id": salesperson_id, "customer_name": customer_name, "sentiment": sentiment}
    )

def retrieve_memory(salesperson_id, customer_name):
    """Retrieves past conversations for a given customer."""
    past_conversations = memory.retrieve(
        query=customer_name,
        filters={"salesperson_id": salesperson_id},
        top_k=3  # Fetch most relevant 3 past interactions
    )
    return past_conversations


# Store interaction at the end of conversation
def log_interaction(state):
    """Stores conversation in LangMem."""
    salesperson_id = state["salesperson_id"]
    customer_name = state["customer"]["name"]
    message = state["message"]
    sentiment = state["emotion"]

    store_interaction(salesperson_id, customer_name, message, sentiment)
    return state



def retrieve_past_interactions(state):
    """Fetches customer history from LangMem."""
    salesperson_id = state["salesperson_id"]
    customer_name = state["customer"]["name"]
    
    past_interactions = retrieve_memory(salesperson_id, customer_name)
    state["customer"]["past_interactions"] = past_interactions
    return state

def handle_multiple_matches(matches):
    """Ask salesperson to select the correct customer if multiple are found."""
    return f"Multiple customers named {matches[0]['name']} found. Please select: {matches}"
def check_missing_data(customer):
    """Check if any key customer fields are missing."""
    missing_fields = []
    if not customer.get("phone"):
        missing_fields.append("phone number")
    if not customer.get("email"):
        missing_fields.append("email")
    return missing_fields if missing_fields else None


def suggest_next_steps(emotion):
    """Suggest follow-up based on emotion analysis."""
    if "hesitant" in emotion.lower():
        return "John seems hesitant. Consider sending a case study."
    elif "excited" in emotion.lower():
        return "John is eager! Prepare a proposal ASAP."
    else:
        return "Maintain engagement and provide additional value."
