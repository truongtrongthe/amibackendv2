from typing import List, Dict
from openai import OpenAI
from pinecone_datastores import pinecone_index
from typing import List, Dict 
client = OpenAI()
# Step 1: Extract Insights from Expert Conversations
import json
from typing import List, Dict
def extract_expert_insights(conversation: List[str]) -> Dict[str, List[str]]:
    """
    Processes expert conversations to extract valuable skills, experiences, and knowledge.
    Ensures that each extracted category is always a list.
    """
    prompt = f"""Extract the following from this expert conversation while preserving the original language:
    - **Skills** (Sales techniques, communication skills, persuasion methods, relationship-building, emotional intelligence, customer engagement strategies)
    - **Experiences** (Lessons learned from real-world scenarios, past successes/failures, customer behavior patterns)
    - **Knowledge** (Product information, industry insights, factual details, sales principles)

    **Conversation:**
    {conversation}

    Format the response as a JSON object with keys: 'skills', 'experiences', and 'knowledge'.
    Ensure each value is a list, even if it contains a single item.
    **Do not translate the extracted insights; keep them in the same language as the input.**
    """


    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AI extracting valuable insights from expert discussions."},
                  {"role": "user", "content": prompt}]
    )
    
    response_content = response.choices[0].message.content if response.choices else ""
    
    if not response_content:
        raise ValueError("Received empty response from OpenAI API")
    
    # Clean up JSON formatting if wrapped in triple backticks
    response_content = response_content.strip()
    if response_content.startswith("```json"):
        response_content = response_content[7:]
    if response_content.endswith("```"):
        response_content = response_content[:-3]
    
    try:
        parsed_response = json.loads(response_content)
        # Ensure all keys contain lists
        return {key: parsed_response.get(key, []) if isinstance(parsed_response.get(key), list) else [parsed_response.get(key)]
                for key in ["skills", "experiences", "knowledge"]}
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {response_content}") from e


# Step 2: Store Insights in Pinecone
def get_embedding(text: str) -> List[float]:
    """
    Generates an embedding for a given text using OpenAI's embedding model.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def store_insights_in_pinecone(insights: Dict[str, List[str]]):
    """
    Stores extracted skills, experiences, and knowledge into Pinecone.
    """
    if not isinstance(insights, dict):
        raise ValueError("Insights must be a dictionary")
    
    for category, items in insights.items():
        if not isinstance(items, list):
            raise ValueError(f"Expected list for category {category}, but got {type(items)}")
        
        for item in items:
            vector = get_embedding(item)  # Generate embedding
            pinecone_index.upsert([(str(hash(item)), vector, {"category": category, "text": item})])



# Step 3: Retrieve Relevant Insights
def retrieve_insights(query: str) -> Dict[str, List[str]]:
    """
    Retrieves relevant skills, experiences, or knowledge from Pinecone based on a query.
    """
    results = pinecone_index.query(vector=get_embedding(query), top_k=5, include_metadata=True)

    insights = {}
    for match in results.matches:  # `.matches` gives a list of `ScoredVector`
        if match.metadata:
            insights[match.id] = match.metadata.get("category", "Unknown")

    return insights

# Example Usage
    #"Expert: One trick I use when customers hesitate is giving them a limited-time offer.",
    #"Expert: If a customer refuses to pay after a signed contract, I immediately escalate to legal action.",
    #"Expert: Our product contains natural ingredients like Vitamin C and Zinc for immune support."

def ami_goal_reasoning(query: str, goal: str) -> str:
    """
    Enhances AMI's reasoning by incorporating expert skills and experiences.
    """
    expert_insights = retrieve_insights(query)
    
    prompt = f"""
    Given the goal: "{goal}", and the retrieved expert insights:
    {expert_insights}
    
    Determine the best next move to achieve the goal, leveraging the expert skills and experiences.
    Provide a strategic and empathetic response.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AI optimizing sales strategies with expert knowledge."},
                  {"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content if response.choices else ""

conversation = [
"Khi giao tiếp cần dùng giọng chân thành!"
]

ami_instinct_style = {
    "tag": "ami_instinct",
    "personality": "You are AMI, a sales apprentice. You are young, energetic, and confident. Show the user you are sharp, quick-witted, and ambitious.",
    "tone": "Smart, witty, and engaging.",
    "example_responses": [
        "Haha, that’s an interesting take! So tell me, how would you handle a tough sales call?",
        "Oh, I like where this is going! Convince me why your approach works best.",
        "Smart move! What’s the secret behind that strategy?"
    ]
}


#insights = extract_expert_insights(conversation)
#store_insights_in_pinecone(insights)
#retrieved = retrieve_insights("handling hesitant customers")
#next_move = ami_goal_reasoning("handling hesitant customers", "close the sale")

#print("Next Move Recommendation:", next_move)

#print("Extracted Insights:", insights)
#print("Retrieved Insights:", retrieved)
