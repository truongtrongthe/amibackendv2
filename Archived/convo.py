import supabase
import os
from openai import OpenAI
import json
from datetime import datetime
import uuid
from numpy import dot
from numpy.linalg import norm
from pinecone_datastores import pinecone_index
from typing import List, Dict 
client = OpenAI()

def generate_embedding(skill_text):
    try:
        response = client.embeddings.create(
        input=skill_text,
        model="text-embedding-3-small",
        dimensions=1536  # Match the dimensions of ada-002
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return None  # Avoid breaking downstream logic
convo_history = []


# Step 1: Fetch and Summarize Past Conversations
def analyze_past_conversations(conversations: List[str]) -> Dict[str, str]:
    """
    Summarizes past conversations to detect the current customer state.
    """
    prompt = f"""Summarize the following conversation history, identify the contact's current intent and potential objections:
    {conversations}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a sales assistant analyzing customer interactions."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Step 2: Determine Customer's Current State
def determine_customer_state(summary: str) -> str:
    """
    Analyzes the summary to determine where the customer is in the sales funnel.
    """
    prompt = f"""Based on the following summary, determine the customer's current state in the sales process:
    {summary}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a sales strategist categorizing customer intent."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


#"Ice-breaking: Understand that customer comes to you mostly likely because they are interested in the company products. Bring up product info, gently!",
#pinecone_index.query(customer_state") query for relevant skills
EXPERT_SALES_SKILLS = [
    "Active Listening: Understand customer needs by paying close attention to what they say.",
    "Ice-breaking: Understand that customer comes to you mostly likely because they are interested in the company products so, try to trigger their need, hard!",
    "Objection Handling: Address concerns effectively with clear, confidence-building responses.",
    "Storytelling: Use real-world success stories to illustrate value.",
    "Urgency Creation: Provide limited-time offers or incentives to drive decisions.",
    "Value Selling: Focus on benefits rather than just features or pricing.",
    "Social Proof: Mention case studies or testimonials to build credibility."
]

PRODUCT_INFO = {
    "key_features": "AI-powered sales assistant that suggests the best outreach strategies.",
    "pricing": "Flexible plans starting at $49/month with custom enterprise options.",
    "customer_success_story": "Client X increased conversions by 35% using our personalized AI outreach.",
    "competitive_advantage": "Unlike generic automation tools, we provide human-like, empathetic messaging."
}

# Step 3: Predict Best Next Move
def predict_next_move(customer_state: str) -> str:
    
    """
    Predicts the best next move using expert sales skills and injects relevant product info when needed.
    Uses Chain of Thought (CoT) reasoning.
    """
    expert_skills_prompt = "\n".join(EXPERT_SALES_SKILLS)

    # Identify if product knowledge is needed
    product_info_prompt = f"""
    When considering the next move, if the strategy involves discussing product details, here is relevant product information:
    - Key Features: {PRODUCT_INFO['key_features']}
    - Pricing: {PRODUCT_INFO['pricing']}
    - Customer Success Story: {PRODUCT_INFO['customer_success_story']}
    - Competitive Advantage: {PRODUCT_INFO['competitive_advantage']}
    If product details are not relevant, ignore this information.
    """

    prompt = f"""Based on the following customer state, predict the best next move a sales representative should take.
    Use expert-trained sales skills and inject relevant product information if necessary.

    Expert Sales Skills:
    {expert_skills_prompt}

    Customer State:
    {customer_state}

    {product_info_prompt}

    Step-by-step reasoning:
    1. Analyze the customer's intent and objections.
    2. Select relevant expert skills to address the situation.
    3. If an expert skill suggests discussing product details, integrate the relevant information.
    4. Determine the best next move based on expert sales techniques.
    5. Justify why this move is optimal.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a sales assistant suggesting the best next move using Chain of Thought reasoning."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Step 4: Generate Personalized Outreach Message
def generate_outreach_message(next_move: str) -> str:
    """
    Generates a proactive, engaging outreach message for instant messaging.
    """
    prompt = f"""Based on the following suggested next move, craft a short, engaging outreach message suitable for instant messaging (e.g., Facebook Messenger):
    {next_move}
    Make it casual, friendly, and action-driven, encouraging an immediate response. Avoid formalities and keep it dynamic, like a real-time chat."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a persuasive sales representative crafting engaging messages in a friendly, conversational style."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Example Usage
conversation_history = [
    "John: Hi!",
    "Sale Reps: Hi John! I'm here to help. What are you looking for?",
    "John: I'm looking for x. How much?"
]

summary = analyze_past_conversations(conversation_history)
customer_state = determine_customer_state(summary)
next_move = predict_next_move(customer_state)
outreach_message = generate_outreach_message(next_move)

#print("Summary:", summary)
print("Customer State:", customer_state)
print("Next Move:", next_move)
print("Outreach Message:", outreach_message)
