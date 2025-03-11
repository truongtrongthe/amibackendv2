from langchain.chat_models import ChatOpenAI
from Archived.graph3 import retrieve_expert_knowledge

llm = ChatOpenAI(model="gpt-4-turbo")

def ami_sales_conversation(user_input):
    """AMI responds to a salesperson using learned knowledge from experts."""
    
    # Step 1: Search FAISS for relevant knowledge
    relevant_knowledge = retrieve_expert_knowledge(user_input)
    
    if relevant_knowledge:
        # Step 2: If relevant knowledge exists, apply it in response
        knowledge_text = "\n".join(relevant_knowledge)
        prompt = f"Based on past expert insights, here's some advice:\n{knowledge_text}\n\nNow, let me tailor this to your situation: {user_input}"
    
    else:
        # Step 3: If no relevant knowledge, ask GPT for advice
        prompt = f"I couldn't find expert knowledge for this exact case. Based on general sales techniques, here's my advice: {user_input}"
    
    response = llm.invoke(prompt)
    return response.content

while True:
    user_input = input("👨‍💼 Salesperson: ")
    if user_input.lower() == "exit":
        break
    response = ami_sales_conversation(user_input)
    print("🤖 AMI:", response)
