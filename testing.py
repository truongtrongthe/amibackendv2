from graph3 import conversation_flow
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Initialize conversation
state = {"history": []}
langmem_store = []
# Simulate conversation
while True:
    # Run a single step of the graph
    state = conversation_flow.invoke(state)
    
    # Display AMI's response
    response = state["history"][-1].content
    print("🤖 AMI:", response)

    # Get expert input (simulate human response)
    user_input = input("👤 Expert: ")
    
    # Add expert response to history
    state["history"].append(HumanMessage(content=user_input))

    # Exit condition (user types 'exit')
    if user_input.lower() == "exit":
        print("🚀 Conversation ended. Saved knowledge:", langmem_store)
        break

