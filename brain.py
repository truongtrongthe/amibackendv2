from langchain_openai import ChatOpenAI
from knowledge import retrieve_relevant_info
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
# Initialize LLM for conversation
llm = ChatOpenAI(model="gpt-4o", streaming=True)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["history", "user_input", "context"],
    template="""History:
{history}

Context:
{context}

User: {user_input}
Ami:"""
)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

def retrieve_context(user_input):
    retrieved_info = retrieve_relevant_info(user_input, k=3)
    return "\n".join(retrieved_info) if retrieved_info else "No relevant context found."

chain = (
    RunnablePassthrough.assign(
        history=lambda _: memory.load_memory_variables({})["history"],
        context=lambda x: retrieve_context(x["user_input"])
    )  # Retrieve and assign context
    | prompt
    | llm
)

def ami_telling(query):
    """Stream a response with retrieved context."""
    # Prepare input data
    input_data = {
        "user_input": query,
    }
    
    # Stream response
    for chunk in chain.stream(input_data):
        yield chunk
    memory.save_context({"input": query}, {"output": chunk.content})  
# Example usage
query = "I don't have money to pay for the course. What should I do?"
for chunk in ami_telling(query):
    print(chunk.content, end="", flush=True)