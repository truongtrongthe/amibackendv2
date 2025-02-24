from langchain_openai import ChatOpenAI
from knowledge import  retrieve_relevant_infov2 , retrieve_relevant_info
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
Make sure to answer in the same language as the user. Ensure the answer is concise and to the point.
Ami:"""
)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)


def retrieve_context(user_input):
    """Retrieve relevant context from Pinecone and format it."""
    retrieved_info = retrieve_relevant_infov2(user_input, top_k=10)  # Ensure correct param name `top_k`
    
    if retrieved_info:
        # Extract the 'content' field safely
        context_texts = [f"- {doc.get('content', '')}" for doc in retrieved_info if doc.get("content")]
        return "\n".join(context_texts) if context_texts else "No relevant context found."
    else:
        return "No relevant context found."

chain = (
    RunnablePassthrough.assign(
        history=lambda _: memory.load_memory_variables({})["history"],
        context=lambda x: retrieve_context(x["user_input"])
    )  # Retrieve and assign context
    | prompt
    | llm
)


def ami_telling(query):
    # Prepare input data
    input_data = {
        "user_input": query,
    }
    print("Current Memory:", memory.load_memory_variables({}))
    # Stream response
    for chunk in chain.stream(input_data):
        yield chunk
    memory.save_context({"input": query}, {"output": chunk.content})  

# Example usage
#query = "How to get taller by 5cm in Tokyo?"
#for chunk in ami_telling(query):
#    print(chunk.content, end="", flush=True)