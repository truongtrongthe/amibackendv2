from langchain_openai import ChatOpenAI
from knowledge import  retrieve_relevant_infov2 , retrieve_relevant_info
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
# Initialize LLM for conversation
llm = ChatOpenAI(model="gpt-4o", streaming=True)

# Define the prompt template
prompt_old = PromptTemplate(
    input_variables=["history", "user_input", "context"],
    template="""History:
{history}

Context:
{context}

User: {user_input}
Make sure to answer in the same language as the user. Ensure the answer is concise and to the point.
Ami:"""
)

prompt_ok = PromptTemplate(
    input_variables=["history", "user_input", "context"],
    template="""Dưới đây là một số thông tin liên quan đến câu hỏi của bạn:

    {context}

    Dựa trên thông tin trên, hãy trả lời một cách chi tiết, rõ ràng và súc tích.

    Người dùng: {user_input}
    Ami:"""
)
prompt = PromptTemplate(
    input_variables=["history", "user_input", "context"],
    template="""
Dựa vào các thông tin trước đây của người dùng, hãy đảm bảo câu trả lời phù hợp với phong cách của họ.
Lịch sử cuộc trò chuyện:
{history}

Ngữ cảnh liên quan:
{context}

Người dùng: {user_input}
AMI (giữ nguyên phong cách của người dùng):"""
)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)


def retrieve_context_old(user_input):
    """Retrieve relevant context from Pinecone and format it."""
    retrieved_info = retrieve_relevant_infov2(user_input, top_k=10)  # Ensure correct param name `top_k`
    
    if retrieved_info:
        # Extract the 'content' field safely
        context_texts = [f"- {doc.get('content', '')}" for doc in retrieved_info if doc.get("content")]
        print("context_texts:", context_texts)
        return "\n".join(context_texts) if context_texts else "No relevant context found."
    else:
        return "No relevant context found."

def retrieve_context(user_input):
    """Retrieve relevant context from Pinecone and return a structured summary."""
    retrieved_info = retrieve_relevant_infov2(user_input, top_k=10)

    if not retrieved_info:
        return "Không tìm thấy ngữ cảnh phù hợp."

    structured_summary = []
    for doc in retrieved_info:
        content = doc.get("content", "").strip()
        if content:
            structured_summary.append(content)

    return "\n\n".join(structured_summary) if structured_summary else "Không tìm thấy ngữ cảnh phù hợp."

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
    last_response = ""
    for chunk in chain.stream(input_data):
        yield chunk
        last_response += chunk.content  # Accumulate responses
    memory.save_context({"input": query}, {"output": last_response.strip()})
    