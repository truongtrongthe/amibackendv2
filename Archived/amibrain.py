from langchain_openai import ChatOpenAI
from Archived.knowledge import  retrieve_relevant_infov2 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
import supabase
import os
from langchain_openai import OpenAIEmbeddings


supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)



# Initialize LLM for conversation
llm = ChatOpenAI(model="gpt-4o", streaming=True)

prompt = PromptTemplate(
    input_variables=["history", "user_input", "products", "user_style", "sales_skills"],
    template="""
Dựa vào các thông tin trước đây của người dùng, hãy đảm bảo câu trả lời phù hợp với phong cách của họ.
Lịch sử cuộc trò chuyện:
{history}

Sản phẩm liên quan:
{products}

Kỹ năng bán hàng cần áp dụng:
{sales_skills}

Phong cách phản hồi của người dùng trước đây:
{user_style}

Người dùng: {user_input}
AMI (giữ nguyên phong cách của người dùng):"""
)

#  chat_history = ChatMessageHistory()
chat_history = ChatMessageHistory()

memory = ConversationBufferMemory(
    chat_memory=chat_history,
    memory_key="history",  # REQUIRED in newer versions
    return_messages=True
)


def retrieve_product(user_input):
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
    

def search_sales_skill(query_text):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
    query_embedding = embedding_model.embed_query(query_text)

    response = supabase_client.rpc("match_skills", {
        "query_embedding": query_embedding,
        "match_threshold": 0.75,
        "match_count": 1
    }).execute()

    print("Raw response from Supabase:", response)  # Debugging

    # Directly access `.data`
    data = response.data  # Correct way to access data

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        skill_text = data[0].get("skill_text")  # Extract skill_text
        if skill_text:
            return skill_text  # Return the found skill

    return "Không tìm thấy kỹ năng phù hợp."


chain = (
    RunnablePassthrough.assign(
        history=lambda _: memory.load_memory_variables({}).get("history", []),
        products=lambda x: retrieve_product(x["user_input"]),
        sales_skills=lambda x: search_sales_skill(x["user_input"]),
        user_style=lambda _: "lịch sự"
        #user_style=lambda x: search_user_style(x["user_input"])
    )  # Retrieve and assign context
    | prompt
    | llm
)

def ami_selling(query):
    input_data = {"user_input": query}
    print("Current Memory:", memory.load_memory_variables({}))
    
    last_response = ""

    # 🔥 Dùng yield from để đảm bảo stream chảy đúng
    response_stream = chain.stream(input_data)
    yield from (chunk.content for chunk in response_stream)  # ✅ Cách khác

    # ✅ Lưu vào memory sau khi hoàn thành
    memory.save_context({"input": query}, {"output": last_response.strip()})

def ami_selling_old(query):
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
    

