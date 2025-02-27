from langchain_openai import ChatOpenAI
from knowledge import  retrieve_relevant_infov2 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

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

    📌 Kỹ năng bán hàng cần áp dụng:
    {sales_skills}

    📢 **Hướng dẫn cho AMI:**
    1. Áp dụng các kỹ năng vào phản hồi để giúp cuộc trò chuyện tự nhiên hơn.
    2. Nếu có kỹ năng "Lắng nghe chủ động" → Hãy paraphrase lại ý của khách hàng trước khi tư vấn.
    3. Nếu có kỹ năng "Kể chuyện" → Hãy thêm một câu chuyện ngắn để minh họa sản phẩm.
    4. Nếu có kỹ năng "Giải quyết phản đối" → Hãy xử lý lo ngại của khách hàng trước khi tư vấn.

    Người dùng: {user_input}
    🎯 **AMI (giữ nguyên phong cách của người dùng + áp dụng kỹ năng bán hàng):**
    """
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


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = "us-east-1"  # Check Pinecone console for your region


index_name = "ami-knowledge"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
existing_indexes = [i['name'] for i in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches your model's output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


def search_sales_skills(query_text, max_skills=3):
    """Truy vấn kỹ năng bán hàng từ Pinecone dựa trên độ tương đồng với input của người dùng."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
    query_embedding = embedding_model.embed_query(query_text)

    # Tìm kiếm trong Pinecone (dùng cosine similarity)
    index = pc.Index(index_name)
    response = index.query(
        vector=query_embedding,
        top_k=max_skills,  # Số lượng kỹ năng tối đa
        include_metadata=True  # Lấy luôn thông tin kỹ năng từ metadata
    )

    skills = []
    if response and "matches" in response:
        for match in response["matches"]:
            skill_text = match.get("metadata", {}).get("skill_text")
            if skill_text:
                skills.append(skill_text)

    return skills if skills else ["Không tìm thấy kỹ năng phù hợp."]


chain = (
    RunnablePassthrough.assign(
        history=lambda _: memory.load_memory_variables({}).get("history", []),
        products=lambda x: retrieve_product(x["user_input"]),
        sales_skills=lambda x: ", ".join(search_sales_skills(x["user_input"], max_skills=3)),  # Lấy kỹ năng từ Pinecone
        user_style=lambda _: "lịch sự"
    )  
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
