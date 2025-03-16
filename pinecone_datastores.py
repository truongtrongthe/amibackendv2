import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "ami"

llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)


# Kiểm tra và tạo index nếu chưa có
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


