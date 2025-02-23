
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


vector_db = FAISS.load_local("hito_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
print("FAISS index reloaded successfully!")


query = "How can I pay in Seul?"
relevant_docs = vector_db.similarity_search(query, k=2)
relevant_info = [doc.page_content for doc in relevant_docs]
print("Relevant Info:", relevant_info)