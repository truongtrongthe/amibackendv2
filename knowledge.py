from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# Initialize embeddings
embeddings = OpenAIEmbeddings()

def tobrain(summary):
    
    #text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    #chunks = text_splitter.split_text(summary)
    vector_db = FAISS.from_texts([summary], embeddings)
    #vector_db.add_texts(chunks)
    # Save the FAISS index locally (optional)
    print("Saved the database.")
    vector_db.save_local("hito_index")

def retrieve_relevant_info(query, k=1):
    """Retrieve the top-k relevant information based on the query."""
    vector_db = FAISS.load_local("hito_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    print("Loaded successfully!")
    relevant_docs = vector_db.similarity_search(query, k=k)
    relevant_info = [doc.page_content for doc in relevant_docs]
    print("Retrieved successfully:", relevant_info)
    return relevant_info
