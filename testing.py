from pinecone_datastores import index
response = index.fetch(["term_canxi_huu_co_global_thread"], namespace="term_memory")
print(response.vectors["term_canxi_huu_co_global_thread"]["metadata"])