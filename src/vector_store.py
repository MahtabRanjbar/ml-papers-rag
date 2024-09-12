import os
from langchain_community.vectorstores import FAISS


def create_or_load_vectordb(texts, embeddings, output_path):
    if not os.path.exists(f"{output_path}/index.faiss"):
        vectordb = FAISS.from_documents(documents=texts, embedding=embeddings)
        vectordb.save_local(f"{output_path}/faiss_index_ml_papers")
    else:
        vectordb = FAISS.load_local(
            output_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return vectordb


def create_retriever(vectordb, k):
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )