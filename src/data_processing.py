from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(pdf_path, debug=False):
    loader = DirectoryLoader(
        pdf_path,
        glob="./*3215v3.pdf" if debug else "./*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    return loader.load()


def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceInstructEmbeddings


def get_embeddings(model_name):
    return HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"}
    )