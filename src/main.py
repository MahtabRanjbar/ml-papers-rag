import yaml
from utils import login_to_huggingface, llm_ans, wrap_text_preserve_newlines, process_llm_response
from data_processing import load_documents, split_documents, get_embeddings
from vector_store import create_or_load_vectordb, create_retriever
from model import build_model
from qa_chain import create_qa_chain
import argparse


def main(prompt):
    # Load configuration
    with open("config/config.yml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load and process documents
    documents = load_documents(config['paths']['pdfs'], config['debug'])
    texts = split_documents(documents, config['text_splitter']['chunk_size'], config['text_splitter']['chunk_overlap'])
    
    # Create or load vector store
    embeddings = get_embeddings(config['embeddings']['model_repo'])
    vectordb = create_or_load_vectordb(texts, embeddings, config['paths']['output'])
    
    # Build model and create QA chain
    llm = build_model(
        config['model']['name'],
        config['model']['temperature'],
        config['model']['top_p'],
        config['model']['repetition_penalty'],
        config['model']['max_new_tokens']
    )
    retriever = create_retriever(vectordb, config['retriever']['k'])
    qa_chain = create_qa_chain(llm, retriever)
    
    # Process the prompt and print the result
    result = llm_ans(qa_chain, prompt)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG chatbot with a given prompt.")
    parser.add_argument("prompt", type=str, help="The prompt to process")
    args = parser.parse_args()
    main(args.prompt)