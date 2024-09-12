# ML Papers RAG Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about machine learning papers. It uses a dataset of 100 LLM (Large Language Model) papers to provide informative and contextually relevant responses to user queries about recent advancements in the field of machine learning, particularly focusing on LLMs.

## Features

- Retrieval-Augmented Generation (RAG) for accurate responses about ML papers
- Utilizes a dataset of 100 recent LLM papers
- Configurable model parameters and pipeline components
- Efficient PDF processing, document splitting, and embedding
- FAISS vector store for fast similarity search
- Command-line interface for quick queries
- Streamlit web interface for interactive chatting (optional)

## Dataset

This project uses the "100 LLM Papers to Explore" dataset from Kaggle:
[https://www.kaggle.com/datasets/ruchi798/100-llm-papers-to-explore](https://www.kaggle.com/datasets/ruchi798/100-llm-papers-to-explore)

The dataset includes 100 PDF papers covering various aspects of Large Language Models, providing a rich knowledge base for the chatbot.



### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/MahtabRanjbar/ml-papers-rag.git
    cd ml-papers-rag
    ```

2. Install the required packages:
    ```bash 
    pip install -r requirements.txt
    ```

3. Configure the project:
- Open `config/config.yml` and adjust the settings as needed, especially the path to the downloaded papers

### Usage

#### Command-line Interface
Run a single query:

```bash
python src/main.py "What are the recent advancements in LLM fine-tuning techniques?"
```
## Models

This project uses the following models:

- LLM: Meta-Llama-3-8B-Instruct
- Embeddings: BAAI/bge-base-en-v1.5

You can change these in the `config.yml` file.

## Dataset

The chatbot's knowledge base is built from a collection of machine learning papers. You can replace or expand this dataset by adding PDF files to the configured input directory.

## Configuration

All configurable parameters are stored in `config/config.yml`. This includes:

- Model settings (name, temperature, top_p, etc.)
- Text splitting parameters
- Embedding model selection
- File paths for input and output
   
