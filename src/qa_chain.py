from langchain import PromptTemplate
from langchain.chains import RetrievalQA

def create_qa_chain(llm, retriever):
    '''
    Create a QA chain with the given LLM model and retriever.
    '''
    
    prompt_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant who receives excerpted parts of a context along with a question.
    Use only the following parts of the context to answer the question at the end.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Context: {context}
    Answer the following question:
    {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False
    )