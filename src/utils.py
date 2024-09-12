import textwrap
import time
import os
from huggingface_hub import login


def wrap_text_preserve_newlines(text, width=700):
    '''
    wrap text to a certain width while preserving newlines
    '''
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    return '\n'.join(wrapped_lines)


def process_llm_response(llm_response):
    '''
    Process the response from the LLM model and return the answer
    ready to be displayed to the user.
    '''
    sources_used = ' \n'.join([
        f"{source.metadata['source'].split('/')[-1][:-4]} - page: {source.metadata['page']}"
        for source in llm_response['source_documents']
    ])

    ans = wrap_text_preserve_newlines(llm_response['result'])
    ans = f"{ans}\n\nSources:\n{sources_used}"

    pattern = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    index = ans.find(pattern)
    if index != -1:
        ans = ans[index + len(pattern):]

    return ans


def llm_ans(qa_chain, query):
    '''
    Get the answer from the LLM model and process it.
    '''
    start = time.time()
    llm_response = qa_chain.invoke(query)
    ans = process_llm_response(llm_response)
    end = time.time()
    time_elapsed = int(round(end - start, 0))
    return f"{ans}\n\nTime elapsed: {time_elapsed} s"


def login_to_huggingface():
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        token = input("Enter your Hugging Face token: ")

    try:
        login(token)
        print("Successfully logged in to Hugging Face!")
    except Exception as e:
        print(f"Failed to log in to Hugging Face. Error: {e}")
        return False

    return True
