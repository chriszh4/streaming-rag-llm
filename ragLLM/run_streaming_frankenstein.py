import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from ragLLM.rag_llm import RagLLM
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import FlatReader
#from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from langchain.embeddings import HuggingFaceBgeEmbeddings
#from llama_index.core import embeddings

from embedding import SimpleEmbedding

def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    rag_llm = RagLLM(model=model, tokenizer=tokenizer, kv_cache=kv_cache, max_gen_len=1000)

    # Define LlamaIndex parameters
    Settings.llm = rag_llm
    Settings.embed_model = SimpleEmbedding()

    # Answers to the QA pairs are unreleased
    novels_root_dir = "./eval_data/NovelQA/Books/PublicDomain"
    qa_root_dir = "./eval_data/NovelQA/Data/PublicDomain"

    demo_root_dir = "./eval_data/NovelQA/Demonstration"

    frankenstein_novel = os.path.join(demo_root_dir, "Frankenstein.txt")
    frankenstein_qa = os.path.join(demo_root_dir, "Frankenstein.json")

    questions = parse_questions(frankenstein_qa)

    documents = load_novel(frankenstein_novel)

    qa_template_str = """You are a literary analysis expert. Use the following excerpts from the novel to answer the multiple choice question. There is only one correct answer among the choices. 
        
        Relevant excerpts from the novel:
        {context_str}
        
        Question: {query_str}
        
        Instructions: There are four choices to this question. <choices>. Select the correct choice to the question out of these four given choices. Your output should be a single letter out of A, B, C, or D. Do not include any further explanation or other output."""

    run_query_engine(documents, qa_template_str, questions)
    
def load_novel(novel_path):
    parser = FlatReader()
    file_extractor = {".txt": parser}

    return SimpleDirectoryReader(
        input_files = [novel_path],
        file_extractor=file_extractor
    ).load_data()

def parse_questions(qa_path):
    with open(qa_path, "r") as qa_json:
        questions = json.load(qa_json)
        return questions

def run_query_engine(documents, qa_template_str, questions):
    parser = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    for idx, query in enumerate(questions):
        print(f"Question {idx + 1}:\n")
        question = query["Question"]
        answer = query["Answer"]
        choices = [f"{letter}: {choice}" for letter, choice in query["Options"].items()]
        choices = " Choices: " + "; ".join(choices)

        qa_template = PromptTemplate(qa_template_str.replace("<choices>", choices))

        query_engine = index.as_query_engine(
            similarity_top_k=10,
            text_qa_template=qa_template
        )

        response = query_engine.query(question).response
        PRINT_CONTEXT = True
        if not PRINT_CONTEXT:
            response_answer = response.split("Instructions:")[-1]
            print("Instructions:" + response_answer)
        else:
            print(response)
        print(f"The correct answer was {answer}")
        print("\n\n\n\n*****************************")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)

    args = parser.parse_args()

    main(args)
