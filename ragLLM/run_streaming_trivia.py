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
    evidence_root_dir = "./eval_data/TriviaQA/evidence/web"
    qa_root_dir = "./eval_data/TriviaQA/qa"

    web_qa = os.path.join(qa_root_dir, "verified-web-dev.json")

    print("Loading questions and evidence...")
    questions = parse_questions(web_qa)
    print(f"Loaded {len(questions)} questions")

    evidence_files = []
    for q in questions:
        if q["SearchResults"]:
            evidence_files.append(q["SearchResults"][0]["Filename"])
        else:
            assert(False)
    evidence_files = [os.path.join(evidence_root_dir, x) for x in evidence_files]

    documents = load_evidence(evidence_files)
    print(f"Loaded {len(documents)} documents")

    qa_template_str = """You are a trivia expert. Using your knowledge and the following evidence, please answer the question.
        
        Relevant evidence:
        {context_str}
        
        Question: {query_str}"""

    run_query_engine(documents, qa_template_str, questions)
    
def load_evidence(evidence_files):
    parser = FlatReader()
    file_extractor = {".txt": parser}

    return SimpleDirectoryReader(
        input_files=evidence_files,
        file_extractor=file_extractor,
    ).load_data()

def parse_questions(qa_path):
    with open(qa_path, "r") as qa_json:
        questions = json.load(qa_json)
        filtered_questions = [q for q in questions["Data"] if q["SearchResults"]]
        return filtered_questions

def run_query_engine(documents, qa_template_str, questions):
    CHUNK_SIZE = 1000
    SIMILARITY_TOP_K = 1
    parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=10
    )
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, show_progress=True)
    print("Press any key to continue...")
    input()
    answer_list = []
    for idx, query in enumerate(questions):
        print(f"Question {idx + 1}:\n")
        question = query["Question"]
        answer = query["Answer"]

        qa_template = PromptTemplate(qa_template_str)

        query_engine = index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K,
            text_qa_template=qa_template
        )

        response = query_engine.query(question).response
        PRINT_CONTEXT = True
        if not PRINT_CONTEXT:
            response_answer = response.split("Instructions:")[-1]
            print("Instructions:" + response_answer)
        else:
            response_answer = response
            print(response)
        print(f"\nThe correct answer was {answer}")
        print("\n\n\n\n*****************************")
        extracted_answer = response_answer.split("ASSISTANT:")[-1]
        answer_list.append(extracted_answer)

    # Save extracted answers to file as json
    with open(f"ragLLM/eval_results/bge_rag_extracted_answers_{CHUNK_SIZE}_{SIMILARITY_TOP_K}.json", "w") as f:
        json.dump(answer_list, f)


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
