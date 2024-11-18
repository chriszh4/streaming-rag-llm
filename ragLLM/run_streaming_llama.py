import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from ragLLM.rag_llm import RagLLM
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
#from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from langchain.embeddings import HuggingFaceBgeEmbeddings
#from llama_index.core import embeddings

from embedding import SimpleEmbedding

def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    rag_llm = RagLLM(model=model, tokenizer=tokenizer, kv_cache=kv_cache, max_gen_len=1000)
    #rag_llm.complete(prompts[0])

    # Define LlamaIndex parameters
    Settings.llm = rag_llm
    """
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    """
    Settings.embed_model = SimpleEmbedding()
    
    #Settings.embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
    #Settings.embed_model = OpenAIEmbedding()

    document = Document(text="Betsy is the nicest person who likes to eat strawberries. She does not like rotten foods")
    
    parser = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200
    )
    nodes = parser.get_nodes_from_documents([document])
    
    index = VectorStoreIndex(nodes)
    
    query_engine = index.as_query_engine(
        similarity_top_k=2,
        response_mode="compact"
    )

    # Query the index
    print(query_engine.query("What is Betsy's food preferences?"))
    
    

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
