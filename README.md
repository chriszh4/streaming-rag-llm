# StreamingLLM with Retrieval-Augmented Generation
https://github.com/user-attachments/assets/6279cb3c-ec01-4909-93a8-b22260de8e8b

## TL;DR
We adapt StreamingLLM with RAG to support effective infinite context in streaming applications.

## Abstract
Ensuring continuity in streaming applications for large language models (LLMs) remains a significant challenge, even with advancements like StreamingLLM, which are constrained by a finite context window. We propose extending this capacity by integrating Retrieval-Augmented Generation (RAG) to enable persistent memory and dynamic context retrieval. Our approach results in an LLM capable of retaining and utilizing all past information, addressing the limitations of fixed context windows. To evaluate this enhanced capability, we conduct experiments on a question-answering benchmark, TriviaQA, and present discussions on limitations and performance on a new long-context benchmark, NovelQA. This work highlights the potential of RAG-augmented LLMs in long-context applications and real-world deployments requiring sustained coherence and memory.

## Usage

### Environment Setup

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece llama-index

python setup.py develop
```

### Run RAGStreamingLLM on TriviaQA

```bash
CUDA_VISIBLE_DEVICES=0 python ragLLM/run_streaming_trivia.py  --enable_streaming --model_name_or_path="lmsys/vicuna-7b-v1.5"
```
