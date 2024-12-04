# StreamingLLM with Retrieval-Augmented Generation
https://github.com/user-attachments/assets/6279cb3c-ec01-4909-93a8-b22260de8e8b

## TL;DR
We adapt StreamingLLM with RAG to support effective infinite context in streaming applications.

## Abstract
TODO

## Usage

### Environment Setup

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```

### Run RAGStreamingLLM on TriviaQA

```bash
CUDA_VISIBLE_DEVICES=0 python ragLLM/run_streaming_trivia.py  --enable_streaming --model_name_or_path="lmsys/vicuna-7b-v1.5"
```
