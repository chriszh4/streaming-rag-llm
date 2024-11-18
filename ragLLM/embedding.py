from typing import Any, List
from transformers import AutoTokenizer, AutoModel
import torch
from llama_index.core.embeddings import BaseEmbedding


class SimpleEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()  # Set the model to evaluation mode

    def _get_query_embedding(self, query: str) -> List[float]:
        """Embed a single query."""
        tokens = self._tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self._model(**tokens)
        # Use the mean of the last hidden states as the embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        """Embed a single text."""
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self._get_query_embedding(text))
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of embedding a single query."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of embedding a single text."""
        return self._get_text_embedding(text)
