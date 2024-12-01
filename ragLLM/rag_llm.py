from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
import torch 
from typing import Optional, List, Mapping, Any
from pydantic import BaseModel, Field
from llama_index.core.llms.callbacks import llm_completion_callback

class RagLLM(CustomLLM):
    model: Any = Field(..., description="The model object")
    tokenizer: Any = Field(..., description="The tokenizer object")
    kv_cache: Optional[Any] = Field(None, description="Key-value cache for the model")
    max_gen_len: int = Field(1000, description="Maximum generation length")
    
    """
    def __init__(self, model, tokenizer, kv_cache=None, max_gen_len=1000):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache
        self.max_gen_len = max_gen_len
    """

    @property
    def metadata(self):
        return LLMMetadata(
            context_window=4096,
            num_output=self.max_gen_len,
            #model_name=self.model_name,
        )

    @torch.no_grad()
    def greedy_generate(self, model, tokenizer, input_ids, past_key_values, max_gen_len):
        output_str = ""
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids = [pred_token_idx.item()]
        pos = 0
        for _ in range(max_gen_len - 1):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids.append(pred_token_idx.item())
            generated_text = (
                tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
            )

            now = len(generated_text) - 1
            if now > pos:
                output_str += " ".join(generated_text[pos:now]) + " "
                pos = now

            if pred_token_idx == tokenizer.eos_token_id:
                break
        output_str += " ".join(generated_text[pos:])
        return past_key_values, output_str

    def clean_prompt(self, prompt):
        lines = prompt.split('\n')
        cleaned_lines = [line for line in lines if not any(
            field in line.lower() for field in ['filename:', 'extension:', 'file_path:']
        )]
        return '\n'.join(cleaned_lines)
        

    @torch.no_grad()
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any):
        use_rag = False
        if use_rag:
            prompt = self.clean_prompt(prompt)
        else:
            prompt = prompt.split("Question:")[-1]
            prompt = "Question: " + prompt

        past_key_values = None
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        output_base = "\n" + prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        seq_len = input_ids.shape[1]
        if self.kv_cache is not None:
            space_needed = seq_len + self.max_gen_len
            past_key_values = self.kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values, output_str = self.greedy_generate(
            self.model, self.tokenizer, input_ids, past_key_values, self.max_gen_len
        )
        return CompletionResponse(text=output_base + output_str)
    
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str
    ) -> CompletionResponseGen:
        yield "HUH"