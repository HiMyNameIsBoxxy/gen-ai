import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

class LLMGenerator:
    def __init__(self, model_path="finetuned_gpt2_rl"):
        """
        Loads the trained GPT-2 model and tokenizer.
        model_path should be either:
        - "finetuned_gpt2"
        - "finetuned_gpt2_rl"
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.model.eval()

    def generate(self, prompt: str, max_length: int = 100):
        """
        Generate text using the trained model.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
