from transformers import AutoTokenizer, AutoModelForCausalLM
from helper_lib.llm_rl_trainer import rl_train

def main():
    print("Loading fine-tuned model...")
    model = AutoModelForCausalLM.from_pretrained("finetuned_gpt2")
    tokenizer = AutoTokenizer.from_pretrained("finetuned_gpt2")

    print("Running RL post-training...")
    rl_train(model, tokenizer, epochs=50)

    print("Saving RL-trained model...")
    model.save_pretrained("rl_finetuned_gpt2")
    tokenizer.save_pretrained("rl_finetuned_gpt2")

if __name__ == "__main__":
    main()
