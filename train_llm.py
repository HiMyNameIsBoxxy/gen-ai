from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from helper_lib.llm_dataset import load_qa_dataset
from helper_lib.llm_tokenizer import get_tokenizer, tokenize_dataset
from helper_lib.llm_trainer import train_gpt2, save_model


def main():
    print("Loading dataset...")
    train_ds, _ = load_qa_dataset()

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Tokenizing...")
    train_tok = tokenize_dataset(train_ds, tokenizer)

    print("Building DataLoader...")
    train_loader = DataLoader(train_tok, batch_size=4, shuffle=True)

    print("Loading GPT-2 base model...")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    print("Training...")
    train_gpt2(model, train_loader, epochs=1)

    print("Saving...")
    save_model(model, tokenizer)


if __name__ == "__main__":
    main()
