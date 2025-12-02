import os
import torch
from torch.optim import AdamW
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_gpt2(model, dataloader, epochs=1, lr=5e-5):
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    model.to(device)

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())


def save_model(model, tokenizer, output_dir="finetuned_gpt2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")
