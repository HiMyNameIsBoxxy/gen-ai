import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_reward(generated_text):
    reward = 0.0

    # Exact behavior (full reward)
    if generated_text.startswith("That is a great question"):
        reward += 1.0
    else:
        # Partial credit—model learns faster
        if "great question" in generated_text:
            reward += 0.5
        if "question" in generated_text:
            reward += 0.2

    if generated_text.endswith("let me know if you have any other questions."):
        reward += 1.0
    else:
        if "other questions" in generated_text:
            reward += 0.5
        if "questions" in generated_text:
            reward += 0.2

    return reward



# Generate text given a prompt
def generate_text(model, tokenizer, prompt, max_len=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        max_length=max_len,
        do_sample=True,
        temperature=0.9,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# Reinforcement Learning (REINFORCE-style training)
def rl_train(model, tokenizer, epochs=100, lr=1e-5):

    optimizer = Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    prompt = "Explain how airplanes fly."

    for epoch in range(epochs):
        # Generate text
        generated_text = generate_text(model, tokenizer, prompt)

        # Compute reward
        reward = compute_reward(generated_text)

        # Compute log-probabilities of the generated tokens
        inputs = tokenizer(generated_text, return_tensors="pt").to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # REINFORCE objective = -reward * log_prob
        rl_loss = -reward * loss
        rl_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"[Epoch {epoch+1}] Reward={reward:.2f}  Loss={rl_loss.item():.4f}")

    print("Saving RL-trained model...")
    model.save_pretrained("finetuned_gpt2_rl")
    tokenizer.save_pretrained("finetuned_gpt2_rl")
    print("RL-trained model saved successfully.")