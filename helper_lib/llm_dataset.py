import datasets

def load_qa_dataset():
    # Load the RLHF Nectar dataset (only has a train split)
    dataset = datasets.load_dataset("berkeley-nest/Nectar")

    full_train = dataset["train"]

    # 90/10 split
    split = full_train.train_test_split(test_size=0.1, seed=42)
    train = split["train"]
    test = split["test"]

    # Format: pick best answer (rank=1.0) and combine with prompt
    def format_qa(example):
        prompt = example["prompt"].strip()

        # Select answer with minimum rank = best answer
        answers = example["answers"]
        best_answer = min(answers, key=lambda x: x["rank"])["answer"].strip()

        example["text"] = f"Question: {prompt}\nAnswer: {best_answer}"
        return example

    train = train.map(format_qa)
    test = test.map(format_qa)

    return train, test
