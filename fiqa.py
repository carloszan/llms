from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from tqdm import tqdm
import torch


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 0
    elif 'negative' in x or 'Negative' in x:
        return 2
    else:
        return 1


def test_fiqa(model, tokenizer, batch_size=8):
    dataset = load_dataset('pauri32/fiqa-2018')
    dataset = dataset["test"]
    dataset = dataset.to_pandas()

    # only to validate function
    dataset = dataset.head(2)

    dataset["output"] = dataset['label']
    dataset[
        "instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."

    dataset = dataset[['sentence', 'output', 'instruction']]
    dataset.columns = ['input', 'output', 'instruction']
    dataset[['context', 'target']] = dataset.apply(
        format_example, axis=1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][1]}\n\n")

    context = dataset['context'].tolist()
    total_steps = dataset.shape[0]//batch_size + 1
    print(
        f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")

    out_text = []

    for i in tqdm(range(total_steps)):
        tmp_context = context[i * batch_size:(i+1) * batch_size]
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()

        res = model.generate(**tokens)
        res_sentences = tokenizer.batch_decode(res)
        print(f"Context: {res_sentences}")
        out_text = [o.split("Answer: ")[1] for o in res_sentences]
        torch.cuda.empty_cache()

    dataset["out_text"] = out_text
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["target"], dataset["new_out"])
    f1_macro = f1_score(dataset["target"], dataset["new_out"], average="macro")
    f1_micro = f1_score(dataset["target"], dataset["new_out"], average="micro")
    f1_weighted = f1_score(
        dataset["target"], dataset["new_out"], average="weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return dataset
